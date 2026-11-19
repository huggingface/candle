use super::{
    BarrierIndex, ClearAttachmentIndex, ExecutableTaskGraph, Instruction, NodeIndex,
    RenderPassIndex, ResourceAccess, SemaphoreIndex,
};
use crate::{
    assert_unsafe_precondition,
    collector::{Deferred, DeferredBatch},
    command_buffer::{CommandBufferState, RecordingCommandBuffer},
    linear_map::LinearMap,
    resource::{
        BufferAccess, BufferState, Flight, GlobalLockSharedGuard, ImageAccess, ImageState,
        Resources, SwapchainState, SwapchainSyncStage,
    },
    ClearValues, Id, InvalidSlotError, ObjectType, TaskContext, TaskError,
};
use ash::vk::{self, TaggedStructure};
use concurrent_slotmap::hyaline;
use smallvec::{smallvec, SmallVec};
use std::{error::Error, fmt, iter, mem, ops::Range, ptr, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferMemory},
    command_buffer as raw,
    device::{Device, DeviceOwned, Queue},
    format::ClearValue,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageSubresourceRange,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::Swapchain,
    sync::{
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::Semaphore,
        AccessFlags, PipelineStages,
    },
    Validated, Version, VulkanError, VulkanObject,
};

impl<W: ?Sized + 'static> ExecutableTaskGraph<W> {
    /// Executes the next frame of the [flight] given by `flight_id`, panicking on a validation
    /// error.
    ///
    /// This is a shortcut for `try_execute().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - There must be no other task graphs executing that access any of the same subresources as
    ///   `self`.
    /// - A subresource in flight must not be accessed in more than one frame in flight.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_execute`] returns a [`ValidationError`].
    /// - Panics if `resource_map` doesn't map the virtual resources of `self` exhaustively.
    /// - Panics if `self.flight_id()` is invalid.
    /// - Panics if another thread is already executing a task graph using the flight.
    /// - Panics if another thread is already executing a task graph using any of the swapchains
    ///   used by the task graph.
    /// - Panics if any swapchain used by the task graph has already been recreated or removed.
    /// - Panics if any swapchain used by the task graph is being recreated on another thread.
    /// - Panics if the oldest frame of the flight wasn't [waited] on.
    ///
    /// [`try_execute`]: Self::try_execute
    /// [waited]: crate::resource::Flight::wait
    #[track_caller]
    pub unsafe fn execute(
        &self,
        resource_map: ResourceMap<'_>,
        world: &W,
        pre_present_notify: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        match unsafe { self.try_execute(resource_map, world, pre_present_notify) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Executes the next frame of the [flight] given by `flight_id`.
    ///
    /// # Safety
    ///
    /// - There must be no other task graphs executing that access any of the same subresources as
    ///   `self`.
    /// - A subresource in flight must not be accessed in more than one frame in flight.
    ///
    /// # Panics
    ///
    /// - Panics if `resource_map` doesn't map the virtual resources of `self` exhaustively.
    /// - Panics if `self.flight_id()` is invalid.
    /// - Panics if another thread is already executing a task graph using the flight.
    /// - Panics if another thread is already executing a task graph using any of the swapchains
    ///   used by the task graph.
    /// - Panics if any swapchain used by the task graph has already been recreated or removed.
    /// - Panics if any swapchain used by the task graph is being recreated on another thread.
    /// - Panics if the oldest frame of the flight wasn't [waited] on.
    ///
    /// [waited]: crate::resource::Flight::wait
    #[track_caller]
    pub unsafe fn try_execute(
        &self,
        resource_map: ResourceMap<'_>,
        world: &W,
        pre_present_notify: impl FnOnce(),
    ) -> Result<(), Validated<ExecuteError>> {
        assert!(ptr::eq(
            resource_map.virtual_resources,
            &self.graph.resources,
        ));
        assert!(resource_map.is_exhaustive());

        let Ok(global_lock_guard) = resource_map.physical_resources.lock_global_shared() else {
            panic!("a task graph cannot be executed from within a task");
        };
        let mut global_lock_guard = Some(global_lock_guard);

        let flight_id = self.flight_id;

        let flight = resource_map
            .physical_resources
            .try_flight_protected(flight_id, &resource_map.guard)
            .expect("invalid flight ID");

        let Some(_flight_lock_guard) = flight.try_lock() else {
            panic!(
                "another thread is already executing a task graph using the flight {flight_id:?}",
            );
        };

        assert!(
            flight.is_oldest_frame_complete(),
            "you must wait on the oldest frame in flight to finish before submitting more work",
        );

        let _resource_lock_guard = unsafe { self.lock_resources(&resource_map) };

        let current_frame_index = flight.current_frame_index();

        unsafe { self.collect_swapchains(&resource_map) }?;

        unsafe { self.acquire_images_khr(&resource_map) }?;

        let mut current_fence = flight.current_fence().write();

        // SAFETY: We checked that the fence has been signaled.
        unsafe { current_fence.reset_unchecked() }.map_err(ExecuteError::VulkanError)?;

        unsafe { self.invalidate_mapped_memory_ranges(&resource_map) }?;

        unsafe { self.create_framebuffers(&resource_map) }?;

        unsafe { flight.start_next_frame() };

        let mut deferred_batch = Some(resource_map.resources().create_deferred_batch());
        let mut state_guard = StateGuard {
            executable: self,
            resource_map: &resource_map,
            global_lock_guard: &mut global_lock_guard,
            flight,
            current_fence: &mut current_fence,
            deferred_batch: &mut deferred_batch,
            submission_count: 0,
        };
        let deferreds = state_guard.deferred_batch.as_mut().unwrap().deferreds_mut();

        let execute_instructions = if self.device().enabled_features().synchronization2 {
            Self::execute_instructions2
        } else {
            Self::execute_instructions
        };

        unsafe {
            execute_instructions(
                self,
                &resource_map,
                deferreds,
                current_frame_index,
                state_guard.current_fence,
                &mut state_guard.submission_count,
                world,
            )
        }?;

        self.last_frame.set(Some(flight.current_frame()));

        unsafe { flight.next_frame() };

        pre_present_notify();

        if let Err(err) = unsafe { self.present_images_khr(&resource_map) } {
            if matches!(err, Validated::Error(ExecuteError::Swapchain { error, .. }) if !matches!(
                error,
                VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory,
            )) {
                // Any `vkQueuePresentKHR` error code except `VK_ERROR_OUT_OF_HOST_MEMORY` and
                // `VK_ERROR_OUT_OF_DEVICE_MEMORY` means that the semaphore wait operation was
                // executed, so we don't have to handle the failure. Otherwise, either one of those
                // two errors or another variant of `ExecuteError` is returned and the state guard
                // runs, which ensures that our swapchain state stays intact.
                mem::forget(state_guard);
            }

            return Err(err);
        }

        mem::forget(state_guard);

        unsafe { self.update_resource_state(&resource_map, &self.last_accesses) };

        let deferred_batch = deferred_batch.take().unwrap();

        // SAFETY: We only defer the destruction of objects that are frame-local.
        unsafe { deferred_batch.enqueue_with_flights(iter::once(self.flight_id)) };

        Ok(())
    }

    #[track_caller]
    unsafe fn lock_resources<'a>(
        &'a self,
        resource_map: &'a ResourceMap<'a>,
    ) -> ResourceLockGuard<'a> {
        for (index, &swapchain_id) in self.swapchains.iter().enumerate() {
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };

            match swapchain_state.try_execute() {
                Ok(()) => {}
                Err(err) => {
                    for &swapchain_id in &self.swapchains[..index] {
                        let swapchain_state =
                            unsafe { resource_map.swapchain_unchecked(swapchain_id) };

                        // SAFETY: We locked these swapchains in the outer loop.
                        unsafe { swapchain_state.unlock() };
                    }

                    panic!("failed to lock swapchain {swapchain_id:?} for execution: {err}");
                }
            }
        }

        // SAFETY: We successfully locked all the swapchains.
        unsafe { ResourceLockGuard::new(resource_map, &self.swapchains) }
    }

    unsafe fn collect_swapchains(&self, resource_map: &ResourceMap<'_>) -> Result {
        for &swapchain_id in &self.swapchains {
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };
            unsafe { swapchain_state.collect() }.map_err(ExecuteError::VulkanError)?;
        }

        Ok(())
    }

    unsafe fn acquire_images_khr(&self, resource_map: &ResourceMap<'_>) -> Result {
        for &swapchain_id in &self.swapchains {
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };

            // Make sure to not acquire another image index if we already acquired one. This can
            // happen when using multiple swapchains, if one acquire succeeds and another fails, or
            // when executing a submission or presenting an image fails.
            if swapchain_state.is_image_acquired() {
                continue;
            }

            if let Err(error) =
                unsafe { swapchain_state.acquire_next_image(self.use_swapchain_maintenance1) }
            {
                return Err(Validated::Error(ExecuteError::Swapchain {
                    swapchain_id,
                    error,
                }));
            }
        }

        Ok(())
    }

    unsafe fn invalidate_mapped_memory_ranges(&self, resource_map: &ResourceMap<'_>) -> Result {
        let mut mapped_memory_ranges = Vec::new();

        for &buffer_id in &self.graph.resources.host_reads {
            let buffer = unsafe { resource_map.buffer_unchecked(buffer_id) }.buffer();

            let allocation = match buffer.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse => todo!("`TaskGraph` doesn't support sparse binding yet"),
                BufferMemory::External => continue,
                _ => unreachable!(),
            };

            if allocation.atom_size().is_none() {
                continue;
            }

            if unsafe { allocation.mapped_slice_unchecked(..) }.is_err() {
                continue;
            }

            // This works because the memory allocator must align allocations to the non-coherent
            // atom size when the memory is host-visible but not host-coherent.
            mapped_memory_ranges.push(
                vk::MappedMemoryRange::default()
                    .memory(allocation.device_memory().handle())
                    .offset(allocation.offset())
                    .size(allocation.size()),
            );
        }

        if !mapped_memory_ranges.is_empty() {
            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.invalidate_mapped_memory_ranges)(
                    self.device().handle(),
                    mapped_memory_ranges.len() as u32,
                    mapped_memory_ranges.as_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)
            .map_err(ExecuteError::VulkanError)?;
        }

        Ok(())
    }

    unsafe fn create_framebuffers(&self, resource_map: &ResourceMap<'_>) -> Result {
        let mut batch = None;

        for render_pass_state in self.render_passes.borrow_mut().iter_mut() {
            // If we're executing this task graph for the first time, there are no framebuffers
            // yet, so we need to create them.
            if render_pass_state.framebuffers.is_empty() {
                unsafe {
                    create_framebuffers(
                        &render_pass_state.render_pass,
                        &render_pass_state.attachments,
                        &mut render_pass_state.framebuffers,
                        &self.swapchains,
                        resource_map,
                    )
                }?;
                continue;
            }

            // Otherwise we need to recreate the framebuffers for a given render pass if any of its
            // attachments changed.
            for (&id, attachment) in render_pass_state
                .attachments
                .keys()
                .zip(render_pass_state.framebuffers[0].attachments())
            {
                let image = match id.object_type() {
                    ObjectType::Image => {
                        let image_id = unsafe { id.parametrize() };

                        unsafe { resource_map.image_unchecked(image_id) }.image()
                    }
                    ObjectType::Swapchain => {
                        let swapchain_id = unsafe { id.parametrize() };

                        &unsafe { resource_map.swapchain_unchecked(swapchain_id) }.images()[0]
                    }
                    _ => unreachable!(),
                };

                if attachment.image() != image {
                    let batch = batch.get_or_insert_with(|| {
                        resource_map.physical_resources.create_deferred_batch()
                    });

                    for framebuffer in render_pass_state.framebuffers.drain(..) {
                        batch.destroy_object(framebuffer);
                    }

                    unsafe {
                        create_framebuffers(
                            &render_pass_state.render_pass,
                            &render_pass_state.attachments,
                            &mut render_pass_state.framebuffers,
                            &self.swapchains,
                            resource_map,
                        )
                    }?;

                    break;
                }
            }
        }

        if let Some(batch) = batch {
            let last_frame = self.last_frame.get().unwrap();

            // SAFETY: We only destroy objects that are graph-local and `last_frame` is the last
            // frame in which they were used.
            unsafe { batch.enqueue_with_frames(iter::once((self.flight_id, last_frame))) };
        }

        Ok(())
    }

    unsafe fn execute_instructions2(
        &self,
        resource_map: &ResourceMap<'_>,
        deferreds: &mut Vec<Deferred>,
        current_frame_index: u32,
        current_fence: &Fence,
        submission_count: &mut usize,
        world: &W,
    ) -> Result {
        let mut state = ExecuteState2::new(
            self,
            resource_map,
            deferreds,
            current_frame_index,
            current_fence,
            submission_count,
            world,
        )?;
        let mut execute_initial_barriers = true;

        for instruction in self.instructions.iter().cloned() {
            if execute_initial_barriers {
                let submission = current_submission!(state);
                state.initial_pipeline_barrier(submission.initial_barrier_range.clone());
                execute_initial_barriers = false;
            }

            match instruction {
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_acquire(swapchain_id, stage_mask);
                }
                Instruction::WaitSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.wait_semaphore(semaphore_index, stage_mask);
                }
                Instruction::ExecuteTask { node_index } => {
                    state.execute_task(node_index)?;
                }
                Instruction::PipelineBarrier { barrier_range } => {
                    state.pipeline_barrier(barrier_range)?;
                }
                Instruction::BeginRenderPass { render_pass_index } => {
                    state.begin_render_pass(render_pass_index)?;
                }
                Instruction::NextSubpass => {
                    state.next_subpass()?;
                }
                Instruction::EndRenderPass => {
                    state.end_render_pass();
                }
                Instruction::ClearAttachments {
                    node_index,
                    render_pass_index,
                    clear_attachment_range,
                } => {
                    state.clear_attachments(
                        node_index,
                        render_pass_index,
                        clear_attachment_range,
                    )?;
                }
                Instruction::SignalSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.signal_semaphore(semaphore_index, stage_mask);
                }
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_pre_present(swapchain_id, stage_mask)?;
                }
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_pre_present(swapchain_id, stage_mask);
                }
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_present(swapchain_id, stage_mask)?;
                }
                Instruction::FlushSubmit => {
                    state.flush_submit()?;
                }
                Instruction::Submit => {
                    state.submit()?;
                    execute_initial_barriers = true;
                }
            }
        }

        Ok(())
    }

    unsafe fn execute_instructions(
        &self,
        resource_map: &ResourceMap<'_>,
        deferreds: &mut Vec<Deferred>,
        current_frame_index: u32,
        current_fence: &Fence,
        submission_count: &mut usize,
        world: &W,
    ) -> Result {
        let mut state = ExecuteState::new(
            self,
            resource_map,
            deferreds,
            current_frame_index,
            current_fence,
            submission_count,
            world,
        )?;
        let mut execute_initial_barriers = true;

        for instruction in self.instructions.iter().cloned() {
            if execute_initial_barriers {
                let submission = current_submission!(state);
                state.initial_pipeline_barrier(submission.initial_barrier_range.clone());
                execute_initial_barriers = false;
            }

            match instruction {
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_acquire(swapchain_id, stage_mask);
                }
                Instruction::WaitSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.wait_semaphore(semaphore_index, stage_mask);
                }
                Instruction::ExecuteTask { node_index } => {
                    state.execute_task(node_index)?;
                }
                Instruction::PipelineBarrier { barrier_range } => {
                    state.pipeline_barrier(barrier_range)?;
                }
                Instruction::BeginRenderPass { render_pass_index } => {
                    state.begin_render_pass(render_pass_index)?;
                }
                Instruction::NextSubpass => {
                    state.next_subpass()?;
                }
                Instruction::EndRenderPass => {
                    state.end_render_pass();
                }
                Instruction::ClearAttachments {
                    node_index,
                    render_pass_index,
                    clear_attachment_range,
                } => {
                    state.clear_attachments(
                        node_index,
                        render_pass_index,
                        clear_attachment_range,
                    )?;
                }
                Instruction::SignalSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.signal_semaphore(semaphore_index, stage_mask);
                }
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_pre_present(swapchain_id, stage_mask)?;
                }
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_pre_present(swapchain_id, stage_mask);
                }
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_present(swapchain_id, stage_mask)?;
                }
                Instruction::FlushSubmit => {
                    state.flush_submit()?;
                }
                Instruction::Submit => {
                    state.submit()?;
                    execute_initial_barriers = true;
                }
            }
        }

        Ok(())
    }

    unsafe fn flush_mapped_memory_ranges(&self, resource_map: &ResourceMap<'_>) -> Result {
        let mut mapped_memory_ranges = Vec::new();

        for &buffer_id in &self.graph.resources.host_writes {
            let buffer = unsafe { resource_map.buffer_unchecked(buffer_id) }.buffer();

            let allocation = match buffer.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse => todo!("`TaskGraph` doesn't support sparse binding yet"),
                BufferMemory::External => continue,
                _ => unreachable!(),
            };

            if allocation.atom_size().is_none() {
                continue;
            }

            if unsafe { allocation.mapped_slice_unchecked(..) }.is_err() {
                continue;
            }

            // This works because the memory allocator must align allocations to the non-coherent
            // atom size when the memory is host-visible but not host-coherent.
            mapped_memory_ranges.push(
                vk::MappedMemoryRange::default()
                    .memory(allocation.device_memory().handle())
                    .offset(allocation.offset())
                    .size(allocation.size()),
            );
        }

        if !mapped_memory_ranges.is_empty() {
            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.flush_mapped_memory_ranges)(
                    self.device().handle(),
                    mapped_memory_ranges.len() as u32,
                    mapped_memory_ranges.as_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)
            .map_err(ExecuteError::VulkanError)?;
        }

        Ok(())
    }

    unsafe fn present_images_khr(&self, resource_map: &ResourceMap<'_>) -> Result {
        let Some(present_queue) = &self.present_queue else {
            return Ok(());
        };

        if self.swapchains.is_empty() {
            return Ok(());
        }

        let swapchain_count = self.swapchains.len();
        let mut semaphores_vk = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut swapchains_vk = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut image_indices = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut results_vk = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut fences_vk = SmallVec::<[_; 1]>::with_capacity(swapchain_count);

        for &swapchain_id in &self.swapchains {
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };
            let semaphore_vk = unsafe { swapchain_state.current_present_semaphore() }.unwrap();
            semaphores_vk.push(semaphore_vk);
            swapchains_vk.push(swapchain_state.swapchain().handle());
            image_indices.push(swapchain_state.current_image_index().unwrap());
            results_vk.push(vk::Result::SUCCESS);

            if self.use_swapchain_maintenance1 {
                let fence_vk = unsafe { swapchain_state.init_present_fence() }
                    .map_err(ExecuteError::VulkanError)?;
                fences_vk.push(fence_vk);
            }
        }

        let mut present_info_vk = vk::PresentInfoKHR::default()
            .wait_semaphores(&semaphores_vk)
            .swapchains(&swapchains_vk)
            .image_indices(&image_indices)
            .results(&mut results_vk);

        let mut fence_info_vk = self
            .use_swapchain_maintenance1
            .then(|| vk::SwapchainPresentFenceInfoEXT::default().fences(&fences_vk));

        if let Some(fence_info_vk) = fence_info_vk.as_mut() {
            present_info_vk = present_info_vk.push(fence_info_vk);
        }

        let fns = self.device().fns();
        let queue_present_khr = fns.khr_swapchain.queue_present_khr;
        let _ = unsafe { queue_present_khr(present_queue.handle(), &present_info_vk) };

        let mut res = Ok(());

        for (&result, &swapchain_id) in results_vk.iter().zip(&self.swapchains) {
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };

            // In case of these error codes, the semaphore wait operation is not executed.
            if !matches!(
                result,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ) {
                unsafe { swapchain_state.handle_presentation() };
            }

            // TODO: Could there be a use case for keeping the old image contents?
            unsafe { swapchain_state.set_access(ImageAccess::NONE) };

            if !matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR) {
                // Return the first error for consistency with the acquisition logic.
                if res.is_ok() {
                    res = Err(Validated::Error(ExecuteError::Swapchain {
                        swapchain_id,
                        error: result.into(),
                    }));
                }
            }
        }

        res
    }

    unsafe fn update_resource_state(
        &self,
        resource_map: &ResourceMap<'_>,
        last_accesses: &[ResourceAccess],
    ) {
        for (id, _) in self.graph.resources.iter() {
            let access = last_accesses[id.index() as usize];

            match id.object_type() {
                ObjectType::Buffer => {
                    let id = unsafe { id.parametrize() };
                    let state = unsafe { resource_map.buffer_unchecked(id) };
                    let access = BufferAccess::from_masks(
                        access.stage_mask,
                        access.access_mask,
                        access.queue_family_index,
                    );
                    unsafe { state.set_access(access) };
                }
                ObjectType::Image => {
                    let id = unsafe { id.parametrize() };
                    let state = unsafe { resource_map.image_unchecked(id) };
                    let access = ImageAccess::from_masks(
                        access.stage_mask,
                        access.access_mask,
                        access.image_layout,
                        access.queue_family_index,
                    );
                    unsafe { state.set_access(access) };
                }
                ObjectType::Swapchain => {
                    let id = unsafe { id.parametrize() };
                    let state = unsafe { resource_map.swapchain_unchecked(id) };
                    let access = ImageAccess::from_masks(
                        access.stage_mask,
                        access.access_mask,
                        access.image_layout,
                        access.queue_family_index,
                    );
                    unsafe { state.set_access(access) };
                }
                _ => unreachable!(),
            }
        }
    }
}

struct ResourceLockGuard<'a> {
    resource_map: &'a ResourceMap<'a>,
    swapchains: &'a [Id<Swapchain>],
}

impl<'a> ResourceLockGuard<'a> {
    unsafe fn new(resource_map: &'a ResourceMap<'a>, swapchains: &'a [Id<Swapchain>]) -> Self {
        ResourceLockGuard {
            resource_map,
            swapchains,
        }
    }
}

impl Drop for ResourceLockGuard<'_> {
    fn drop(&mut self) {
        for &swapchain_id in self.swapchains {
            // SAFETY: Enforced by the caller of `ResourceLockGuard::new`.
            let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };

            // SAFETY: Enforced by the caller of `ResourceLockGuard::new`.
            unsafe { swapchain_state.unlock() };
        }
    }
}

unsafe fn create_framebuffers(
    render_pass: &Arc<RenderPass>,
    attachments: &LinearMap<Id, super::AttachmentState>,
    framebuffers: &mut Vec<Arc<Framebuffer>>,
    swapchains: &[Id<Swapchain>],
    resource_map: &ResourceMap<'_>,
) -> Result {
    debug_assert!(framebuffers.is_empty());

    let swapchain_image_counts = swapchains
        .iter()
        .map(|&id| {
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(id) };

            swapchain_state.swapchain().image_count()
        })
        .collect::<SmallVec<[_; 1]>>();
    let mut swapchain_image_indices: SmallVec<[u32; 1]> =
        smallvec![0; swapchain_image_counts.len()];

    framebuffers.reserve_exact(swapchain_image_counts.iter().product::<u32>() as usize);

    'outer: loop {
        let attachments = attachments
            .iter()
            .map(|(&id, attachment)| {
                let image = match id.object_type() {
                    ObjectType::Image => {
                        let image_id = unsafe { id.parametrize() };

                        unsafe { resource_map.image_unchecked(image_id) }.image()
                    }
                    ObjectType::Swapchain => {
                        let swapchain_id = unsafe { id.parametrize() };
                        let swapchain_state =
                            unsafe { resource_map.swapchain_unchecked(swapchain_id) };
                        let i = swapchains.iter().position(|x| x.erase() == id).unwrap();
                        let image_index = swapchain_image_indices[i];

                        &swapchain_state.images()[image_index as usize]
                    }
                    _ => unreachable!(),
                };

                ImageView::try_new(
                    image,
                    &ImageViewCreateInfo {
                        format: attachment.format,
                        component_mapping: attachment.component_mapping,
                        subresource_range: ImageSubresourceRange {
                            aspects: attachment.format.aspects(),
                            base_mip_level: attachment.mip_level,
                            level_count: Some(1),
                            base_array_layer: attachment.base_array_layer,
                            // FIXME:
                            layer_count: Some(1),
                        },
                        ..Default::default()
                    },
                )
                // FIXME:
                .map_err(Validated::unwrap)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ExecuteError::VulkanError)?;

        let framebuffer = Framebuffer::try_new(
            render_pass,
            &FramebufferCreateInfo {
                attachments: &attachments.iter().collect::<Vec<_>>(),
                ..Default::default()
            },
        )
        // FIXME:
        .map_err(Validated::unwrap)
        .map_err(ExecuteError::VulkanError)?;

        framebuffers.push(framebuffer);

        let mut i = 0;

        loop {
            let Some(image_index) = swapchain_image_indices.get_mut(i) else {
                break 'outer;
            };

            *image_index += 1;

            if *image_index == swapchain_image_counts[i] {
                *image_index = 0;
                i += 1;
            } else {
                break;
            }
        }
    }

    Ok(())
}

struct ExecuteState2<'a, W: ?Sized + 'static> {
    executable: &'a ExecutableTaskGraph<W>,
    resource_map: &'a ResourceMap<'a>,
    deferreds: &'a mut Vec<Deferred>,
    current_frame_index: u32,
    current_fence: &'a Fence,
    submission_count: &'a mut usize,
    world: &'a W,
    cmd_pipeline_barrier2: vk::PFN_vkCmdPipelineBarrier2,
    queue_submit2: vk::PFN_vkQueueSubmit2,
    per_submits: SmallVec<[PerSubmitInfo2; 4]>,
    current_per_submit: PerSubmitInfo2,
    current_command_buffer: Option<raw::RecordingCommandBuffer>,
    command_buffer_state: CommandBufferState,
    command_buffers: Vec<Arc<raw::CommandBuffer>>,
    current_buffer_barriers_vk: Vec<vk::BufferMemoryBarrier2<'static>>,
    current_image_barriers_vk: Vec<vk::ImageMemoryBarrier2<'static>>,
    current_render_pass_index: RenderPassIndex,
    current_framebuffer_index: usize,
    current_subpass_index: usize,
    clear_values: LinearMap<Id, Option<ClearValue>>,
    clear_values_vk: Vec<vk::ClearValue>,
    clear_attachments_vk: Vec<vk::ClearAttachment>,
}

#[derive(Default)]
struct PerSubmitInfo2 {
    wait_semaphore_infos_vk: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 4]>,
    command_buffer_infos_vk: SmallVec<[vk::CommandBufferSubmitInfo<'static>; 1]>,
    signal_semaphore_infos_vk: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 4]>,
}

impl<'a, W: ?Sized + 'static> ExecuteState2<'a, W> {
    fn new(
        executable: &'a ExecutableTaskGraph<W>,
        resource_map: &'a ResourceMap<'a>,
        deferreds: &'a mut Vec<Deferred>,
        current_frame_index: u32,
        current_fence: &'a Fence,
        submission_count: &'a mut usize,
        world: &'a W,
    ) -> Result<Self> {
        let fns = executable.device().fns();
        let (cmd_pipeline_barrier2, queue_submit2);

        if executable.device().api_version() >= Version::V1_3 {
            cmd_pipeline_barrier2 = fns.v1_3.cmd_pipeline_barrier2;
            queue_submit2 = fns.v1_3.queue_submit2;
        } else {
            cmd_pipeline_barrier2 = fns.khr_synchronization2.cmd_pipeline_barrier2_khr;
            queue_submit2 = fns.khr_synchronization2.queue_submit2_khr;
        }

        Ok(ExecuteState2 {
            executable,
            resource_map,
            deferreds,
            current_frame_index,
            current_fence,
            submission_count,
            world,
            cmd_pipeline_barrier2,
            queue_submit2,
            per_submits: SmallVec::new(),
            current_per_submit: PerSubmitInfo2::default(),
            current_command_buffer: None,
            command_buffer_state: CommandBufferState::default(),
            command_buffers: Vec::new(),
            current_buffer_barriers_vk: Vec::new(),
            current_image_barriers_vk: Vec::new(),
            current_render_pass_index: 0,
            current_framebuffer_index: 0,
            current_subpass_index: 0,
            clear_values: LinearMap::new(),
            clear_values_vk: Vec::new(),
            clear_attachments_vk: Vec::new(),
        })
    }

    fn initial_pipeline_barrier(&mut self, barrier_range: Range<BarrierIndex>) {
        self.convert_initial_barriers(barrier_range);
    }

    fn convert_initial_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
        let queue_family_index = current_submission!(self).queue.queue_family_index();

        for barrier in &self.executable.barriers[barrier_range] {
            match barrier.resource.object_type() {
                ObjectType::Buffer => {
                    let buffer_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.buffer_unchecked(buffer_id) };
                    let buffer = state.buffer();
                    let access = state.access();

                    let mut src_stage_mask = PipelineStages::empty();
                    let mut src_access_mask = AccessFlags::empty();
                    let dst_stage_mask = barrier.dst_stage_mask;
                    let mut dst_access_mask = barrier.dst_access_mask;

                    if access.queue_family_index() == queue_family_index {
                        src_stage_mask = access.stage_mask();
                        src_access_mask = access.access_mask();
                    }

                    if src_access_mask.contains_writes() && dst_access_mask.contains_reads() {
                    } else if dst_access_mask.contains_writes() {
                        src_access_mask = AccessFlags::empty();
                        dst_access_mask = AccessFlags::empty();
                    } else {
                        continue;
                    }

                    self.current_buffer_barriers_vk.push(
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(src_stage_mask.into())
                            .src_access_mask(src_access_mask.into())
                            .dst_stage_mask(dst_stage_mask.into())
                            .dst_access_mask(dst_access_mask.into())
                            // FIXME:
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(buffer.handle())
                            .offset(0)
                            .size(buffer.size()),
                    );
                }
                ObjectType::Image => {
                    let image_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.image_unchecked(image_id) };
                    let image = state.image();
                    let access = state.access();

                    let mut src_stage_mask = PipelineStages::empty();
                    let mut src_access_mask = AccessFlags::empty();
                    let dst_stage_mask = barrier.dst_stage_mask;
                    let mut dst_access_mask = barrier.dst_access_mask;

                    if access.queue_family_index() == queue_family_index {
                        src_stage_mask = access.stage_mask();
                        src_access_mask = access.access_mask();
                    }

                    #[allow(clippy::if_same_then_else)]
                    if access.image_layout() != barrier.new_layout {
                    } else if src_access_mask.contains_writes() && dst_access_mask.contains_reads()
                    {
                    } else if dst_access_mask.contains_writes() {
                        src_access_mask = AccessFlags::empty();
                        dst_access_mask = AccessFlags::empty();
                    } else {
                        continue;
                    }

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(src_stage_mask.into())
                            .src_access_mask(src_access_mask.into())
                            .dst_stage_mask(dst_stage_mask.into())
                            .dst_access_mask(dst_access_mask.into())
                            .old_layout(access.image_layout().into())
                            .new_layout(barrier.new_layout.into())
                            // FIXME:
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
                    let image = state.current_image();
                    let access = state.access();

                    let mut src_stage_mask = PipelineStages::empty();
                    let mut src_access_mask = AccessFlags::empty();
                    let dst_stage_mask = barrier.dst_stage_mask;
                    let mut dst_access_mask = barrier.dst_access_mask;

                    if access.queue_family_index() == queue_family_index {
                        src_stage_mask = access.stage_mask();
                        src_access_mask = access.access_mask();
                    }

                    #[allow(clippy::if_same_then_else)]
                    if access.image_layout() != barrier.new_layout {
                    } else if src_access_mask.contains_writes() && dst_access_mask.contains_reads()
                    {
                    } else if dst_access_mask.contains_writes() {
                        src_access_mask = AccessFlags::empty();
                        dst_access_mask = AccessFlags::empty();
                    } else {
                        continue;
                    }

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(src_stage_mask.into())
                            .src_access_mask(src_access_mask.into())
                            .dst_stage_mask(dst_stage_mask.into())
                            .dst_access_mask(dst_access_mask.into())
                            .old_layout(access.image_layout().into())
                            .new_layout(barrier.new_layout.into())
                            // FIXME:
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    fn wait_acquire(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };

        // The acquire semaphore isn't set if an image index has already been acquired but not used
        // in a previous task graph execution and the semaphore has already been waited on.
        let Some(semaphore_vk) = (unsafe { swapchain_state.current_acquire_semaphore() }) else {
            return;
        };

        self.current_per_submit.wait_semaphore_infos_vk.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore_vk)
                .stage_mask(stage_mask.into()),
        );
    }

    fn wait_semaphore(&mut self, semaphore_index: SemaphoreIndex, stage_mask: PipelineStages) {
        self.current_per_submit.wait_semaphore_infos_vk.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.executable.semaphores.borrow()[semaphore_index].handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn execute_task(&mut self, node_index: NodeIndex) -> Result {
        if !self.current_buffer_barriers_vk.is_empty() || !self.current_image_barriers_vk.is_empty()
        {
            self.flush_barriers()?;
        }

        let task_node = unsafe { self.executable.graph.nodes.task_node_unchecked(node_index) };
        let task = &task_node.task;
        let mut current_command_buffer = unsafe {
            RecordingCommandBuffer::new(
                current_command_buffer!(self),
                &mut self.command_buffer_state,
                self.resource_map,
                self.deferreds,
            )
        };
        let mut context = TaskContext {
            resource_map: self.resource_map,
            current_frame_index: self.current_frame_index,
            command_buffers: &mut self.command_buffers,
        };

        unsafe { task.execute(&mut current_command_buffer, &mut context, self.world) }
            .map_err(|err| err.map(|error| ExecuteError::Task { node_index, error }))?;

        if !self.command_buffers.is_empty() {
            unsafe { self.flush_current_command_buffer() }?;

            for command_buffer in self.command_buffers.drain(..) {
                self.current_per_submit.command_buffer_infos_vk.push(
                    vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer.handle()),
                );
                self.deferreds.push(Deferred::destroy(command_buffer));
            }
        }

        Ok(())
    }

    fn pipeline_barrier(&mut self, barrier_range: Range<BarrierIndex>) -> Result {
        self.convert_barriers(barrier_range);

        self.flush_barriers()
    }

    fn convert_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;

        for barrier in &self.executable.barriers[barrier_range] {
            match barrier.resource.object_type() {
                ObjectType::Buffer => {
                    let buffer_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.buffer_unchecked(buffer_id) };
                    let buffer = state.buffer();

                    self.current_buffer_barriers_vk.push(
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(barrier.src_stage_mask.into())
                            .src_access_mask(barrier.src_access_mask.into())
                            .dst_stage_mask(barrier.dst_stage_mask.into())
                            .dst_access_mask(barrier.dst_access_mask.into())
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .buffer(buffer.handle())
                            .offset(0)
                            .size(buffer.size()),
                    );
                }
                ObjectType::Image => {
                    let image_id = unsafe { barrier.resource.parametrize() };
                    let image = unsafe { self.resource_map.image_unchecked(image_id) }.image();

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(barrier.src_stage_mask.into())
                            .src_access_mask(barrier.src_access_mask.into())
                            .dst_stage_mask(barrier.dst_stage_mask.into())
                            .dst_access_mask(barrier.dst_access_mask.into())
                            .old_layout(barrier.old_layout.into())
                            .new_layout(barrier.new_layout.into())
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.resource.parametrize() };
                    let image = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) }
                        .current_image();

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(barrier.src_stage_mask.into())
                            .src_access_mask(barrier.src_access_mask.into())
                            .dst_stage_mask(barrier.dst_stage_mask.into())
                            .dst_access_mask(barrier.dst_access_mask.into())
                            .old_layout(barrier.old_layout.into())
                            .new_layout(barrier.new_layout.into())
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    fn begin_render_pass(&mut self, render_pass_index: RenderPassIndex) -> Result {
        if !self.current_buffer_barriers_vk.is_empty() || !self.current_image_barriers_vk.is_empty()
        {
            self.flush_barriers()?;
        }

        let render_pass_state = &self.executable.render_passes.borrow()[render_pass_index];
        let framebuffer_index =
            unsafe { framebuffer_index(self.resource_map, &self.executable.swapchains) };
        let framebuffer = &render_pass_state.framebuffers[framebuffer_index];

        self.current_render_pass_index = render_pass_index;
        self.current_framebuffer_index = framebuffer_index;
        self.current_subpass_index = 0;

        unsafe {
            self.command_buffer_state.set_local_set(
                self.resource_map,
                self.deferreds,
                framebuffer,
                0,
            )
        }
        .map_err(ExecuteError::VulkanError)?;

        // FIXME:
        let mut render_area_vk = vk::Rect2D::default();
        [render_area_vk.extent.width, render_area_vk.extent.height] = framebuffer.extent();

        unsafe {
            set_clear_values(
                &self.executable.graph.nodes,
                self.resource_map,
                self.world,
                render_pass_state,
                &mut self.clear_values,
                &mut self.clear_values_vk,
            )
        };

        let render_pass_begin_info_vk = vk::RenderPassBeginInfo::default()
            .render_pass(framebuffer.render_pass().handle())
            .framebuffer(framebuffer.handle())
            .render_area(render_area_vk)
            .clear_values(&self.clear_values_vk);

        let fns = self.executable.device().fns();
        unsafe {
            (fns.v1_0.cmd_begin_render_pass)(
                current_command_buffer!(self).handle(),
                &render_pass_begin_info_vk,
                vk::SubpassContents::INLINE,
            )
        };

        Ok(())
    }

    fn next_subpass(&mut self) -> Result {
        self.current_subpass_index += 1;

        let render_pass_state =
            &self.executable.render_passes.borrow()[self.current_render_pass_index];
        let framebuffer = &render_pass_state.framebuffers[self.current_framebuffer_index];

        unsafe {
            self.command_buffer_state.set_local_set(
                self.resource_map,
                self.deferreds,
                framebuffer,
                self.current_subpass_index,
            )
        }
        .map_err(ExecuteError::VulkanError)?;

        let fns = self.executable.device().fns();
        unsafe {
            (fns.v1_0.cmd_next_subpass)(
                self.current_command_buffer.as_ref().unwrap().handle(),
                vk::SubpassContents::INLINE,
            )
        };

        Ok(())
    }

    fn end_render_pass(&mut self) {
        self.command_buffer_state.reset_local_set();

        let fns = self.executable.device().fns();
        unsafe {
            (fns.v1_0.cmd_end_render_pass)(self.current_command_buffer.as_ref().unwrap().handle())
        };
    }

    fn clear_attachments(
        &mut self,
        node_index: NodeIndex,
        render_pass_index: RenderPassIndex,
        clear_attachment_range: Range<ClearAttachmentIndex>,
    ) -> Result {
        if !self.current_buffer_barriers_vk.is_empty() || !self.current_image_barriers_vk.is_empty()
        {
            self.flush_barriers()?;
        }

        let render_pass_state = &self.executable.render_passes.borrow()[render_pass_index];

        let attachments = &self.executable.clear_attachments[clear_attachment_range];

        unsafe {
            set_clear_attachments(
                &self.executable.graph.nodes,
                self.resource_map,
                self.world,
                node_index,
                render_pass_state,
                attachments,
                &mut self.clear_values,
                &mut self.clear_attachments_vk,
            )
        };

        Ok(())
    }

    fn signal_semaphore(&mut self, semaphore_index: SemaphoreIndex, stage_mask: PipelineStages) {
        self.current_per_submit.signal_semaphore_infos_vk.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.executable.semaphores.borrow()[semaphore_index].handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn signal_pre_present(
        &mut self,
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    ) -> Result {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore_vk = unsafe { swapchain_state.init_pre_present_semaphore() }
            .map_err(ExecuteError::VulkanError)?;

        self.current_per_submit.signal_semaphore_infos_vk.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore_vk)
                .stage_mask(stage_mask.into()),
        );

        Ok(())
    }

    fn wait_pre_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore_vk = unsafe { swapchain_state.current_pre_present_semaphore() }.unwrap();

        self.current_per_submit.wait_semaphore_infos_vk.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore_vk)
                .stage_mask(stage_mask.into()),
        );
    }

    fn signal_present(
        &mut self,
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    ) -> Result {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore_vk = unsafe { swapchain_state.init_present_semaphore() }
            .map_err(ExecuteError::VulkanError)?;

        self.current_per_submit.signal_semaphore_infos_vk.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore_vk)
                .stage_mask(stage_mask.into()),
        );

        Ok(())
    }

    fn flush_barriers(&mut self) -> Result {
        unsafe {
            (self.cmd_pipeline_barrier2)(
                current_command_buffer!(self).handle(),
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(&self.current_buffer_barriers_vk)
                    .image_memory_barriers(&self.current_image_barriers_vk),
            )
        };

        self.current_buffer_barriers_vk.clear();
        self.current_image_barriers_vk.clear();

        Ok(())
    }

    fn flush_submit(&mut self) -> Result {
        unsafe { self.flush_current_command_buffer() }?;

        self.per_submits
            .push(mem::take(&mut self.current_per_submit));

        Ok(())
    }

    fn submit(&mut self) -> Result {
        unsafe {
            self.executable
                .flush_mapped_memory_ranges(self.resource_map)
        }?;

        let submission = current_submission!(self);

        let mut submit_infos_vk = SmallVec::<[_; 4]>::with_capacity(self.per_submits.len());
        submit_infos_vk.extend(self.per_submits.iter().map(|per_submit| {
            vk::SubmitInfo2::default()
                .wait_semaphore_infos(&per_submit.wait_semaphore_infos_vk)
                .command_buffer_infos(&per_submit.command_buffer_infos_vk)
                .signal_semaphore_infos(&per_submit.signal_semaphore_infos_vk)
        }));

        let max_submission_index = self.executable.submissions.len() - 1;
        let fence_handle = if *self.submission_count == max_submission_index {
            self.current_fence.handle()
        } else {
            vk::Fence::null()
        };

        submission
            .queue
            .with(|_guard| {
                unsafe {
                    (self.queue_submit2)(
                        submission.queue.handle(),
                        submit_infos_vk.len() as u32,
                        submit_infos_vk.as_ptr(),
                        fence_handle,
                    )
                }
                .result()
                .map_err(VulkanError::from)
            })
            .map_err(ExecuteError::VulkanError)?;

        drop(submit_infos_vk);
        self.per_submits.clear();

        *self.submission_count += 1;

        Ok(())
    }

    unsafe fn flush_current_command_buffer(&mut self) -> Result {
        let current_command_buffer = self.current_command_buffer.take().unwrap();
        let command_buffer =
            unsafe { current_command_buffer.end() }.map_err(ExecuteError::VulkanError)?;
        self.current_per_submit
            .command_buffer_infos_vk
            .push(vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer.handle()));
        self.deferreds.push(Deferred::destroy(command_buffer));
        self.command_buffer_state.reset();

        Ok(())
    }
}

struct ExecuteState<'a, W: ?Sized + 'static> {
    executable: &'a ExecutableTaskGraph<W>,
    resource_map: &'a ResourceMap<'a>,
    deferreds: &'a mut Vec<Deferred>,
    current_frame_index: u32,
    current_fence: &'a Fence,
    submission_count: &'a mut usize,
    world: &'a W,
    cmd_pipeline_barrier: vk::PFN_vkCmdPipelineBarrier,
    queue_submit: vk::PFN_vkQueueSubmit,
    per_submits: SmallVec<[PerSubmitInfo; 4]>,
    current_per_submit: PerSubmitInfo,
    current_command_buffer: Option<raw::RecordingCommandBuffer>,
    command_buffer_state: CommandBufferState,
    command_buffers: Vec<Arc<raw::CommandBuffer>>,
    current_buffer_barriers_vk: Vec<vk::BufferMemoryBarrier<'static>>,
    current_image_barriers_vk: Vec<vk::ImageMemoryBarrier<'static>>,
    current_src_stage_mask_vk: vk::PipelineStageFlags,
    current_dst_stage_mask_vk: vk::PipelineStageFlags,
    current_render_pass_index: RenderPassIndex,
    current_framebuffer_index: usize,
    current_subpass_index: usize,
    clear_values: LinearMap<Id, Option<ClearValue>>,
    clear_values_vk: Vec<vk::ClearValue>,
    clear_attachments_vk: Vec<vk::ClearAttachment>,
}

#[derive(Default)]
struct PerSubmitInfo {
    wait_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
    wait_dst_stage_mask_vk: SmallVec<[vk::PipelineStageFlags; 4]>,
    command_buffers_vk: SmallVec<[vk::CommandBuffer; 1]>,
    signal_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
}

impl<'a, W: ?Sized + 'static> ExecuteState<'a, W> {
    fn new(
        executable: &'a ExecutableTaskGraph<W>,
        resource_map: &'a ResourceMap<'a>,
        deferreds: &'a mut Vec<Deferred>,
        current_frame_index: u32,
        current_fence: &'a Fence,
        submission_count: &'a mut usize,
        world: &'a W,
    ) -> Result<Self> {
        let fns = executable.device().fns();
        let cmd_pipeline_barrier = fns.v1_0.cmd_pipeline_barrier;
        let queue_submit = fns.v1_0.queue_submit;

        Ok(ExecuteState {
            executable,
            resource_map,
            deferreds,
            current_frame_index,
            current_fence,
            submission_count,
            world,
            cmd_pipeline_barrier,
            queue_submit,
            per_submits: SmallVec::new(),
            current_per_submit: PerSubmitInfo::default(),
            current_command_buffer: None,
            command_buffer_state: CommandBufferState::default(),
            command_buffers: Vec::new(),
            current_buffer_barriers_vk: Vec::new(),
            current_image_barriers_vk: Vec::new(),
            current_src_stage_mask_vk: vk::PipelineStageFlags::empty(),
            current_dst_stage_mask_vk: vk::PipelineStageFlags::empty(),
            current_render_pass_index: 0,
            current_framebuffer_index: 0,
            current_subpass_index: 0,
            clear_values: LinearMap::new(),
            clear_values_vk: Vec::new(),
            clear_attachments_vk: Vec::new(),
        })
    }

    fn initial_pipeline_barrier(&mut self, barrier_range: Range<BarrierIndex>) {
        self.convert_initial_barriers(barrier_range);
    }

    fn convert_initial_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
        let queue_family_index = current_submission!(self).queue.queue_family_index();

        for barrier in &self.executable.barriers[barrier_range] {
            match barrier.resource.object_type() {
                ObjectType::Buffer => {
                    let buffer_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.buffer_unchecked(buffer_id) };
                    let buffer = state.buffer();
                    let access = state.access();

                    let mut src_stage_mask = PipelineStages::empty();
                    let mut src_access_mask = AccessFlags::empty();
                    let dst_stage_mask = barrier.dst_stage_mask;
                    let mut dst_access_mask = barrier.dst_access_mask;

                    if access.queue_family_index() == queue_family_index {
                        src_stage_mask = access.stage_mask();
                        src_access_mask = access.access_mask();
                    }

                    if src_access_mask.contains_writes() && dst_access_mask.contains_reads() {
                    } else if dst_access_mask.contains_writes() {
                        src_access_mask = AccessFlags::empty();
                        dst_access_mask = AccessFlags::empty();
                    } else {
                        continue;
                    }

                    self.current_buffer_barriers_vk.push(
                        vk::BufferMemoryBarrier::default()
                            .src_access_mask(convert_access_mask(src_access_mask))
                            .dst_access_mask(convert_access_mask(dst_access_mask))
                            // FIXME:
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(buffer.handle())
                            .offset(0)
                            .size(buffer.size()),
                    );

                    self.current_src_stage_mask_vk |= convert_stage_mask(src_stage_mask);
                    self.current_dst_stage_mask_vk |= convert_stage_mask(dst_stage_mask);
                }
                ObjectType::Image => {
                    let image_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.image_unchecked(image_id) };
                    let image = state.image();
                    let access = state.access();

                    let mut src_stage_mask = PipelineStages::empty();
                    let mut src_access_mask = AccessFlags::empty();
                    let dst_stage_mask = barrier.dst_stage_mask;
                    let mut dst_access_mask = barrier.dst_access_mask;

                    if access.queue_family_index() == queue_family_index {
                        src_stage_mask = access.stage_mask();
                        src_access_mask = access.access_mask();
                    }

                    #[allow(clippy::if_same_then_else)]
                    if access.image_layout() != barrier.new_layout {
                    } else if src_access_mask.contains_writes() && dst_access_mask.contains_reads()
                    {
                    } else if dst_access_mask.contains_writes() {
                        src_access_mask = AccessFlags::empty();
                        dst_access_mask = AccessFlags::empty();
                    } else {
                        continue;
                    }

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(convert_access_mask(src_access_mask))
                            .dst_access_mask(convert_access_mask(dst_access_mask))
                            .old_layout(access.image_layout().into())
                            .new_layout(barrier.new_layout.into())
                            // FIXME:
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );

                    self.current_src_stage_mask_vk |= convert_stage_mask(src_stage_mask);
                    self.current_dst_stage_mask_vk |= convert_stage_mask(dst_stage_mask);
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
                    let image = state.current_image();
                    let access = state.access();

                    let mut src_stage_mask = PipelineStages::empty();
                    let mut src_access_mask = AccessFlags::empty();
                    let dst_stage_mask = barrier.dst_stage_mask;
                    let mut dst_access_mask = barrier.dst_access_mask;

                    if access.queue_family_index() == queue_family_index {
                        src_stage_mask = access.stage_mask();
                        src_access_mask = access.access_mask();
                    }

                    #[allow(clippy::if_same_then_else)]
                    if access.image_layout() != barrier.new_layout {
                    } else if src_access_mask.contains_writes() && dst_access_mask.contains_reads()
                    {
                    } else if dst_access_mask.contains_writes() {
                        src_access_mask = AccessFlags::empty();
                        dst_access_mask = AccessFlags::empty();
                    } else {
                        continue;
                    }

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(convert_access_mask(src_access_mask))
                            .dst_access_mask(convert_access_mask(dst_access_mask))
                            .old_layout(access.image_layout().into())
                            .new_layout(barrier.new_layout.into())
                            // FIXME:
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );

                    self.current_src_stage_mask_vk |= convert_stage_mask(src_stage_mask);
                    self.current_dst_stage_mask_vk |= convert_stage_mask(dst_stage_mask);
                }
                _ => unreachable!(),
            }
        }
    }

    fn wait_acquire(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };

        // The acquire semaphore isn't set if an image index has already been acquired but not used
        // in a previous task graph execution and the semaphore has already been waited on.
        let Some(semaphore_vk) = (unsafe { swapchain_state.current_acquire_semaphore() }) else {
            return;
        };

        self.current_per_submit
            .wait_semaphores_vk
            .push(semaphore_vk);
        self.current_per_submit
            .wait_dst_stage_mask_vk
            .push(convert_stage_mask(stage_mask));
    }

    fn wait_semaphore(&mut self, semaphore_index: SemaphoreIndex, stage_mask: PipelineStages) {
        self.current_per_submit
            .wait_semaphores_vk
            .push(self.executable.semaphores.borrow()[semaphore_index].handle());
        self.current_per_submit
            .wait_dst_stage_mask_vk
            .push(convert_stage_mask(stage_mask));
    }

    fn execute_task(&mut self, node_index: NodeIndex) -> Result {
        if !self.current_buffer_barriers_vk.is_empty() || !self.current_image_barriers_vk.is_empty()
        {
            self.flush_barriers()?;
        }

        let task_node = unsafe { self.executable.graph.nodes.task_node_unchecked(node_index) };
        let task = &task_node.task;
        let mut current_command_buffer = unsafe {
            RecordingCommandBuffer::new(
                current_command_buffer!(self),
                &mut self.command_buffer_state,
                self.resource_map,
                self.deferreds,
            )
        };
        let mut context = TaskContext {
            resource_map: self.resource_map,
            current_frame_index: self.current_frame_index,
            command_buffers: &mut self.command_buffers,
        };

        unsafe { task.execute(&mut current_command_buffer, &mut context, self.world) }
            .map_err(|err| err.map(|error| ExecuteError::Task { node_index, error }))?;

        if !self.command_buffers.is_empty() {
            unsafe { self.flush_current_command_buffer() }?;

            for command_buffer in self.command_buffers.drain(..) {
                self.current_per_submit
                    .command_buffers_vk
                    .push(command_buffer.handle());
                self.deferreds.push(Deferred::destroy(command_buffer));
            }
        }

        Ok(())
    }

    fn pipeline_barrier(&mut self, barrier_range: Range<BarrierIndex>) -> Result {
        self.convert_barriers(barrier_range);

        self.flush_barriers()
    }

    fn convert_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;

        for barrier in &self.executable.barriers[barrier_range] {
            match barrier.resource.object_type() {
                ObjectType::Buffer => {
                    let buffer_id = unsafe { barrier.resource.parametrize() };
                    let state = unsafe { self.resource_map.buffer_unchecked(buffer_id) };
                    let buffer = state.buffer();

                    self.current_buffer_barriers_vk.push(
                        vk::BufferMemoryBarrier::default()
                            .src_access_mask(convert_access_mask(barrier.src_access_mask))
                            .dst_access_mask(convert_access_mask(barrier.dst_access_mask))
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .buffer(state.buffer().handle())
                            .offset(0)
                            .size(buffer.size()),
                    );

                    self.current_src_stage_mask_vk |= convert_stage_mask(barrier.src_stage_mask);
                    self.current_dst_stage_mask_vk |= convert_stage_mask(barrier.dst_stage_mask);
                }
                ObjectType::Image => {
                    let image_id = unsafe { barrier.resource.parametrize() };
                    let image = unsafe { self.resource_map.image_unchecked(image_id) }.image();

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(convert_access_mask(barrier.src_access_mask))
                            .dst_access_mask(convert_access_mask(barrier.dst_access_mask))
                            .old_layout(barrier.old_layout.into())
                            .new_layout(barrier.new_layout.into())
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );

                    self.current_src_stage_mask_vk |= convert_stage_mask(barrier.src_stage_mask);
                    self.current_dst_stage_mask_vk |= convert_stage_mask(barrier.dst_stage_mask);
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.resource.parametrize() };
                    let image = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) }
                        .current_image();

                    self.current_image_barriers_vk.push(
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(convert_access_mask(barrier.src_access_mask))
                            .dst_access_mask(convert_access_mask(barrier.dst_access_mask))
                            .old_layout(barrier.old_layout.into())
                            .new_layout(barrier.new_layout.into())
                            .src_queue_family_index(barrier.src_queue_family_index)
                            .dst_queue_family_index(barrier.dst_queue_family_index)
                            .image(image.handle())
                            .subresource_range(image.subresource_range().to_vk()),
                    );

                    self.current_src_stage_mask_vk |= convert_stage_mask(barrier.src_stage_mask);
                    self.current_dst_stage_mask_vk |= convert_stage_mask(barrier.dst_stage_mask);
                }
                _ => unreachable!(),
            }
        }
    }

    fn begin_render_pass(&mut self, render_pass_index: RenderPassIndex) -> Result {
        if !self.current_buffer_barriers_vk.is_empty() || !self.current_image_barriers_vk.is_empty()
        {
            self.flush_barriers()?;
        }

        let render_pass_state = &self.executable.render_passes.borrow()[render_pass_index];
        let framebuffer_index =
            unsafe { framebuffer_index(self.resource_map, &self.executable.swapchains) };
        let framebuffer = &render_pass_state.framebuffers[framebuffer_index];

        self.current_render_pass_index = render_pass_index;
        self.current_framebuffer_index = framebuffer_index;
        self.current_subpass_index = 0;

        unsafe {
            self.command_buffer_state.set_local_set(
                self.resource_map,
                self.deferreds,
                framebuffer,
                0,
            )
        }
        .map_err(ExecuteError::VulkanError)?;

        // FIXME:
        let mut render_area_vk = vk::Rect2D::default();
        [render_area_vk.extent.width, render_area_vk.extent.height] = framebuffer.extent();

        unsafe {
            set_clear_values(
                &self.executable.graph.nodes,
                self.resource_map,
                self.world,
                render_pass_state,
                &mut self.clear_values,
                &mut self.clear_values_vk,
            )
        };

        let render_pass_begin_info_vk = vk::RenderPassBeginInfo::default()
            .render_pass(framebuffer.render_pass().handle())
            .framebuffer(framebuffer.handle())
            .render_area(render_area_vk)
            .clear_values(&self.clear_values_vk);

        let fns = self.executable.device().fns();
        unsafe {
            (fns.v1_0.cmd_begin_render_pass)(
                current_command_buffer!(self).handle(),
                &render_pass_begin_info_vk,
                vk::SubpassContents::INLINE,
            )
        };

        Ok(())
    }

    fn next_subpass(&mut self) -> Result {
        self.current_subpass_index += 1;

        let render_pass_state =
            &self.executable.render_passes.borrow()[self.current_render_pass_index];
        let framebuffer = &render_pass_state.framebuffers[self.current_framebuffer_index];

        unsafe {
            self.command_buffer_state.set_local_set(
                self.resource_map,
                self.deferreds,
                framebuffer,
                self.current_subpass_index,
            )
        }
        .map_err(ExecuteError::VulkanError)?;

        let fns = self.executable.device().fns();
        unsafe {
            (fns.v1_0.cmd_next_subpass)(
                self.current_command_buffer.as_ref().unwrap().handle(),
                vk::SubpassContents::INLINE,
            )
        };

        Ok(())
    }

    fn end_render_pass(&mut self) {
        self.command_buffer_state.reset_local_set();

        let fns = self.executable.device().fns();
        unsafe {
            (fns.v1_0.cmd_end_render_pass)(self.current_command_buffer.as_ref().unwrap().handle())
        };
    }

    fn clear_attachments(
        &mut self,
        node_index: NodeIndex,
        render_pass_index: RenderPassIndex,
        clear_attachment_range: Range<ClearAttachmentIndex>,
    ) -> Result {
        if !self.current_buffer_barriers_vk.is_empty() || !self.current_image_barriers_vk.is_empty()
        {
            self.flush_barriers()?;
        }

        let render_pass_state = &self.executable.render_passes.borrow()[render_pass_index];

        let attachments = &self.executable.clear_attachments[clear_attachment_range];

        unsafe {
            set_clear_attachments(
                &self.executable.graph.nodes,
                self.resource_map,
                self.world,
                node_index,
                render_pass_state,
                attachments,
                &mut self.clear_values,
                &mut self.clear_attachments_vk,
            )
        };

        Ok(())
    }

    fn signal_semaphore(&mut self, semaphore_index: SemaphoreIndex, _stage_mask: PipelineStages) {
        self.current_per_submit
            .signal_semaphores_vk
            .push(self.executable.semaphores.borrow()[semaphore_index].handle());
    }

    fn signal_pre_present(
        &mut self,
        swapchain_id: Id<Swapchain>,
        _stage_mask: PipelineStages,
    ) -> Result {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore_vk = unsafe { swapchain_state.init_pre_present_semaphore() }
            .map_err(ExecuteError::VulkanError)?;

        self.current_per_submit
            .signal_semaphores_vk
            .push(semaphore_vk);

        Ok(())
    }

    fn wait_pre_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore_vk = unsafe { swapchain_state.current_pre_present_semaphore() }.unwrap();

        self.current_per_submit
            .wait_semaphores_vk
            .push(semaphore_vk);
        self.current_per_submit
            .wait_dst_stage_mask_vk
            .push(convert_stage_mask(stage_mask));
    }

    fn signal_present(
        &mut self,
        swapchain_id: Id<Swapchain>,
        _stage_mask: PipelineStages,
    ) -> Result {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore_vk = unsafe { swapchain_state.init_present_semaphore() }
            .map_err(ExecuteError::VulkanError)?;

        self.current_per_submit
            .signal_semaphores_vk
            .push(semaphore_vk);

        Ok(())
    }

    fn flush_submit(&mut self) -> Result {
        unsafe { self.flush_current_command_buffer() }?;

        self.per_submits
            .push(mem::take(&mut self.current_per_submit));

        Ok(())
    }

    fn flush_barriers(&mut self) -> Result {
        if self.current_src_stage_mask_vk.is_empty() {
            self.current_src_stage_mask_vk = vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        if self.current_dst_stage_mask_vk.is_empty() {
            self.current_dst_stage_mask_vk = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }

        unsafe {
            (self.cmd_pipeline_barrier)(
                current_command_buffer!(self).handle(),
                self.current_src_stage_mask_vk,
                self.current_dst_stage_mask_vk,
                vk::DependencyFlags::empty(),
                0,
                ptr::null(),
                self.current_buffer_barriers_vk.len() as u32,
                self.current_buffer_barriers_vk.as_ptr(),
                self.current_image_barriers_vk.len() as u32,
                self.current_image_barriers_vk.as_ptr(),
            )
        };

        self.current_buffer_barriers_vk.clear();
        self.current_image_barriers_vk.clear();
        self.current_src_stage_mask_vk = vk::PipelineStageFlags::empty();
        self.current_dst_stage_mask_vk = vk::PipelineStageFlags::empty();

        Ok(())
    }

    fn submit(&mut self) -> Result {
        unsafe {
            self.executable
                .flush_mapped_memory_ranges(self.resource_map)
        }?;

        let submission = current_submission!(self);

        let mut submit_infos_vk = SmallVec::<[_; 4]>::with_capacity(self.per_submits.len());
        submit_infos_vk.extend(self.per_submits.iter().map(|per_submit| {
            vk::SubmitInfo::default()
                .wait_semaphores(&per_submit.wait_semaphores_vk)
                .wait_dst_stage_mask(&per_submit.wait_dst_stage_mask_vk)
                .command_buffers(&per_submit.command_buffers_vk)
                .signal_semaphores(&per_submit.signal_semaphores_vk)
        }));

        let max_submission_index = self.executable.submissions.len() - 1;
        let fence_vk = if *self.submission_count == max_submission_index {
            self.current_fence.handle()
        } else {
            vk::Fence::null()
        };

        submission
            .queue
            .with(|_guard| {
                unsafe {
                    (self.queue_submit)(
                        submission.queue.handle(),
                        submit_infos_vk.len() as u32,
                        submit_infos_vk.as_ptr(),
                        fence_vk,
                    )
                }
                .result()
                .map_err(VulkanError::from)
            })
            .map_err(ExecuteError::VulkanError)?;

        drop(submit_infos_vk);
        self.per_submits.clear();

        *self.submission_count += 1;

        Ok(())
    }

    unsafe fn flush_current_command_buffer(&mut self) -> Result {
        let current_command_buffer = self.current_command_buffer.take().unwrap();
        let command_buffer =
            unsafe { current_command_buffer.end() }.map_err(ExecuteError::VulkanError)?;
        self.current_per_submit
            .command_buffers_vk
            .push(command_buffer.handle());
        self.deferreds.push(Deferred::destroy(command_buffer));
        self.command_buffer_state.reset();

        Ok(())
    }
}

macro_rules! current_submission {
    ($state:expr) => {
        &$state.executable.submissions[*$state.submission_count]
    };
}
use current_submission;

macro_rules! current_command_buffer {
    ($state:expr) => {{
        if $state.current_command_buffer.is_none() {
            $state.current_command_buffer = Some(
                create_command_buffer($state.resource_map, &current_submission!($state).queue)
                    .map_err(ExecuteError::VulkanError)?,
            );
        }

        $state.current_command_buffer.as_mut().unwrap()
    }};
}
use current_command_buffer;

fn create_command_buffer(
    resource_map: &ResourceMap<'_>,
    queue: &Queue,
) -> Result<raw::RecordingCommandBuffer, VulkanError> {
    let allocator = resource_map.physical_resources.command_buffer_allocator();

    // SAFETY: The parameters are valid.
    unsafe {
        raw::RecordingCommandBuffer::new_unchecked(
            allocator,
            queue.queue_family_index(),
            raw::CommandBufferLevel::Primary,
            &raw::CommandBufferBeginInfo {
                usage: raw::CommandBufferUsage::OneTimeSubmit,
                inheritance_info: None,
                ..Default::default()
            },
        )
    }
}

unsafe fn framebuffer_index(resource_map: &ResourceMap<'_>, swapchains: &[Id<Swapchain>]) -> usize {
    let mut index = 0;
    let mut factor = 1;

    for &swapchain_id in swapchains {
        let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };
        let swapchain = swapchain_state.swapchain();
        index += swapchain_state.current_image_index().unwrap() * factor;
        factor *= swapchain.image_count();
    }

    index as usize
}

unsafe fn set_clear_values<W: ?Sized + 'static>(
    nodes: &super::Nodes<W>,
    resource_map: &ResourceMap<'_>,
    world: &W,
    render_pass_state: &super::RenderPassState,
    clear_values: &mut LinearMap<Id, Option<ClearValue>>,
    clear_values_vk: &mut Vec<vk::ClearValue>,
) {
    clear_values_vk.clear();

    if render_pass_state.clear_node_indices.is_empty() {
        return;
    }

    clear_values.clear();
    clear_values.extend(render_pass_state.attachments.keys().map(|&id| (id, None)));

    for &node_index in &render_pass_state.clear_node_indices {
        let task_node = unsafe { nodes.task_node_unchecked(node_index) };

        task_node.task.clear_values(
            &mut ClearValues {
                inner: clear_values,
                resource_map,
            },
            world,
        );
    }

    clear_values_vk.extend(clear_values.values().map(|clear_value| {
        clear_value
            .as_ref()
            .map_or_else(Default::default, ClearValue::to_vk)
    }));
}

unsafe fn set_clear_attachments<W: ?Sized + 'static>(
    nodes: &super::Nodes<W>,
    resource_map: &ResourceMap<'_>,
    world: &W,
    node_index: NodeIndex,
    render_pass_state: &super::RenderPassState,
    attachments: &[Id],
    clear_values: &mut LinearMap<Id, Option<ClearValue>>,
    clear_attachments_vk: &mut Vec<vk::ClearAttachment>,
) {
    clear_attachments_vk.clear();
    clear_values.clear();
    clear_values.extend(attachments.iter().map(|&id| (id, None)));

    let task_node = unsafe { nodes.task_node_unchecked(node_index) };

    task_node.task.clear_values(
        &mut ClearValues {
            inner: clear_values,
            resource_map,
        },
        world,
    );

    clear_attachments_vk.extend(clear_values.iter().map(|(id, clear_value)| {
        let attachment_state = render_pass_state.attachments.get(id).unwrap();

        vk::ClearAttachment {
            aspect_mask: attachment_state.format.aspects().into(),
            color_attachment: attachment_state.index,
            clear_value: clear_value
                .as_ref()
                .map_or_else(Default::default, ClearValue::to_vk),
        }
    }));
}

fn convert_stage_mask(mut stage_mask: PipelineStages) -> vk::PipelineStageFlags {
    const VERTEX_INPUT_FLAGS: PipelineStages =
        PipelineStages::INDEX_INPUT.union(PipelineStages::VERTEX_ATTRIBUTE_INPUT);
    const TRANSFER_FLAGS: PipelineStages = PipelineStages::COPY
        .union(PipelineStages::RESOLVE)
        .union(PipelineStages::BLIT)
        .union(PipelineStages::CLEAR);

    if stage_mask.intersects(VERTEX_INPUT_FLAGS) {
        stage_mask -= VERTEX_INPUT_FLAGS;
        stage_mask |= PipelineStages::VERTEX_INPUT;
    }

    if stage_mask.intersects(TRANSFER_FLAGS) {
        stage_mask -= TRANSFER_FLAGS;
        stage_mask |= PipelineStages::ALL_TRANSFER;
    }

    stage_mask.into()
}

fn convert_access_mask(mut access_mask: AccessFlags) -> vk::AccessFlags {
    const READ_FLAGS: AccessFlags =
        AccessFlags::SHADER_SAMPLED_READ.union(AccessFlags::SHADER_STORAGE_READ);
    const WRITE_FLAGS: AccessFlags = AccessFlags::SHADER_STORAGE_WRITE;

    if access_mask.intersects(READ_FLAGS) {
        access_mask -= READ_FLAGS;
        access_mask |= AccessFlags::SHADER_READ;
    }

    if access_mask.intersects(WRITE_FLAGS) {
        access_mask -= WRITE_FLAGS;
        access_mask |= AccessFlags::SHADER_WRITE;
    }

    access_mask.into()
}

struct StateGuard<'a, 'b, W: ?Sized + 'static> {
    executable: &'a ExecutableTaskGraph<W>,
    resource_map: &'a ResourceMap<'a>,
    global_lock_guard: &'a mut Option<GlobalLockSharedGuard<'b>>,
    flight: &'a Flight,
    current_fence: &'a mut Fence,
    deferred_batch: &'a mut Option<DeferredBatch<'b>>,
    submission_count: usize,
}

impl<W: ?Sized + 'static> Drop for StateGuard<'_, '_, W> {
    #[cold]
    fn drop(&mut self) {
        let resources = self.resource_map.resources();
        let device = resources.device();

        let has_unfinished_submissions = self.submission_count < self.executable.submissions.len();

        if has_unfinished_submissions {
            // Make sure that the started frame cannot outrun the current frame.
            unsafe { self.flight.undo_start_next_frame() };
        }

        // If at least one submission failed, the fence was never submitted, so it's not in use.
        // Since it's still unsignaled, we have to recreate it.
        if has_unfinished_submissions {
            // SAFETY: The parameters are valid.
            match unsafe {
                Fence::new_unchecked(
                    device,
                    &FenceCreateInfo {
                        flags: FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    },
                )
            } {
                Ok(new_fence) => {
                    *self.current_fence = new_fence;
                }
                Err(err) => {
                    // Device loss is already a form of poisoning built into Vulkan. There's no
                    // invalid state that can be observed by design.
                    if err == VulkanError::DeviceLost {
                        return;
                    }

                    eprintln!(
                        "failed to recreate the current fence after failed execution rendering \
                        recovery impossible: {err}; aborting",
                    );
                    std::process::abort();
                }
            }
        }

        // If no submissions succeeded, the only thing that succeeded was acquiring swapchain
        // images. We keep those acquired for the next task graph execution.
        if self.submission_count == 0 {
            let deferred_batch = self.deferred_batch.take().unwrap();

            // SAFETY: No submissions were made, so it's safe to collect the deferred batch
            // immediately.
            unsafe { deferred_batch.enqueue_with_flights(iter::empty()) };

            return;
        }

        // At this point, we have to enter an exclusive critical section of the global lock because
        // of the following `wait_idle_inner` call. But first, we have to exit our shared section.
        // Any other option is out of the question: we can't upgrade the shared guard to an
        // exclusive guard because that would either be mutual exclusion rendering multi-threading
        // pointless or lead to deadlocks.
        //
        // Note that this is only sound because our resources are still locked between leaving the
        // shared section and entering the exclusive section. This prevents another thread from
        // executing a task graph that uses any of our resources (just as if our thread was still
        // executing our task graph) and prevents another thread from recreating or removing any of
        // our swapchains. If the aforementioned were allowed, another thread would see our
        // resources in an invalid state because we haven't updated their state yet. In order to do
        // so, we must first wait for device idle. One thing the shared section protects against is
        // a device wait idle from another thread sneaking in between 2 submits of ours, which
        // would be UB because semaphores and fences have to be in the unsignaled state when
        // submitting. That's fine, however, since we are done with submissions, and we need to do
        // a device wait idle ourselves. The other thing the shared section protects against is
        // that a device wait idle, by virtue of being inside an exclusive section, can mutably
        // access swapchain state. This too is fine, however, since we don't hold any references to
        // any swapchain state at this point. We must only access swapchain state in a critical
        // section of the global lock.
        let _ = self.global_lock_guard.take().unwrap();
        let _global_lock_guard = resources.lock_global_exclusive();

        // We must make sure that invalid state cannot be observed, because if at least one
        // submission succeeded while one failed, that means that there are pending semaphore
        // signal operations. And if all submissions succeeded, presenting failed.
        //
        // SAFETY: We have locked the global lock exclusively above.
        if let Err(err) = unsafe { resources.wait_idle_inner() } {
            if err == VulkanError::DeviceLost {
                return;
            }

            eprintln!(
                "failed to wait on device idle after partly failed submissions rendering recovery \
                impossible: {err}; aborting",
            );
            std::process::abort();
        }

        let deferred_batch = self.deferred_batch.take().unwrap();

        // SAFETY: We have waited for device idle, so it's safe to collect the deferred batch
        // immediately.
        unsafe { deferred_batch.enqueue_with_flights(iter::empty()) };

        // If at least one submission failed while one succeeded, the state of the graph is invalid
        // because some semaphores are still signaled, so we have to recreate them.
        if has_unfinished_submissions {
            for semaphore in self.executable.semaphores.borrow_mut().iter_mut() {
                // SAFETY: The parameters are valid.
                match unsafe { Semaphore::new_unchecked(device, &Default::default()) } {
                    Ok(new_semaphore) => {
                        *semaphore = new_semaphore;
                    }
                    Err(err) => {
                        if err == VulkanError::DeviceLost {
                            return;
                        }

                        eprintln!(
                            "failed to recreate semaphores after partly failed submissions \
                            rendering recovery impossible: {err}; aborting",
                        );
                        std::process::abort();
                    }
                }
            }
        }

        let mut last_accesses = vec![
            ResourceAccess::default();
            self.executable.graph.resources.reserved_len() as usize
        ];
        let mut last_swapchain_stages = self
            .executable
            .swapchains
            .iter()
            .map(|&swapchain_id| (swapchain_id, SwapchainSyncStage::SignalAcquire))
            .collect::<LinearMap<_, _>>();
        let submissions = &self.executable.submissions;
        let instruction_range = 0..submissions[self.submission_count - 1].instruction_range.end;

        // Determine the last accesses of resources up until before the failed submission.
        for instruction in &self.executable.instructions[instruction_range] {
            match instruction {
                Instruction::ExecuteTask { node_index } => {
                    let task_node =
                        unsafe { self.executable.graph.nodes.task_node_unchecked(*node_index) };

                    for (id, access) in task_node.accesses.iter() {
                        let prev_access = &mut last_accesses[id.index() as usize];
                        let access = ResourceAccess {
                            queue_family_index: task_node.queue_family_index,
                            ..*access
                        };

                        if prev_access.queue_family_index != access.queue_family_index
                            || prev_access.image_layout != access.image_layout
                            || prev_access.access_mask.contains_writes()
                            || access.access_mask.contains_writes()
                        {
                            *prev_access = access;
                        } else {
                            prev_access.stage_mask |= access.stage_mask;
                            prev_access.access_mask |= access.access_mask;
                        }
                    }
                }
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask: _,
                } => {
                    *last_swapchain_stages.get_mut(swapchain_id).unwrap() =
                        SwapchainSyncStage::WaitAcquire;
                }
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask: _,
                } => {
                    *last_swapchain_stages.get_mut(swapchain_id).unwrap() =
                        SwapchainSyncStage::SignalPrePresent;
                }
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask: _,
                } => {
                    *last_swapchain_stages.get_mut(swapchain_id).unwrap() =
                        SwapchainSyncStage::WaitPrePresent;
                }
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask: _,
                } => {
                    *last_swapchain_stages.get_mut(swapchain_id).unwrap() =
                        SwapchainSyncStage::SignalPresent;
                }
                _ => {}
            }
        }

        // Update the resource state with the correct last accesses.
        unsafe {
            self.executable
                .update_resource_state(self.resource_map, &last_accesses)
        };

        // Fix the swapchain state.
        for (&swapchain_id, &sync_stage) in last_swapchain_stages.iter() {
            let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
            unsafe { swapchain_state.handle_execution_failure(sync_stage) };
        }

        // It is fine to exit the exclusive section here as we won't be touching any resources
        // after this point.
    }
}

/// Maps [virtual resources] to physical resources.
pub struct ResourceMap<'a> {
    virtual_resources: &'a super::Resources,
    physical_resources: Arc<Resources>,
    map: Vec<*const ()>,
    len: u32,
    guard: hyaline::Guard<'a>,
}

impl<'a> ResourceMap<'a> {
    /// Creates a new `ResourceMap` mapping the virtual resources of the given `executable`.
    pub fn new(executable: &'a ExecutableTaskGraph<impl ?Sized>) -> Result<Self, InvalidSlotError> {
        let virtual_resources = &executable.graph.resources;
        let physical_resources = virtual_resources.physical_resources.clone();
        let mut map = vec![ptr::null(); virtual_resources.reserved_len() as usize];
        let guard = virtual_resources.physical_resources.pin();

        for (&physical_id, &virtual_id) in &virtual_resources.physical_map {
            // SAFETY: Virtual IDs inside the `physical_map` are always valid.
            let slot = unsafe { map.get_unchecked_mut(virtual_id.index() as usize) };

            *slot = match physical_id.object_type() {
                ObjectType::Buffer => {
                    let physical_id = unsafe { physical_id.parametrize() };

                    <*const _>::cast(physical_resources.try_buffer_protected(physical_id, &guard)?)
                }
                ObjectType::Image => {
                    let physical_id = unsafe { physical_id.parametrize() };

                    <*const _>::cast(physical_resources.try_image_protected(physical_id, &guard)?)
                }
                ObjectType::Swapchain => {
                    let physical_id = unsafe { physical_id.parametrize() };

                    <*const _>::cast(
                        physical_resources.try_swapchain_protected(physical_id, &guard)?,
                    )
                }
                _ => unreachable!(),
            };
        }

        let len = virtual_resources.physical_map.len() as u32;

        Ok(ResourceMap {
            virtual_resources,
            physical_resources,
            map,
            len,
            guard,
        })
    }

    #[doc(hidden)]
    #[inline]
    pub fn insert<R: Resource>(
        &mut self,
        virtual_id: Id<R>,
        physical_id: Id<R>,
    ) -> Result<(), InvalidSlotError> {
        R::insert(self, virtual_id, physical_id)
    }

    /// Inserts a mapping from the [virtual buffer resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id`.
    ///
    /// # Panics
    ///
    /// - Panics if the physical resource doesn't match the virtual resource.
    /// - Panics if the physical resource already has a mapping from another virtual resource.
    #[inline]
    pub fn insert_buffer(
        &mut self,
        virtual_id: Id<Buffer>,
        physical_id: Id<Buffer>,
    ) -> Result<(), InvalidSlotError> {
        self.virtual_resources.get(virtual_id.erase())?;

        let state = self
            .physical_resources
            .try_buffer_protected(physical_id, &self.guard)?;

        assert_eq!(
            state.buffer().sharing().is_exclusive(),
            virtual_id.is_exclusive(),
        );

        let ptr = <*const _>::cast(state);
        let is_duplicate = self.map.contains(&ptr);

        // SAFETY: We checked that `virtual_id` is present in `self.virtual_resources` above, and
        // since we initialized `self.map` with a length at least that of `self.virtual_resources`,
        // the index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if *slot != ptr {
            assert!(!is_duplicate);
        }

        if slot.is_null() {
            self.len += 1;
        }

        *slot = ptr;

        Ok(())
    }

    /// Inserts a mapping from the [virtual buffer resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id` without doing any checks.
    ///
    /// # Safety
    ///
    /// - `virtual_id` must be a valid virtual resource ID.
    /// - `physical_id` must be a valid physical resource ID.
    /// - The physical resource must match the virtual resource.
    /// - The physical resource must not have a mapping from another virtual resource.
    #[inline]
    pub unsafe fn insert_buffer_unchecked(
        &mut self,
        virtual_id: Id<Buffer>,
        physical_id: Id<Buffer>,
    ) {
        // SAFETY: The caller must ensure that `physical_id` is a valid ID.
        let state = unsafe {
            self.physical_resources
                .buffer_unchecked_protected(physical_id, &self.guard)
        };

        // SAFETY: The caller must ensure that `virtual_id` is a valid virtual ID, and since we
        // initialized `self.map` with a length at least that of `self.virtual_resources`, the
        // index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if slot.is_null() {
            self.len += 1;
        }

        *slot = <*const _>::cast(state);
    }

    /// Inserts a mapping from the [virtual image resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id`.
    ///
    /// # Panics
    ///
    /// - Panics if the physical resource doesn't match the virtual resource.
    /// - Panics if the physical resource already has a mapping from another virtual resource.
    /// - Panics if `virtual_id` refers to a swapchain image.
    #[inline]
    pub fn insert_image(
        &mut self,
        virtual_id: Id<Image>,
        physical_id: Id<Image>,
    ) -> Result<(), InvalidSlotError> {
        assert_ne!(virtual_id.object_type(), ObjectType::Swapchain);

        self.virtual_resources.get(virtual_id.erase())?;

        let state = self
            .physical_resources
            .try_image_protected(physical_id, &self.guard)?;

        assert_eq!(
            state.image().sharing().is_exclusive(),
            virtual_id.is_exclusive(),
        );

        let ptr = <*const _>::cast(state);
        let is_duplicate = self.map.contains(&ptr);

        // SAFETY: We checked that `virtual_id` is present in `self.virtual_resources` above, and
        // since we initialized `self.map` with a length at least that of `self.virtual_resources`,
        // the index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if *slot != ptr {
            assert!(!is_duplicate);
        }

        if slot.is_null() {
            self.len += 1;
        }

        *slot = ptr;

        Ok(())
    }

    /// Inserts a mapping from the [virtual image resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id` without doing any checks.
    ///
    /// # Safety
    ///
    /// - `virtual_id` must be a valid virtual resource ID.
    /// - `physical_id` must be a valid physical resource ID.
    /// - The physical resource must match the virtual resource.
    /// - The physical resource must not have a mapping from another virtual resource.
    #[inline]
    pub unsafe fn insert_image_unchecked(&mut self, virtual_id: Id<Image>, physical_id: Id<Image>) {
        // SAFETY: The caller must ensure that `physical_id` is a valid ID.
        let state = unsafe {
            self.physical_resources
                .image_unchecked_protected(physical_id, &self.guard)
        };

        // SAFETY: The caller must ensure that `virtual_id` is a valid virtual ID, and since we
        // initialized `self.map` with a length at least that of `self.virtual_resources`, the
        // index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if slot.is_null() {
            self.len += 1;
        }

        *slot = <*const _>::cast(state);
    }

    /// Inserts a mapping from the [virtual swapchain resource] corresponding to `virtual_id` to
    /// the physical resource corresponding to `physical_id`.
    ///
    /// # Panics
    ///
    /// - Panics if the physical resource doesn't match the virtual resource.
    /// - Panics if the physical resource already has a mapping from another virtual resource.
    #[inline]
    pub fn insert_swapchain(
        &mut self,
        virtual_id: Id<Swapchain>,
        physical_id: Id<Swapchain>,
    ) -> Result<(), InvalidSlotError> {
        self.virtual_resources.get(virtual_id.erase())?;

        let state = self
            .physical_resources
            .try_swapchain_protected(physical_id, &self.guard)?;

        assert_eq!(
            state.swapchain().image_sharing().is_exclusive(),
            virtual_id.is_exclusive(),
        );

        let ptr = <*const _>::cast(state);
        let is_duplicate = self.map.contains(&ptr);

        // SAFETY: We checked that `virtual_id` is present in `self.virtual_resources` above, and
        // since we initialized `self.map` with a length at least that of `self.virtual_resources`,
        // the index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if *slot != ptr {
            assert!(!is_duplicate);
        }

        if slot.is_null() {
            self.len += 1;
        }

        *slot = ptr;

        Ok(())
    }

    /// Inserts a mapping from the [virtual swapchain resource] corresponding to `virtual_id` to
    /// the physical resource corresponding to `physical_id` without doing any checks.
    ///
    /// # Safety
    ///
    /// - `virtual_id` must be a valid virtual resource ID.
    /// - `physical_id` must be a valid physical resource ID.
    /// - The physical resource must match the virtual resource.
    /// - The physical resource must not have a mapping from another virtual resource.
    #[inline]
    pub unsafe fn insert_swapchain_unchecked(
        &mut self,
        virtual_id: Id<Swapchain>,
        physical_id: Id<Swapchain>,
    ) {
        // SAFETY: The caller must ensure that `physical_id` is a valid ID.
        let state = unsafe {
            self.physical_resources
                .swapchain_unchecked_protected(physical_id, &self.guard)
        };

        // SAFETY: The caller must ensure that `virtual_id` is a valid virtual ID, and since we
        // initialized `self.map` with a length at least that of `self.virtual_resources`, the
        // index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if slot.is_null() {
            self.len += 1;
        }

        *slot = <*const _>::cast(state);
    }

    pub(crate) fn virtual_resources(&self) -> &super::Resources {
        self.virtual_resources
    }

    /// Returns the `Resources` collection.
    #[inline]
    #[must_use]
    pub fn resources(&self) -> &Arc<Resources> {
        &self.physical_resources
    }

    /// Returns the number of mappings in the map.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Returns `true` if the map maps every virtual resource.
    #[inline]
    #[must_use]
    pub fn is_exhaustive(&self) -> bool {
        // By our own invariant, the map can only contain mappings for virtual resources that are
        // present in `self.virtual_resources`. It follows then, that when the length of `self` is
        // that of `self.virtual_resources`, that the virtual resources are mapped exhaustively.
        self.len() == self.virtual_resources.len()
    }

    pub(crate) fn guard(&self) -> &hyaline::Guard<'_> {
        &self.guard
    }

    pub(crate) unsafe fn buffer(&self, id: Id<Buffer>) -> Result<&BufferState, InvalidSlotError> {
        self.virtual_resources.get(id.erase())?;

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        Ok(unsafe { self.buffer_unchecked(id) })
    }

    pub(crate) unsafe fn buffer_unchecked(&self, id: Id<Buffer>) -> &BufferState {
        assert_unsafe_precondition!(
            self.virtual_resources.get(id.erase()).is_ok(),
            "`id` must be a valid virtual buffer ID",
        );

        // SAFETY: The caller must ensure that `id` is a valid virtual ID.
        let &slot = unsafe { self.map.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        unsafe { &*slot.cast::<BufferState>() }
    }

    pub(crate) unsafe fn image(&self, id: Id<Image>) -> Result<&ImageState, InvalidSlotError> {
        self.virtual_resources.get(id.erase())?;

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        Ok(unsafe { self.image_unchecked(id) })
    }

    pub(crate) unsafe fn image_unchecked(&self, id: Id<Image>) -> &ImageState {
        assert_unsafe_precondition!(
            self.virtual_resources.get(id.erase()).is_ok() && id.is::<Image>(),
            "`id` must be a valid virtual image ID",
        );

        // SAFETY: The caller must ensure that `id` is a valid virtual ID.
        let &slot = unsafe { self.map.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        unsafe { &*slot.cast::<ImageState>() }
    }

    pub(crate) unsafe fn swapchain(
        &self,
        id: Id<Swapchain>,
    ) -> Result<&SwapchainState, InvalidSlotError> {
        self.virtual_resources.get(id.erase())?;

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        Ok(unsafe { self.swapchain_unchecked(id) })
    }

    pub(crate) unsafe fn swapchain_unchecked(&self, id: Id<Swapchain>) -> &SwapchainState {
        assert_unsafe_precondition!(
            self.virtual_resources.get(id.erase()).is_ok(),
            "`id` must be a valid virtual swapchain ID",
        );

        // SAFETY: The caller must ensure that `id` is a valid virtual ID.
        let &slot = unsafe { self.map.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        unsafe { &*slot.cast::<SwapchainState>() }
    }
}

unsafe impl DeviceOwned for ResourceMap<'_> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.physical_resources.device()
    }
}

pub trait Resource: Sized {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError>;
}

impl Resource for Buffer {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError> {
        map.insert_buffer(virtual_id, physical_id)
    }
}

impl Resource for Image {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError> {
        map.insert_image(virtual_id, physical_id)
    }
}

impl Resource for Swapchain {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError> {
        map.insert_swapchain(virtual_id, physical_id)
    }
}

/// Creates a [`ResourceMap`] containing the given mappings.
#[macro_export]
macro_rules! resource_map {
    ($executable:expr $(, $virtual_id:expr => $physical_id:expr)* $(,)?) => {
        match $crate::graph::ResourceMap::new($executable) {
            #[allow(unused_mut)]
            ::std::result::Result::Ok(mut map) => {
                $(if let ::std::result::Result::Err(err) = map.insert($virtual_id, $physical_id) {
                    ::std::result::Result::Err(err)
                } else)* {
                    ::std::result::Result::Ok::<_, $crate::InvalidSlotError>(map)
                }
            }
            ::std::result::Result::Err(err) => ::std::result::Result::Err(err),
        }
    };
}

type Result<T = (), E = Validated<ExecuteError>> = ::std::result::Result<T, E>;

/// Error that can happen when [executing] an [`ExecutableTaskGraph`].
///
/// [executing]: ExecutableTaskGraph::execute
#[derive(Debug)]
pub enum ExecuteError {
    Task {
        node_index: NodeIndex,
        error: TaskError,
    },
    Swapchain {
        swapchain_id: Id<Swapchain>,
        error: VulkanError,
    },
    VulkanError(VulkanError),
}

impl From<ExecuteError> for Validated<ExecuteError> {
    fn from(err: ExecuteError) -> Self {
        Self::Error(err)
    }
}

impl fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task { node_index, .. } => {
                write!(f, "an error occurred while executing task {node_index:?}")
            }
            Self::Swapchain { swapchain_id, .. } => write!(
                f,
                "an error occurred while using swapchain {swapchain_id:?}",
            ),
            Self::VulkanError(_) => f.write_str("a runtime error occurred"),
        }
    }
}

impl Error for ExecuteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Task { error, .. } => Some(error),
            Self::Swapchain { error, .. } => Some(error),
            Self::VulkanError(err) => Some(err),
        }
    }
}
