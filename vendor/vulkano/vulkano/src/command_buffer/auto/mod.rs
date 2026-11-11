//! Contains `AutoCommandBufferBuilder` and the built types `PrimaryCommandBuffer` and
//! `SecondaryCommandBuffer`.
//!
//! # How pipeline stages work in Vulkan
//!
//! Imagine you create a command buffer that contains 10 dispatch commands, and submit that command
//! buffer. According to the Vulkan specs, the implementation is free to execute the 10 commands
//! simultaneously.
//!
//! Now imagine that the command buffer contains 10 draw commands instead. Contrary to the dispatch
//! commands, the draw pipeline contains multiple stages: draw indirect, vertex input, vertex
//! shader, ..., fragment shader, late fragment test, color output. When there are multiple stages,
//! the implementations must start and end the stages in order. In other words it can start the
//! draw indirect stage of all 10 commands, then start the vertex input stage of all 10 commands,
//! and so on. But it can't for example start the fragment shader stage of a command before
//! starting the vertex shader stage of another command. Same thing for ending the stages in the
//! right order.
//!
//! Depending on the type of the command, the pipeline stages are different. Compute shaders use
//! the compute stage, while transfer commands use the transfer stage. The compute and transfer
//! stages aren't ordered.
//!
//! When you submit multiple command buffers to a queue, the implementation doesn't do anything in
//! particular and behaves as if the command buffers were appended to one another. Therefore if you
//! submit a command buffer with 10 dispatch commands, followed with another command buffer with 5
//! dispatch commands, then the implementation can perform the 15 commands simultaneously.
//!
//! ## Introducing barriers
//!
//! In some situations this is not the desired behavior. If you add a command that writes to a
//! buffer followed with another command that reads that buffer, you don't want them to execute
//! simultaneously. Instead you want the second one to wait until the first one is finished. This
//! is done by adding a pipeline barrier between the two commands.
//!
//! A pipeline barriers has a source stage and a destination stage (plus various other things).
//! A barrier represents a split in the list of commands. When you add it, the stages of the
//! commands before the barrier corresponding to the source stage of the barrier, must finish
//! before the stages of the commands after the barrier corresponding to the destination stage of
//! the barrier can start.
//!
//! For example if you add a barrier that transitions from the compute stage to the compute stage,
//! then the compute stage of all the commands before the barrier must end before the compute stage
//! of all the commands after the barrier can start. This is appropriate for the example about
//! writing then reading the same buffer.
//!
//! ## Batching barriers
//!
//! Since barriers are "expensive" (as the queue must block), vulkano attempts to group as many
//! pipeline barriers as possible into one.
//!
//! Adding a command to an `AutoCommandBufferBuilder` does not immediately add it to the underlying
//! command buffer builder. Instead the command is added to a queue, and the builder keeps a
//! prototype of a barrier that must be added before the commands in the queue are flushed.
//!
//! Whenever you add a command, the builder will find out whether a barrier is needed before the
//! command. If so, it will try to merge this barrier with the prototype and add the command to the
//! queue. If not possible, the queue will be entirely flushed and the command added to a fresh new
//! queue with a fresh new barrier prototype.

pub use self::builder::*;
pub(in crate::command_buffer) use self::builder::{
    BeginRenderPassState, BeginRenderingState, QueryState, RenderPassState,
    RenderPassStateAttachments, RenderPassStateType, SetOrPush,
};
use super::{
    sys::{CommandBuffer, RecordingCommandBuffer},
    CommandBufferInheritanceInfo, CommandBufferResourcesUsage, CommandBufferState,
    CommandBufferUsage, PrimaryCommandBufferAbstract, ResourceInCommand,
    SecondaryCommandBufferAbstract, SecondaryCommandBufferResourcesUsage, SecondaryResourceUseRef,
};
use crate::{
    buffer::Subbuffer,
    device::{Device, DeviceOwned},
    image::{Image, ImageLayout, ImageSubresourceRange},
    sync::PipelineStageAccessFlags,
    DeviceSize, ValidationError, VulkanObject,
};
use ash::vk;
use parking_lot::{Mutex, MutexGuard};
use std::{
    fmt::{Debug, Error as FmtError, Formatter},
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

mod builder;
pub(crate) mod commands;

pub struct PrimaryAutoCommandBuffer {
    inner: CommandBuffer,
    _keep_alive_objects: Vec<Box<dyn Fn(&mut RecordingCommandBuffer) + Send + Sync + 'static>>,
    resources_usage: CommandBufferResourcesUsage,
    state: Mutex<CommandBufferState>,
}

unsafe impl VulkanObject for PrimaryAutoCommandBuffer {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for PrimaryAutoCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl Debug for PrimaryAutoCommandBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("PrimaryAutoCommandBuffer")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

unsafe impl PrimaryCommandBufferAbstract for PrimaryAutoCommandBuffer {
    #[inline]
    fn as_raw(&self) -> &CommandBuffer {
        &self.inner
    }

    #[inline]
    fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index()
    }

    #[inline]
    fn usage(&self) -> CommandBufferUsage {
        self.inner.usage()
    }

    fn state(&self) -> MutexGuard<'_, CommandBufferState> {
        self.state.lock()
    }

    fn resources_usage(&self) -> &CommandBufferResourcesUsage {
        &self.resources_usage
    }
}

pub struct SecondaryAutoCommandBuffer {
    inner: CommandBuffer,
    _keep_alive_objects: Vec<Box<dyn Fn(&mut RecordingCommandBuffer) + Send + Sync + 'static>>,
    resources_usage: SecondaryCommandBufferResourcesUsage,
    submit_state: SubmitState,
}

unsafe impl VulkanObject for SecondaryAutoCommandBuffer {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for SecondaryAutoCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl Debug for SecondaryAutoCommandBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("SecondaryAutoCommandBuffer")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

unsafe impl SecondaryCommandBufferAbstract for SecondaryAutoCommandBuffer {
    #[inline]
    fn as_raw(&self) -> &CommandBuffer {
        &self.inner
    }

    #[inline]
    fn usage(&self) -> CommandBufferUsage {
        self.inner.usage()
    }

    #[inline]
    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        self.inner.inheritance_info().unwrap()
    }

    fn lock_record(&self) -> Result<(), Box<ValidationError>> {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(Box::new(ValidationError {
                        problem: "the command buffer was created with the \
                            `CommandBufferUsage::OneTimeSubmit` usage, but \
                            it was already submitted before"
                            .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(Box::new(ValidationError {
                        problem: "the command buffer was created with the \
                            `CommandBufferUsage::MultipleSubmit` usage, but \
                            it is currently being executed"
                            .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
            SubmitState::Concurrent => (),
        };

        Ok(())
    }

    unsafe fn unlock(&self) {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                debug_assert!(already_submitted.load(Ordering::SeqCst));
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::SeqCst);
                debug_assert!(old_val);
            }
            SubmitState::Concurrent => (),
        };
    }

    fn resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage {
        &self.resources_usage
    }
}

// Whether the command buffer can be submitted.
#[derive(Debug)]
enum SubmitState {
    // The command buffer was created with the "SimultaneousUse" flag. Can always be submitted at
    // any time.
    Concurrent,

    // The command buffer can only be submitted once simultaneously.
    ExclusiveUse {
        // True if the command buffer is current in use by the GPU.
        in_use: AtomicBool,
    },

    // The command buffer can only ever be submitted once.
    OneTime {
        // True if the command buffer has already been submitted once and can be no longer be
        // submitted.
        already_submitted: AtomicBool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::command_buffer) struct ResourceUseRef2 {
    pub(in crate::command_buffer) resource_in_command: ResourceInCommand,
    pub(in crate::command_buffer) secondary_use_ref: Option<SecondaryResourceUseRef>,
}

impl From<ResourceInCommand> for ResourceUseRef2 {
    #[inline]
    fn from(resource_in_command: ResourceInCommand) -> Self {
        Self {
            resource_in_command,
            secondary_use_ref: None,
        }
    }
}

/// Type of resource whose state is to be tracked.
#[derive(Clone, Debug)]
pub(super) enum Resource {
    Buffer {
        buffer: Subbuffer<[u8]>,
        range: Range<DeviceSize>,
        memory_access: PipelineStageAccessFlags,
    },
    Image {
        image: Arc<Image>,
        subresource_range: ImageSubresourceRange,
        memory_access: PipelineStageAccessFlags,
        start_layout: ImageLayout,
        end_layout: ImageLayout,
    },
}

struct CommandInfo {
    name: &'static str,
    used_resources: Vec<(ResourceUseRef2, Resource)>,
    render_pass: RenderPassCommand,
}

#[derive(Debug)]
enum RenderPassCommand {
    None,
    Begin,
    End,
}

impl Debug for CommandInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.write_str(self.name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        buffer::{Buffer, BufferCreateFlags, BufferCreateInfo, BufferUsage},
        command_buffer::{
            allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
            pool::CommandPoolCreateFlags,
            AutoCommandBufferBuilder, BufferCopy, CommandBufferUsage, CopyBufferInfoTyped,
            PrimaryCommandBufferAbstract,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator,
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            DescriptorImageInfo, DescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceFeatures,
            QueueCreateFlags, QueueCreateInfo,
        },
        image::sampler::{Sampler, SamplerCreateInfo},
        memory::{
            allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
            MemoryPropertyFlags,
        },
        pipeline::{layout::PipelineLayoutCreateInfo, PipelineBindPoint, PipelineLayout},
        shader::ShaderStages,
        sync::GpuFuture,
    };
    use std::sync::Arc;

    #[test]
    fn basic_creation() {
        let (device, queue) = gfx_dev_and_queue!();

        let allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));

        AutoCommandBufferBuilder::primary(
            allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
    }

    #[test]
    fn copy_buffer_dimensions() {
        let instance = instance!();

        let physical_device = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter().next().unwrap(),
            Err(_) => return,
        };

        let (device, queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let [queue] = queues.try_into().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let source = Buffer::from_iter(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [1_u32, 2].iter().copied(),
        )
        .unwrap();

        let destination = Buffer::from_iter(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0_u32, 10, 20, 3, 4].iter().copied(),
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut cbb = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.copy_buffer(CopyBufferInfoTyped {
            regions: [BufferCopy {
                src_offset: 0,
                dst_offset: 1,
                size: 2,
                ..Default::default()
            }]
            .into(),
            ..CopyBufferInfoTyped::new(source, destination.clone())
        })
        .unwrap();

        let cb = cbb.build().unwrap();

        let future = cb
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = destination.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 2, 3, 4]);
    }

    #[test]
    fn secondary_nonconcurrent_conflict() {
        let (device, queue) = gfx_dev_and_queue!();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 1,
                ..Default::default()
            },
        ));

        // Make a secondary CB that doesn't support simultaneous use.
        let builder = AutoCommandBufferBuilder::secondary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            Default::default(),
        )
        .unwrap();
        let secondary = builder.build().unwrap();

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                cb_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Add the secondary a first time
            builder.execute_commands(secondary.clone()).unwrap();

            // Recording the same non-concurrent secondary command buffer twice into the same
            // primary is an error.
            assert!(builder.execute_commands(secondary.clone()).is_err());
        }

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                cb_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();
            builder.execute_commands(secondary.clone()).unwrap();
            let cb1 = builder.build().unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Recording the same non-concurrent secondary command buffer into multiple
            // primaries is an error.
            assert!(builder.execute_commands(secondary.clone()).is_err());

            drop(cb1);

            // Now that the first cb is dropped, we should be able to record.
            builder.execute_commands(secondary).unwrap();
        }
    }

    #[test]
    fn buffer_self_copy_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let source = Buffer::from_iter(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 2,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::new(source.clone(), source.clone())
            })
            .unwrap();

        let cb = builder.build().unwrap();

        let future = cb
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = source.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 0, 1]);
    }

    #[test]
    fn build_protected() {
        let instance = instance!();

        let protected_memory_feature = DeviceFeatures {
            protected_memory: true,
            ..DeviceFeatures::empty()
        };
        let select = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter(),
            Err(_) => return,
        }
        .filter(|p| p.supported_features().contains(&protected_memory_feature))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| {
                    q.queue_flags
                        .intersects(crate::device::QueueFlags::GRAPHICS)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        });

        let (physical_device, queue_family_index) = match select {
            Some(x) => x,
            None => return, // skips the test if no device supported protected_memory
        };

        let (device, queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_features: &DeviceFeatures {
                    protected_memory: true,
                    ..DeviceFeatures::empty()
                },
                queue_create_infos: &[QueueCreateInfo {
                    flags: QueueCreateFlags::PROTECTED,
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let [queue] = queues.try_into().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let my_buffer = Buffer::new_slice::<u32>(
            &memory_allocator,
            &BufferCreateInfo {
                flags: BufferCreateFlags::PROTECTED,
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::PROTECTED,
                    preferred_flags: MemoryPropertyFlags::empty(),
                    not_preferred_flags: MemoryPropertyFlags::empty(),
                },
                ..Default::default()
            },
            1024,
        )
        .expect("Failed to create protected buffer");

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo {
                command_pool_create_flags: CommandPoolCreateFlags::PROTECTED,
                ..Default::default()
            },
        ));
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.fill_buffer(my_buffer.clone(), 0u32).unwrap();

        let cb = builder.build().unwrap();
        assert!(cb.inner.is_protected());

        let future = cb
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

    #[test]
    fn buffer_self_copy_not_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let source = Buffer::from_iter(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        assert!(builder
            .copy_buffer(CopyBufferInfoTyped {
                regions: [BufferCopy {
                    src_offset: 0,
                    dst_offset: 1,
                    size: 2,
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferInfoTyped::new(source.clone(), source)
            })
            .is_err());
    }

    #[test]
    fn secondary_conflicting_writes() {
        let (device, queue) = gfx_dev_and_queue!();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 1,
                ..Default::default()
            },
        ));
        let cbb = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        // Create a tiny test buffer
        let buffer = Buffer::from_data(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            0u32,
        )
        .unwrap();

        cbb.build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // Two secondary command buffers that both write to the buffer
        let secondary = (0..2)
            .map(|_| {
                let mut builder = AutoCommandBufferBuilder::secondary(
                    cb_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::SimultaneousUse,
                    Default::default(),
                )
                .unwrap();
                builder
                    .fill_buffer(buffer.clone().into_slice(), 42)
                    .unwrap();
                builder.build().unwrap()
            })
            .collect::<Vec<_>>();

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                cb_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Add both secondary command buffers using separate execute_commands calls.
            secondary.iter().cloned().for_each(|secondary| {
                unsafe {
                    builder.execute_commands_unchecked([secondary as _].into_iter().collect())
                };
            });

            let _primary = builder.build().unwrap();
            /*
            let names = primary._commands.iter().map(|c| c.name).collect::<Vec<_>>();

            // Ensure that the builder added a barrier between the two writes
            assert_eq!(&names, &["execute_commands", "execute_commands"]);
            assert_eq!(&primary._barriers, &[0, 1]);
             */
        }

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Add a single execute_commands for all secondary command buffers at once
            unsafe {
                builder.execute_commands_unchecked(
                    secondary
                        .into_iter()
                        .map(|secondary| secondary as _)
                        .collect(),
                )
            };
        }
    }

    #[test]
    fn vertex_buffer_binding() {
        let (device, queue) = gfx_dev_and_queue!();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut sync = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let buf = Buffer::from_data(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            0u32,
        )
        .unwrap();
        unsafe { sync.bind_vertex_buffers_unchecked(1, buf) };

        assert!(!sync.builder_state.vertex_buffers.contains_key(&0));
        assert!(sync.builder_state.vertex_buffers.contains_key(&1));
        assert!(!sync.builder_state.vertex_buffers.contains_key(&2));
    }

    #[test]
    fn descriptor_set_binding() {
        let (device, queue) = gfx_dev_and_queue!();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut sync = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
        let set_layout = DescriptorSetLayout::new(
            &device,
            &DescriptorSetLayoutCreateInfo {
                bindings: &[DescriptorSetLayoutBinding {
                    stages: ShaderStages::all_graphics(),
                    ..DescriptorSetLayoutBinding::new(DescriptorType::Sampler)
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let pipeline_layout = PipelineLayout::new(
            &device,
            &PipelineLayoutCreateInfo {
                set_layouts: &[&set_layout, &set_layout],
                ..Default::default()
            },
        )
        .unwrap();

        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));

        let set = DescriptorSet::new(
            &ds_allocator,
            &set_layout,
            &[WriteDescriptorSet::image(
                0,
                &DescriptorImageInfo {
                    sampler: Some(
                        &Sampler::new(&device, &SamplerCreateInfo::simple_repeat_linear()).unwrap(),
                    ),
                    ..Default::default()
                },
            )],
            &[],
        )
        .unwrap();

        unsafe {
            sync.bind_descriptor_sets_unchecked(
                PipelineBindPoint::Graphics,
                pipeline_layout.clone(),
                1,
                set.clone(),
            )
        };

        assert!(!sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Compute)
            .is_some_and(|state| state.descriptor_sets.contains_key(&0)));
        assert!(!sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&0)));
        assert!(sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&1)));
        assert!(!sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&2)));

        unsafe {
            sync.bind_descriptor_sets_unchecked(
                PipelineBindPoint::Graphics,
                pipeline_layout,
                0,
                set,
            )
        };

        assert!(sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&0)));
        assert!(sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&1)));

        let pipeline_layout = PipelineLayout::new(
            &device,
            &PipelineLayoutCreateInfo {
                set_layouts: &[
                    &DescriptorSetLayout::new(&device, &Default::default()).unwrap(),
                    &set_layout,
                ],
                ..Default::default()
            },
        )
        .unwrap();

        let set = DescriptorSet::new(
            &ds_allocator,
            &set_layout,
            &[WriteDescriptorSet::image(
                0,
                &DescriptorImageInfo {
                    sampler: Some(
                        &Sampler::new(&device, &SamplerCreateInfo::simple_repeat_linear()).unwrap(),
                    ),
                    ..Default::default()
                },
            )],
            &[],
        )
        .unwrap();

        unsafe {
            sync.bind_descriptor_sets_unchecked(
                PipelineBindPoint::Graphics,
                pipeline_layout,
                1,
                set,
            )
        };

        assert!(!sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&0)));
        assert!(sync
            .builder_state
            .descriptor_sets
            .get(&PipelineBindPoint::Graphics)
            .is_some_and(|state| state.descriptor_sets.contains_key(&1)));
    }
}
