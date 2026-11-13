// FIXME: host read barriers

use super::{
    Attachments, BarrierIndex, ExecutableTaskGraph, Instruction, NodeIndex, NodeInner,
    RenderPassIndex, ResourceAccess, ResourceAccesses, SemaphoreIndex, Submission, TaskGraph,
};
use crate::{linear_map::LinearMap, resource::Flight, Id, QueueFamilyType};
use ash::vk;
use smallvec::{smallvec, SmallVec};
use std::{
    cell::{Cell, RefCell},
    cmp,
    error::Error,
    fmt,
    mem::ManuallyDrop,
    ops::Range,
    sync::Arc,
};
use vulkano::{
    device::{Device, DeviceOwned, Queue, QueueFlags},
    format::Format,
    image::{sampler::ComponentMapping, Image, ImageLayout, ImageUsage},
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, RenderPass, RenderPassCreateInfo, SubpassDependency, SubpassDescription,
    },
    swapchain::Swapchain,
    sync::{semaphore::Semaphore, AccessFlags, DependencyFlags, PipelineStages},
    Validated, VulkanError,
};

impl<W: ?Sized> TaskGraph<W> {
    /// Compiles the task graph into an executable form.
    ///
    /// # Safety
    ///
    /// - There must be no conflicting device accesses in task nodes with no path between them.
    /// - There must be no accesses that are incompatible with the queue family type of the task
    ///   node.
    /// - There must be no accesses that are unsupported by the device.
    ///
    /// # Panics
    ///
    /// - Panics if `compile_info.queues` is empty.
    /// - Panics if the device of any queue in `compile_info.queues` or
    ///   `compile_info.present_queue` is not the same as that of `self`.
    /// - Panics if `compile_info.queues` contains duplicate queue families.
    /// - Panics if `compile_info.present_queue` is `None` and the task graph uses any swapchains.
    ///
    /// # Errors
    ///
    /// In order to get a successful compilation, the graph must satisfy the following conditions:
    /// - It must be [weakly connected]: every node must be able to reach every other node when
    ///   disregarding the direction of the edges.
    /// - It must have no [directed cycles]: if you were to walk starting from any node following
    ///   the direction of the edges, there must be no way to end up at the node you started at.
    ///
    /// [weakly connected]: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_vertices_and_graphs
    /// [directed cycles]: https://en.wikipedia.org/wiki/Cycle_(graph_theory)#Directed_circuit_and_directed_cycle
    pub unsafe fn compile(
        mut self,
        compile_info: &CompileInfo<'_>,
    ) -> Result<ExecutableTaskGraph<W>, CompileError<W>> {
        let &CompileInfo {
            queues,
            present_queue,
            flight_id,
            _ne: _,
        } = compile_info;

        assert_ne!(queues.len(), 0, "expected to be given at least one queue");

        let device = &self.device().clone();

        for queue in queues {
            assert_eq!(queue.device(), device);
            assert_eq!(
                queues
                    .iter()
                    .filter(|q| q.queue_family_index() == queue.queue_family_index())
                    .count(),
                1,
                "expected each queue in `compile_info.queues` to be from a unique queue family",
            );
        }

        if let Some(present_queue) = &present_queue {
            assert_eq!(present_queue.device(), device);
        }

        if !self.is_weakly_connected() {
            return Err(CompileError::new(self, CompileErrorKind::Unconnected));
        }

        let topological_order = match self.topological_sort() {
            Ok(topological_order) => topological_order,
            Err(kind) => return Err(CompileError::new(self, kind)),
        };
        unsafe { self.dependency_levels(&topological_order) };
        let queue_family_indices =
            match unsafe { self.queue_family_indices(device, queues, &topological_order) } {
                Ok(queue_family_indices) => queue_family_indices,
                Err(kind) => return Err(CompileError::new(self, kind)),
            };
        let mut queues_by_queue_family_index: SmallVec<[_; 8]> =
            smallvec![None; *queue_family_indices.iter().max().unwrap() as usize + 1];

        for &queue in queues {
            if let Some(x) =
                queues_by_queue_family_index.get_mut(queue.queue_family_index() as usize)
            {
                *x = Some(queue);
            }
        }

        let (builder, last_swapchain_accesses) =
            match unsafe { self.lower(present_queue, &topological_order) } {
                Ok(x) => x,
                Err(kind) => return Err(CompileError::new(self, kind)),
            };
        let IntermediateRepresentationBuilder {
            submissions,
            nodes,
            prev_accesses: last_accesses,
            prev_node_indices: _,
            semaphore_count,
            render_passes,
            is_render_pass_instance_active: _,
            pre_present_queue_family_ownership_transfers,
        } = builder;

        let mut builder = FinalRepresentationBuilder::new(present_queue);
        let mut prev_submission_end = 0;
        let mut submission_index = 0;

        while prev_submission_end < topological_order.len() {
            let submission_state = &submissions[submission_index];
            builder.initial_pipeline_barrier(&submission_state.initial_barriers);

            for (i, &node_index) in
                (prev_submission_end..).zip(&topological_order[prev_submission_end..])
            {
                let node = unsafe { self.nodes.node_unchecked_mut(node_index) };
                let NodeInner::Task(task_node) = &mut node.inner else {
                    unreachable!();
                };
                let queue_family_index = task_node.queue_family_index;
                let node_state = &nodes[node_index as usize];

                for &(swapchain_id, stage_mask) in &node_state.wait_acquire {
                    builder.wait_acquire(swapchain_id, stage_mask);
                }

                for &semaphore_index in &node_state.wait_semaphores {
                    builder.wait_semaphore(semaphore_index);
                }

                builder.pipeline_barrier(&node_state.start_barriers);

                if let Some(subpass) = node_state.subpass {
                    let render_pass_state = &render_passes[subpass.render_pass_index];

                    if node_index == render_pass_state.first_node_index {
                        builder.begin_render_pass(subpass.render_pass_index);
                    }

                    builder.next_subpass(subpass.subpass_index);

                    if !node_state.clear_attachments.is_empty() {
                        builder.clear_attachments(
                            node_index,
                            subpass.render_pass_index,
                            &node_state.clear_attachments,
                        );
                    }
                }

                builder.execute_task(node_index);

                if let Some(subpass) = node_state.subpass {
                    let render_pass_state = &render_passes[subpass.render_pass_index];

                    if node_index == render_pass_state.last_node_index {
                        builder.end_render_pass();
                    }
                }

                builder.pipeline_barrier(&node_state.end_barriers);

                for &semaphore_index in &node_state.signal_semaphores {
                    builder.signal_semaphore(semaphore_index);
                }

                for &(swapchain_id, stage_mask) in &node_state.signal_pre_present {
                    builder.signal_pre_present(swapchain_id, stage_mask);
                }

                for &(swapchain_id, stage_mask) in &node_state.signal_present {
                    builder.signal_present(swapchain_id, stage_mask);
                }

                let should_submit = if let Some(&next_node_index) = topological_order.get(i + 1) {
                    let next_node = unsafe { self.nodes.node_unchecked(next_node_index) };
                    let NodeInner::Task(next_task_node) = &next_node.inner else {
                        unreachable!();
                    };

                    next_task_node.queue_family_index != queue_family_index
                } else {
                    true
                };

                if builder.should_flush_submit || should_submit {
                    builder.flush_submit();
                }

                if should_submit {
                    let queue = queues_by_queue_family_index[queue_family_index as usize]
                        .unwrap()
                        .clone();
                    builder.submit(queue);
                    prev_submission_end = i + 1;
                    submission_index += 1;
                    break;
                }
            }
        }

        if !pre_present_queue_family_ownership_transfers.is_empty() {
            for swapchain_id in pre_present_queue_family_ownership_transfers {
                builder.pre_present_acquire_queue_family_ownership(&last_accesses, swapchain_id);
            }

            builder.flush_submit();
            builder.submit(present_queue.unwrap().clone());
        }

        let render_passes = match render_passes
            .into_iter()
            .map(|render_pass_state| create_render_pass(&self.resources, render_pass_state))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(render_passes) => render_passes,
            Err(kind) => return Err(CompileError::new(self, kind)),
        };

        if !render_passes.is_empty() {
            for (id, node) in self.nodes.nodes_mut() {
                let NodeInner::Task(task_node) = &mut node.inner else {
                    unreachable!();
                };

                if let Some(subpass) = nodes[id.index() as usize].subpass {
                    let render_pass = &render_passes[subpass.render_pass_index].render_pass;
                    task_node.subpass = Some(
                        vulkano::render_pass::Subpass::new(
                            render_pass,
                            subpass.subpass_index as u32,
                        )
                        .unwrap(),
                    );
                }
            }
        }

        let semaphores = match (0..semaphore_count)
            // SAFETY: The parameters are valid.
            .map(|_| unsafe { Semaphore::new_unchecked(device, &Default::default()) })
            .collect::<Result<_, _>>()
        {
            Ok(semaphores) => semaphores,
            Err(err) => return Err(CompileError::new(self, CompileErrorKind::VulkanError(err))),
        };

        let swapchains = last_swapchain_accesses.keys().copied().collect();

        Ok(ExecutableTaskGraph {
            graph: ManuallyDrop::new(self),
            flight_id,
            use_swapchain_maintenance1: device.enabled_features().swapchain_maintenance1,
            instructions: builder.instructions,
            submissions: builder.submissions,
            barriers: builder.barriers,
            render_passes: RefCell::new(render_passes),
            clear_attachments: builder.clear_attachments,
            semaphores: RefCell::new(semaphores),
            swapchains,
            present_queue: present_queue.cloned(),
            last_accesses,
            last_frame: Cell::new(None),
            drop_graph: true,
        })
    }

    /// Performs [depth-first search] on the equivalent undirected graph to determine if every node
    /// is visited, meaning the undirected graph is [connected]. If it is, then the directed graph
    /// is [weakly connected]. This property is required because otherwise it could happen that we
    /// end up with a submit that is in no way synchronized with the host.
    ///
    /// [depth-first search]: https://en.wikipedia.org/wiki/Depth-first_search
    /// [connected]: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_vertices_and_graphs
    /// [weakly connected]: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_vertices_and_graphs
    fn is_weakly_connected(&self) -> bool {
        unsafe fn dfs<W: ?Sized>(
            graph: &TaskGraph<W>,
            node_index: NodeIndex,
            visited: &mut [bool],
            visited_count: &mut u32,
        ) {
            let is_visited = &mut visited[node_index as usize];

            if *is_visited {
                return;
            }

            *is_visited = true;
            *visited_count += 1;

            let node = unsafe { graph.nodes.node_unchecked(node_index) };

            for &node_index in node.in_edges.iter().chain(&node.out_edges) {
                unsafe { dfs(graph, node_index, visited, visited_count) };
            }
        }

        let mut visited = vec![false; self.nodes.reserved_len() as usize];
        let mut visited_count = 0;

        if let Some((id, _)) = self.nodes.nodes().next() {
            unsafe { dfs(self, id.index(), &mut visited, &mut visited_count) };
        }

        visited_count == self.nodes.len()
    }

    /// Performs [topological sort using depth-first search]. Returns a vector of node indices in
    /// topological order.
    ///
    /// [topological sort using depth-first search]: https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    fn topological_sort(&self) -> Result<Vec<NodeIndex>, CompileErrorKind> {
        type NodeState = u8;

        const VISITED_BIT: NodeState = 1 << 0;
        const ON_STACK_BIT: NodeState = 1 << 1;

        unsafe fn dfs<W: ?Sized>(
            graph: &TaskGraph<W>,
            node_index: NodeIndex,
            state: &mut [NodeState],
            output: &mut [NodeIndex],
            mut output_index: u32,
        ) -> Result<u32, CompileErrorKind> {
            let node_state = &mut state[node_index as usize];

            if *node_state == VISITED_BIT {
                return Ok(output_index);
            }

            if *node_state == ON_STACK_BIT {
                return Err(CompileErrorKind::Cycle);
            }

            *node_state = ON_STACK_BIT;

            let node = unsafe { graph.nodes.node_unchecked(node_index) };

            for &node_index in &node.out_edges {
                output_index = unsafe { dfs(graph, node_index, state, output, output_index) }?;
            }

            state[node_index as usize] = VISITED_BIT;
            output[output_index as usize] = node_index;

            Ok(output_index.wrapping_sub(1))
        }

        let mut state = vec![0; self.nodes.reserved_len() as usize];
        let mut output = vec![0; self.nodes.len() as usize];
        let mut output_index = self.nodes.len().wrapping_sub(1);

        for (id, _) in self.nodes.nodes() {
            output_index = unsafe { dfs(self, id.index(), &mut state, &mut output, output_index) }?;
        }

        debug_assert_eq!(output_index, u32::MAX);

        Ok(output)
    }

    /// Performs [longest path search] to assign the dependency level index to each task node.
    /// Tasks in the same dependency level don't depend on each other and can therefore be run in
    /// parallel. Returns a vector of dependency levels in topological order indexed by the node's
    /// dependency level index.
    ///
    /// [longest path search]: https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs
    unsafe fn dependency_levels(&mut self, topological_order: &[NodeIndex]) -> Vec<Vec<NodeIndex>> {
        let mut distances = vec![0; self.nodes.reserved_len() as usize];
        let mut max_level = 0;

        for &node_index in topological_order {
            let node = unsafe { self.nodes.node_unchecked(node_index) };

            for &out_node_index in &node.out_edges {
                let new_distance = distances[node_index as usize] + 1;

                if distances[out_node_index as usize] < new_distance {
                    distances[out_node_index as usize] = new_distance;
                    max_level = cmp::max(max_level, new_distance);
                }
            }
        }

        let mut levels = vec![Vec::new(); max_level as usize + 1];

        for (id, node) in self.nodes.nodes_mut() {
            let NodeInner::Task(task_node) = &mut node.inner else {
                unreachable!();
            };

            let level_index = distances[id.index() as usize];
            levels[level_index as usize].push(id.index());
            task_node.dependency_level_index = level_index;
        }

        levels
    }

    /// Assigns a queue family index to each task node. Returns a vector of the used queue family
    /// indices in topological order.
    unsafe fn queue_family_indices(
        &mut self,
        device: &Device,
        queues: &[&Arc<Queue>],
        topological_order: &[NodeIndex],
    ) -> Result<SmallVec<[u32; 3]>, CompileErrorKind> {
        let queue_family_properties = device.physical_device().queue_family_properties();
        let graphics_queue_family_index = queues
            .iter()
            .find(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .map(|q| q.queue_family_index());
        let compute_queue_family_index = queues
            .iter()
            .filter(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .contains(QueueFlags::COMPUTE)
            })
            .min_by_key(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .count()
            })
            .map(|q| q.queue_family_index());
        let transfer_queue_family_index = queues
            .iter()
            .filter(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .contains(QueueFlags::TRANSFER)
            })
            .min_by_key(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .count()
            })
            .map(|q| q.queue_family_index())
            .or(compute_queue_family_index)
            .or(graphics_queue_family_index);

        let mut queue_family_indices = SmallVec::new();

        for &node_index in topological_order {
            let node = unsafe { self.nodes.node_unchecked_mut(node_index) };
            let NodeInner::Task(task_node) = &mut node.inner else {
                unreachable!();
            };

            let queue_family_index = match task_node.queue_family_type() {
                QueueFamilyType::Graphics => graphics_queue_family_index,
                QueueFamilyType::Compute => compute_queue_family_index,
                QueueFamilyType::Transfer => transfer_queue_family_index,
                QueueFamilyType::Specific { index } => queues
                    .iter()
                    .any(|q| q.queue_family_index() == index)
                    .then_some(index),
            }
            .ok_or(CompileErrorKind::InsufficientQueues)?;

            task_node.queue_family_index = queue_family_index;

            if !queue_family_indices.contains(&queue_family_index) {
                queue_family_indices.push(queue_family_index);
            }
        }

        Ok(queue_family_indices)
    }

    /// Lowers the task graph to the intermediate representation.
    // TODO: Cull redundant semaphores.
    unsafe fn lower(
        &mut self,
        present_queue: Option<&Arc<Queue>>,
        topological_order: &[NodeIndex],
    ) -> Result<
        (
            IntermediateRepresentationBuilder,
            LinearMap<Id<Swapchain>, NodeIndex>,
        ),
        CompileErrorKind,
    > {
        let mut builder = IntermediateRepresentationBuilder::new(
            self.nodes.reserved_len(),
            self.resources.reserved_len(),
        );
        let mut prev_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
        let mut last_swapchain_accesses = LinearMap::new();

        for &node_index in topological_order {
            let node = unsafe { self.nodes.node_unchecked(node_index) };
            let NodeInner::Task(task_node) = &node.inner else {
                unreachable!();
            };
            let queue_family_index = task_node.queue_family_index;

            for &out_node_index in &node.out_edges {
                let out_node = unsafe { self.nodes.node_unchecked(out_node_index) };
                let NodeInner::Task(out_task_node) = &out_node.inner else {
                    unreachable!();
                };

                if queue_family_index != out_task_node.queue_family_index {
                    let semaphore_index = builder.semaphore_count;
                    builder.nodes[node_index as usize]
                        .signal_semaphores
                        .push(semaphore_index);
                    builder.nodes[out_node_index as usize]
                        .wait_semaphores
                        .push(semaphore_index);
                    builder.semaphore_count += 1;
                }
            }

            if prev_queue_family_index != queue_family_index {
                builder.submissions.push(SubmissionState::new(node_index));
                builder.is_render_pass_instance_active = false;
            }

            let submission_index = builder.submissions.len() - 1;
            builder.nodes[node_index as usize].submission_index = submission_index;

            if let Some(attachments) = &task_node.attachments {
                builder.subpass(node_index, &task_node.accesses, attachments);
                builder.is_render_pass_instance_active = true;
            } else {
                builder.is_render_pass_instance_active = false;
            }

            for (id, access) in task_node.accesses.iter() {
                let access = ResourceAccess {
                    queue_family_index,
                    ..*access
                };
                builder.resource_access(node_index, id, access);

                if id.is::<Swapchain>() {
                    *last_swapchain_accesses
                        .get_or_insert(unsafe { id.parametrize() }, node_index) = node_index;
                }
            }

            builder.submissions.last_mut().unwrap().last_node_index = node_index;

            prev_queue_family_index = queue_family_index;
        }

        if !last_swapchain_accesses.is_empty() {
            let present_queue = present_queue.expect("expected to be given a present queue");

            for (&swapchain_id, &node_index) in last_swapchain_accesses.iter() {
                builder.swapchain_present(node_index, swapchain_id, present_queue);
            }
        }

        Ok((builder, last_swapchain_accesses))
    }
}

struct IntermediateRepresentationBuilder {
    submissions: Vec<SubmissionState>,
    nodes: Vec<NodeState>,
    prev_accesses: Vec<ResourceAccess>,
    prev_node_indices: Vec<NodeIndex>,
    semaphore_count: usize,
    render_passes: Vec<RenderPassState>,
    is_render_pass_instance_active: bool,
    pre_present_queue_family_ownership_transfers: Vec<Id<Swapchain>>,
}

struct SubmissionState {
    initial_barriers: Vec<super::MemoryBarrier>,
    first_node_index: NodeIndex,
    last_node_index: NodeIndex,
}

#[derive(Clone, Default)]
struct NodeState {
    submission_index: usize,
    wait_acquire: Vec<(Id<Swapchain>, PipelineStages)>,
    wait_semaphores: Vec<SemaphoreIndex>,
    start_barriers: Vec<super::MemoryBarrier>,
    subpass: Option<Subpass>,
    clear_attachments: Vec<Id>,
    end_barriers: Vec<super::MemoryBarrier>,
    signal_semaphores: Vec<SemaphoreIndex>,
    signal_pre_present: Vec<(Id<Swapchain>, PipelineStages)>,
    signal_present: Vec<(Id<Swapchain>, PipelineStages)>,
}

#[derive(Clone, Copy)]
struct Subpass {
    render_pass_index: RenderPassIndex,
    subpass_index: usize,
}

struct RenderPassState {
    framebuffer_id: Id<Framebuffer>,
    attachments: LinearMap<Id, AttachmentState>,
    subpasses: Vec<SubpassState>,
    first_node_index: NodeIndex,
    last_node_index: NodeIndex,
    clear_node_indices: Vec<NodeIndex>,
}

struct AttachmentState {
    first_node_index: NodeIndex,
    first_access: ResourceAccess,
    last_access: ResourceAccess,
    has_reads: bool,
    index: u32,
    clear: bool,
    format: Format,
    component_mapping: ComponentMapping,
    mip_level: u32,
    base_array_layer: u32,
}

struct SubpassState {
    input_attachments: LinearMap<Id, (u32, ImageLayout)>,
    color_attachments: LinearMap<Id, (u32, ImageLayout)>,
    depth_stencil_attachment: Option<(Id, ImageLayout)>,
    accesses: LinearMap<Id, ResourceAccess>,
}

impl IntermediateRepresentationBuilder {
    fn new(node_capacity: u32, resource_capacity: u32) -> Self {
        IntermediateRepresentationBuilder {
            submissions: Vec::new(),
            nodes: vec![NodeState::default(); node_capacity as usize],
            prev_accesses: vec![ResourceAccess::default(); resource_capacity as usize],
            prev_node_indices: vec![0; resource_capacity as usize],
            semaphore_count: 0,
            render_passes: Vec::new(),
            is_render_pass_instance_active: false,
            pre_present_queue_family_ownership_transfers: Vec::new(),
        }
    }

    fn subpass(
        &mut self,
        node_index: NodeIndex,
        accesses: &ResourceAccesses,
        attachments: &Attachments,
    ) {
        let is_same_render_pass = self.is_render_pass_instance_active
            && is_render_pass_mergeable(self.render_passes.last().unwrap(), accesses, attachments);

        if !is_same_render_pass {
            self.render_passes
                .push(RenderPassState::new(attachments.framebuffer_id, node_index));
        }

        let render_pass_state = self.render_passes.last_mut().unwrap();

        let is_same_subpass = render_pass_state
            .subpasses
            .last()
            .is_some_and(|subpass_state| {
                is_subpass_mergeable(subpass_state, accesses, attachments)
            });

        if !is_same_subpass {
            render_pass_state
                .subpasses
                .push(SubpassState::from_task_node(accesses, attachments));
        }

        let subpass_state = render_pass_state.subpasses.last_mut().unwrap();

        for (id, access) in accesses.iter() {
            let current_access = subpass_state
                .accesses
                .get_or_insert_with(id, Default::default);
            current_access.stage_mask |= access.stage_mask;
            current_access.access_mask |= access.access_mask;
        }

        for (&id, attachment_info) in attachments.iter() {
            let access = *accesses.get(id.erase()).unwrap();
            let attachment_state =
                render_pass_state
                    .attachments
                    .get_or_insert_with(id, || AttachmentState {
                        first_node_index: node_index,
                        first_access: access,
                        last_access: ResourceAccess::default(),
                        has_reads: false,
                        index: attachment_info.index,
                        clear: attachment_info.clear,
                        format: attachment_info.format,
                        component_mapping: attachment_info.component_mapping,
                        mip_level: attachment_info.mip_level,
                        base_array_layer: attachment_info.base_array_layer,
                    });
            attachment_state.last_access = access;
            attachment_state.has_reads |= access.access_mask.contains_reads();

            if attachment_info.clear {
                // If we are the first node in the render pass to clear this attachment, we can
                // make use of `AttachmentLoadOp::Clear`. Otherwise, we have to use the
                // `clear_attachments` command.
                if attachment_state.first_node_index == node_index {
                    if !render_pass_state.clear_node_indices.contains(&node_index) {
                        render_pass_state.clear_node_indices.push(node_index);
                    }
                } else {
                    let node_state = &mut self.nodes[node_index as usize];

                    if !node_state.clear_attachments.contains(&id) {
                        node_state.clear_attachments.push(id);
                    }
                }
            }
        }

        render_pass_state.last_node_index = node_index;

        self.nodes[node_index as usize].subpass = Some(Subpass {
            subpass_index: render_pass_state.subpasses.len() - 1,
            render_pass_index: self.render_passes.len() - 1,
        });
    }

    fn resource_access(&mut self, node_index: NodeIndex, id: Id, access: ResourceAccess) {
        let prev_access = self.prev_accesses[id.index() as usize];
        let prev_node_index = self.prev_node_indices[id.index() as usize];
        let mut barriered = true;

        if prev_access.stage_mask.is_empty() {
            if id.is::<Swapchain>() {
                self.swapchain_acquire(node_index, unsafe { id.parametrize() }, access);
            } else if id.is::<Image>() {
                self.initial_image_layout_transition(id, access);
            } else if access.access_mask.contains_reads() {
                self.initial_memory_barrier(id, access);
            }
        } else if prev_access.queue_family_index != access.queue_family_index {
            if id.is_exclusive() {
                self.queue_family_ownership_release(prev_node_index, id, access);
                self.queue_family_ownership_acquire(node_index, id, access);
            } else {
                let prev_access = &mut self.prev_accesses[id.index() as usize];
                prev_access.stage_mask = PipelineStages::empty();
                prev_access.access_mask = AccessFlags::empty();

                if prev_access.image_layout != access.image_layout {
                    self.image_layout_transition(node_index, id, access);
                }
            }
        } else if self.is_render_pass_instance_active
            && self.nodes[prev_node_index as usize]
                .subpass
                .is_some_and(|subpass| subpass.render_pass_index == self.render_passes.len() - 1)
        {
            // Dependencies within a render pass instance are handled using subpass dependencies.
        } else if prev_access.image_layout != access.image_layout {
            self.image_layout_transition(node_index, id, access);
        } else if prev_access.access_mask.contains_writes() {
            self.memory_barrier(node_index, id, access);
        } else if access.access_mask.contains_writes() {
            self.execution_barrier(node_index, id, access);
        } else {
            barriered = false;
        }

        let prev_access = &mut self.prev_accesses[id.index() as usize];
        let prev_node_index = &mut self.prev_node_indices[id.index() as usize];

        if barriered {
            *prev_access = access;
            *prev_node_index = node_index;
        } else {
            prev_access.stage_mask |= access.stage_mask;
            prev_access.access_mask |= access.access_mask;

            let prev_barrier = self.nodes[*prev_node_index as usize]
                .start_barriers
                .iter_mut()
                .find(|barrier| barrier.resource == id)
                .unwrap_or_else(|| {
                    self.submissions
                        .last_mut()
                        .unwrap()
                        .initial_barriers
                        .iter_mut()
                        .find(|barrier| barrier.resource == id)
                        .unwrap()
                });
            prev_barrier.dst_stage_mask |= access.stage_mask;
            prev_barrier.dst_access_mask |= access.access_mask;
        }
    }

    fn swapchain_acquire(
        &mut self,
        node_index: NodeIndex,
        swapchain_id: Id<Swapchain>,
        access: ResourceAccess,
    ) {
        let src = ResourceAccess {
            stage_mask: access.stage_mask,
            access_mask: AccessFlags::empty(),
            image_layout: ImageLayout::Undefined,
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        };
        let dst = ResourceAccess {
            stage_mask: access.stage_mask,
            access_mask: access.access_mask,
            image_layout: access.image_layout,
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        };

        self.memory_barrier_inner(node_index, swapchain_id.erase(), src, dst, false);

        let submission_state = self.submissions.last().unwrap();

        self.nodes[submission_state.first_node_index as usize]
            .wait_acquire
            .push((swapchain_id, access.stage_mask));
    }

    fn initial_image_layout_transition(&mut self, id: Id, access: ResourceAccess) {
        self.initial_memory_barrier(id, access);
    }

    fn initial_memory_barrier(&mut self, id: Id, access: ResourceAccess) {
        let submission_state = self.submissions.last_mut().unwrap();

        submission_state
            .initial_barriers
            .push(super::MemoryBarrier {
                src_stage_mask: PipelineStages::empty(),
                src_access_mask: AccessFlags::empty(),
                dst_stage_mask: access.stage_mask,
                dst_access_mask: access.access_mask,
                old_layout: ImageLayout::Undefined,
                new_layout: access.image_layout,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                resource: id,
            });
    }

    fn queue_family_ownership_release(
        &mut self,
        node_index: NodeIndex,
        id: Id,
        access: ResourceAccess,
    ) {
        debug_assert!(id.is_exclusive());

        let prev_access = &mut self.prev_accesses[id.index() as usize];
        let mut src = *prev_access;
        let dst = ResourceAccess {
            stage_mask: PipelineStages::empty(),
            access_mask: AccessFlags::empty(),
            ..access
        };

        if !prev_access.access_mask.contains_writes() {
            src.access_mask = AccessFlags::empty();
        }

        debug_assert_ne!(src.queue_family_index, dst.queue_family_index);

        self.memory_barrier_inner(node_index, id, src, dst, true);
    }

    fn queue_family_ownership_acquire(
        &mut self,
        node_index: NodeIndex,
        id: Id,
        access: ResourceAccess,
    ) {
        debug_assert!(id.is_exclusive());

        let prev_access = self.prev_accesses[id.index() as usize];
        let src = ResourceAccess {
            stage_mask: PipelineStages::empty(),
            access_mask: AccessFlags::empty(),
            ..prev_access
        };
        let dst = access;

        debug_assert_ne!(src.queue_family_index, dst.queue_family_index);

        self.memory_barrier_inner(node_index, id, src, dst, false);
    }

    fn image_layout_transition(&mut self, node_index: NodeIndex, id: Id, access: ResourceAccess) {
        debug_assert_ne!(
            self.prev_accesses[id.index() as usize].image_layout,
            access.image_layout,
        );

        self.memory_barrier(node_index, id, access);
    }

    fn memory_barrier(&mut self, node_index: NodeIndex, id: Id, access: ResourceAccess) {
        let prev_access = self.prev_accesses[id.index() as usize];
        let src = ResourceAccess {
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..prev_access
        };
        let dst = ResourceAccess {
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..access
        };

        self.memory_barrier_inner(node_index, id, src, dst, false);
    }

    fn execution_barrier(&mut self, node_index: NodeIndex, id: Id, access: ResourceAccess) {
        let prev_access = self.prev_accesses[id.index() as usize];
        let src = ResourceAccess {
            access_mask: AccessFlags::empty(),
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..prev_access
        };
        let dst = ResourceAccess {
            access_mask: AccessFlags::empty(),
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..access
        };

        debug_assert_eq!(prev_access.image_layout, access.image_layout);

        self.memory_barrier_inner(node_index, id, src, dst, false);
    }

    fn memory_barrier_inner(
        &mut self,
        node_index: NodeIndex,
        id: Id,
        src: ResourceAccess,
        dst: ResourceAccess,
        is_end_barrier: bool,
    ) {
        let mut node_state = &mut self.nodes[node_index as usize];

        // Regular pipeline barriers are not permitted during a render pass instance, so we need to
        // move them before/after the render pass instance. This works because dependencies between
        // subpasses are handled with subpass dependencies and the only pipeline barriers that
        // could be needed are image layout transitions and/or queue family ownership transfers on
        // first/last use of a resource used during the render pass instance.
        if let Some(subpass) = node_state.subpass {
            let render_pass_state = &self.render_passes[subpass.render_pass_index];
            let moved_node_index = if is_end_barrier {
                render_pass_state.last_node_index
            } else {
                render_pass_state.first_node_index
            };
            node_state = &mut self.nodes[moved_node_index as usize];
        }

        let barriers = if is_end_barrier {
            &mut node_state.end_barriers
        } else {
            &mut node_state.start_barriers
        };

        barriers.push(super::MemoryBarrier {
            src_stage_mask: src.stage_mask,
            src_access_mask: src.access_mask,
            dst_stage_mask: dst.stage_mask,
            dst_access_mask: dst.access_mask,
            old_layout: src.image_layout,
            new_layout: dst.image_layout,
            src_queue_family_index: src.queue_family_index,
            dst_queue_family_index: dst.queue_family_index,
            resource: id,
        });
    }

    fn swapchain_present(
        &mut self,
        node_index: NodeIndex,
        swapchain_id: Id<Swapchain>,
        present_queue: &Queue,
    ) {
        let prev_access = self.prev_accesses[swapchain_id.index() as usize];
        let src = ResourceAccess {
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..prev_access
        };

        let needs_queue_family_ownership_transfer = prev_access.queue_family_index
            != present_queue.queue_family_index()
            && swapchain_id.is_exclusive();

        if needs_queue_family_ownership_transfer {
            self.queue_family_ownership_release(
                node_index,
                swapchain_id.erase(),
                ResourceAccess {
                    stage_mask: PipelineStages::empty(),
                    access_mask: AccessFlags::empty(),
                    image_layout: ImageLayout::PresentSrc,
                    queue_family_index: present_queue.queue_family_index(),
                },
            );
        } else {
            self.memory_barrier_inner(
                node_index,
                swapchain_id.erase(),
                src,
                ResourceAccess {
                    stage_mask: PipelineStages::empty(),
                    access_mask: AccessFlags::empty(),
                    image_layout: ImageLayout::PresentSrc,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
                true,
            );
        }

        let node_state = &self.nodes[node_index as usize];
        let submission_state = &self.submissions[node_state.submission_index];

        if needs_queue_family_ownership_transfer {
            self.nodes[submission_state.last_node_index as usize]
                .signal_pre_present
                .push((swapchain_id, prev_access.stage_mask));
            self.pre_present_queue_family_ownership_transfers
                .push(swapchain_id);
        } else {
            self.nodes[submission_state.last_node_index as usize]
                .signal_present
                .push((swapchain_id, prev_access.stage_mask));
        }
    }
}

impl SubmissionState {
    fn new(node_index: u32) -> Self {
        Self {
            initial_barriers: Vec::new(),
            first_node_index: node_index,
            last_node_index: node_index,
        }
    }
}

impl RenderPassState {
    fn new(framebuffer: Id<Framebuffer>, node_index: NodeIndex) -> Self {
        RenderPassState {
            framebuffer_id: framebuffer,
            attachments: LinearMap::new(),
            subpasses: Vec::new(),
            first_node_index: node_index,
            last_node_index: node_index,
            clear_node_indices: Vec::new(),
        }
    }
}

impl SubpassState {
    fn from_task_node(accesses: &ResourceAccesses, attachments: &Attachments) -> Self {
        SubpassState {
            input_attachments: attachments
                .input_attachments
                .iter()
                .map(|(id, attachment_info)| {
                    let input_attachment_index = attachment_info.index;
                    let image_layout = accesses.get(id.erase()).unwrap().image_layout;

                    (*id, (input_attachment_index, image_layout))
                })
                .collect(),
            color_attachments: attachments
                .color_attachments
                .iter()
                .map(|(id, attachment_info)| {
                    let location = attachment_info.index;
                    let image_layout = accesses.get(id.erase()).unwrap().image_layout;

                    (*id, (location, image_layout))
                })
                .collect(),
            depth_stencil_attachment: attachments.depth_stencil_attachment.as_ref().map(
                |(id, _)| {
                    let image_layout = accesses.get(id.erase()).unwrap().image_layout;

                    (*id, image_layout)
                },
            ),
            accesses: accesses.inner.clone(),
        }
    }
}

fn is_render_pass_mergeable(
    render_pass_state: &RenderPassState,
    accesses: &ResourceAccesses,
    attachments: &Attachments,
) -> bool {
    const ATTACHMENT_ACCESSES: AccessFlags = AccessFlags::INPUT_ATTACHMENT_READ
        .union(AccessFlags::COLOR_ATTACHMENT_READ)
        .union(AccessFlags::COLOR_ATTACHMENT_WRITE)
        .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
        .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

    // Different framebuffers may have different dimensions so we can't merge them.
    if render_pass_state.framebuffer_id != attachments.framebuffer_id {
        return false;
    }

    // We can't merge with a task node that accesses one of our attachments as anything other
    // than an attachment, as render passes don't allow this.
    if render_pass_state.attachments.keys().any(|id| {
        accesses
            .get(id.erase())
            .is_some_and(|access| !ATTACHMENT_ACCESSES.contains(access.access_mask))
    }) {
        return false;
    }

    // We can't merge with a task node that has an attachment which we access as anything other
    // than an attachment, as render passes don't allow this.
    if attachments.keys().any(|id| {
        render_pass_state.subpasses.iter().any(|subpass_state| {
            subpass_state
                .accesses
                .get(&id.erase())
                .is_some_and(|access| !ATTACHMENT_ACCESSES.contains(access.access_mask))
        })
    }) {
        return false;
    }

    true
}

fn is_subpass_mergeable(
    subpass_state: &SubpassState,
    accesses: &ResourceAccesses,
    attachments: &Attachments,
) -> bool {
    // We can only merge with a task node that has the exact same attachments. Keep in mind that
    // being here, `is_render_pass_mergeable` must have returned `true`, which rules out the other
    // incompatibilities.

    // Early nope-out to avoid the expense below.
    if subpass_state.color_attachments.len() != attachments.color_attachments.len()
        || subpass_state.input_attachments.len() != attachments.input_attachments.len()
        || subpass_state.depth_stencil_attachment.is_some()
            != attachments.depth_stencil_attachment.is_some()
    {
        return false;
    }

    for (id, &(location, image_layout)) in subpass_state.color_attachments.iter() {
        if !attachments
            .color_attachments
            .get(id)
            .is_some_and(|attachment_info| attachment_info.index == location)
            || accesses.get(id.erase()).unwrap().image_layout != image_layout
        {
            return false;
        }
    }

    for (id, &(input_attachment_index, image_layout)) in subpass_state.input_attachments.iter() {
        if !attachments
            .input_attachments
            .get(id)
            .is_some_and(|attachment_info| attachment_info.index == input_attachment_index)
            || accesses.get(id.erase()).unwrap().image_layout != image_layout
        {
            return false;
        }
    }

    if subpass_state.depth_stencil_attachment
        != attachments
            .depth_stencil_attachment
            .as_ref()
            .map(|(id, _)| (*id, accesses.get(id.erase()).unwrap().image_layout))
    {
        return false;
    }

    true
}

fn create_render_pass(
    resources: &super::Resources,
    render_pass_state: RenderPassState,
) -> Result<super::RenderPassState, CompileErrorKind> {
    const FRAMEBUFFER_SPACE_ACCESS_FLAGS: AccessFlags = AccessFlags::INPUT_ATTACHMENT_READ
        .union(AccessFlags::COLOR_ATTACHMENT_READ)
        .union(AccessFlags::COLOR_ATTACHMENT_WRITE)
        .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
        .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

    let attachments = render_pass_state
        .attachments
        .iter()
        .map(|(&id, attachment_state)| {
            let resource_info = resources.get(id.erase()).unwrap();
            let is_resident = !resource_info
                .usage
                .contains(ImageUsage::TRANSIENT_ATTACHMENT);

            AttachmentDescription {
                format: attachment_state.format,
                samples: resource_info.samples,
                load_op: if attachment_state.clear {
                    AttachmentLoadOp::Clear
                } else if attachment_state.has_reads && is_resident {
                    AttachmentLoadOp::Load
                } else {
                    AttachmentLoadOp::DontCare
                },
                store_op: if is_resident {
                    AttachmentStoreOp::Store
                } else {
                    AttachmentStoreOp::DontCare
                },
                initial_layout: attachment_state.first_access.image_layout,
                final_layout: attachment_state.last_access.image_layout,
                ..Default::default()
            }
        })
        .collect::<Vec<_>>();

    let subpasses = render_pass_state
        .subpasses
        .iter()
        .map(|subpass_state| {
            (
                convert_attachments(&render_pass_state, &subpass_state.input_attachments),
                convert_attachments(&render_pass_state, &subpass_state.color_attachments),
                subpass_state
                    .depth_stencil_attachment
                    .map(|(id, layout)| AttachmentReference {
                        attachment: render_pass_state.attachments.index_of(&id).unwrap() as u32,
                        layout,
                        ..Default::default()
                    }),
                render_pass_state
                    .attachments
                    .keys()
                    .enumerate()
                    .filter_map(|(attachment, &id)| {
                        (!subpass_state.input_attachments.contains_key(&id)
                            && !subpass_state.color_attachments.contains_key(&id)
                            && !subpass_state
                                .depth_stencil_attachment
                                .is_some_and(|(x, _)| x == id))
                        .then_some(attachment as u32)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let subpasses = subpasses
        .iter()
        .map(
            |(
                input_attachments,
                color_attachments,
                depth_stencil_attachment,
                preserve_attachments,
            )| {
                SubpassDescription {
                    // FIXME:
                    view_mask: 0,
                    input_attachments,
                    color_attachments,
                    depth_stencil_attachment: Some(depth_stencil_attachment),
                    preserve_attachments,
                    ..Default::default()
                }
            },
        )
        .collect::<Vec<_>>();

    let mut dependencies = Vec::<SubpassDependency<'static>>::new();

    // FIXME: subpass dependency chains
    for (src_subpass_index, src_subpass_state) in render_pass_state.subpasses.iter().enumerate() {
        let src_subpass = src_subpass_index as u32;

        for (dst_subpass_index, dst_subpass_state) in render_pass_state
            .subpasses
            .iter()
            .enumerate()
            .skip(src_subpass_index + 1)
        {
            let dst_subpass = dst_subpass_index as u32;

            for (id, src_access) in src_subpass_state.accesses.iter() {
                let Some(dst_access) = dst_subpass_state.accesses.get(id) else {
                    continue;
                };
                let (src_access_mask, dst_access_mask) = if src_access.access_mask.contains_writes()
                {
                    (src_access.access_mask, dst_access.access_mask)
                } else if dst_access.access_mask.contains_writes() {
                    (AccessFlags::empty(), AccessFlags::empty())
                } else {
                    continue;
                };

                let dependency =
                    get_or_insert_subpass_dependency(&mut dependencies, src_subpass, dst_subpass);
                dependency.src_stages |= src_access.stage_mask;
                dependency.src_access |= src_access_mask;
                dependency.dst_stages |= dst_access.stage_mask;
                dependency.dst_access |= dst_access_mask;

                // Attachments can only have framebuffer-space accesses, for which
                // `DependencyFlags::BY_REGION` is always sound. For resources other than
                // attachments, it could be sound, but we have no way of knowing. There should be
                // no reason for this in the first place, or at least be really uncommon.
                if !FRAMEBUFFER_SPACE_ACCESS_FLAGS.contains(src_access_mask) {
                    dependency.dependency_flags = DependencyFlags::empty();
                }
            }
        }
    }

    // TODO: What do we do about non-framebuffer-space subpass self-dependencies?
    for (subpass_index, subpass_state) in render_pass_state.subpasses.iter().enumerate() {
        let subpass = subpass_index as u32;

        for access in subpass_state.accesses.values() {
            let access_mask = access.access_mask;

            if access_mask.contains(AccessFlags::INPUT_ATTACHMENT_READ) {
                if access_mask.contains(AccessFlags::COLOR_ATTACHMENT_WRITE) {
                    let dependency =
                        get_or_insert_subpass_dependency(&mut dependencies, subpass, subpass);
                    dependency.src_stages |= PipelineStages::COLOR_ATTACHMENT_OUTPUT;
                    dependency.dst_stages |= PipelineStages::FRAGMENT_SHADER;
                    dependency.src_access |= AccessFlags::COLOR_ATTACHMENT_WRITE;
                    dependency.dst_access |= AccessFlags::INPUT_ATTACHMENT_READ;
                } else if access_mask.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
                    let dependency =
                        get_or_insert_subpass_dependency(&mut dependencies, subpass, subpass);
                    dependency.src_stages |=
                        PipelineStages::EARLY_FRAGMENT_TESTS | PipelineStages::LATE_FRAGMENT_TESTS;
                    dependency.dst_stages |= PipelineStages::FRAGMENT_SHADER;
                    dependency.src_access |= AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    dependency.dst_access |= AccessFlags::INPUT_ATTACHMENT_READ;
                }
            }
        }
    }

    let render_pass = RenderPass::try_new(
        resources.physical_resources.device(),
        &RenderPassCreateInfo {
            attachments: &attachments,
            subpasses: &subpasses,
            dependencies: &dependencies,
            ..Default::default()
        },
    )
    // FIXME:
    .map_err(Validated::unwrap)
    .map_err(CompileErrorKind::VulkanError)?;

    let attachments = render_pass_state
        .attachments
        .iter()
        .map(|(&id, attachment_state)| {
            let attachment = super::AttachmentState {
                index: attachment_state.index,
                format: attachment_state.format,
                component_mapping: attachment_state.component_mapping,
                mip_level: attachment_state.mip_level,
                base_array_layer: attachment_state.base_array_layer,
            };

            (id, attachment)
        })
        .collect();

    Ok(super::RenderPassState {
        render_pass,
        attachments,
        framebuffers: Vec::new(),
        clear_node_indices: render_pass_state.clear_node_indices,
    })
}

fn convert_attachments(
    render_pass_state: &RenderPassState,
    attachments: &LinearMap<Id, (u32, ImageLayout)>,
) -> Vec<Option<AttachmentReference<'static>>> {
    let len = attachments.values().map(|(i, _)| *i + 1).max().unwrap_or(0);
    let mut attachments_out = vec![None; len as usize];

    for (&id, &(index, layout)) in attachments.iter() {
        let attachment = &mut attachments_out[index as usize];

        debug_assert!(attachment.is_none());

        *attachment = Some(AttachmentReference {
            attachment: render_pass_state.attachments.index_of(&id).unwrap() as u32,
            layout,
            ..Default::default()
        });
    }

    attachments_out
}

fn get_or_insert_subpass_dependency<'a>(
    dependencies: &'a mut Vec<SubpassDependency<'static>>,
    src_subpass: u32,
    dst_subpass: u32,
) -> &'a mut SubpassDependency<'static> {
    let dependency_index = dependencies
        .iter_mut()
        .position(|dependency| {
            dependency.src_subpass == Some(src_subpass)
                && dependency.dst_subpass == Some(dst_subpass)
        })
        .unwrap_or_else(|| {
            let index = dependencies.len();

            dependencies.push(SubpassDependency {
                src_subpass: Some(src_subpass),
                dst_subpass: Some(dst_subpass),
                dependency_flags: DependencyFlags::BY_REGION,
                ..Default::default()
            });

            index
        });

    &mut dependencies[dependency_index]
}

struct FinalRepresentationBuilder {
    instructions: Vec<Instruction>,
    submissions: Vec<Submission>,
    barriers: Vec<super::MemoryBarrier>,
    clear_attachments: Vec<Id>,
    present_queue: Option<Arc<Queue>>,
    initial_barrier_range: Range<BarrierIndex>,
    has_flushed_submit: bool,
    should_flush_submit: bool,
    prev_barrier_index: usize,
    prev_subpass_index: usize,
}

impl FinalRepresentationBuilder {
    fn new(present_queue: Option<&Arc<Queue>>) -> Self {
        FinalRepresentationBuilder {
            instructions: Vec::new(),
            submissions: Vec::new(),
            barriers: Vec::new(),
            clear_attachments: Vec::new(),
            present_queue: present_queue.cloned(),
            initial_barrier_range: 0..0,
            has_flushed_submit: true,
            should_flush_submit: false,
            prev_barrier_index: 0,
            prev_subpass_index: 0,
        }
    }

    fn initial_pipeline_barrier(&mut self, barriers: &[super::MemoryBarrier]) {
        self.barriers.extend_from_slice(barriers);
        self.initial_barrier_range =
            self.prev_barrier_index as BarrierIndex..self.barriers.len() as BarrierIndex;
        self.prev_barrier_index = self.barriers.len();
    }

    fn pipeline_barrier(&mut self, barriers: &[super::MemoryBarrier]) {
        self.barriers.extend_from_slice(barriers);
    }

    fn wait_acquire(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        if !self.has_flushed_submit {
            self.flush_submit();
        }

        self.instructions.push(Instruction::WaitAcquire {
            swapchain_id,
            stage_mask,
        });
    }

    fn wait_semaphore(&mut self, semaphore_index: SemaphoreIndex) {
        if !self.has_flushed_submit {
            self.flush_submit();
        }

        self.instructions.push(Instruction::WaitSemaphore {
            semaphore_index,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });
    }

    fn begin_render_pass(&mut self, render_pass_index: RenderPassIndex) {
        self.flush_barriers();

        self.instructions
            .push(Instruction::BeginRenderPass { render_pass_index });

        self.prev_subpass_index = 0;
    }

    fn next_subpass(&mut self, subpass_index: usize) {
        if self.prev_subpass_index != subpass_index {
            self.instructions.push(Instruction::NextSubpass);
            self.prev_subpass_index = subpass_index;
        }
    }

    fn execute_task(&mut self, node_index: NodeIndex) {
        self.flush_barriers();

        self.instructions
            .push(Instruction::ExecuteTask { node_index });

        self.has_flushed_submit = false;
    }

    fn end_render_pass(&mut self) {
        self.instructions.push(Instruction::EndRenderPass);
    }

    fn clear_attachments(
        &mut self,
        node_index: NodeIndex,
        render_pass_index: RenderPassIndex,
        clear_attachments: &[Id],
    ) {
        self.flush_barriers();

        let clear_attachment_range_start = self.clear_attachments.len();
        self.clear_attachments.extend_from_slice(clear_attachments);
        self.instructions.push(Instruction::ClearAttachments {
            node_index,
            render_pass_index,
            clear_attachment_range: clear_attachment_range_start..self.clear_attachments.len(),
        });
    }

    fn signal_semaphore(&mut self, semaphore_index: SemaphoreIndex) {
        self.flush_barriers();

        self.instructions.push(Instruction::SignalSemaphore {
            semaphore_index,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });

        self.should_flush_submit = true;
    }

    fn signal_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        self.flush_barriers();

        self.instructions.push(Instruction::SignalPresent {
            swapchain_id,
            stage_mask,
        });

        self.should_flush_submit = true;
    }

    fn signal_pre_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        self.flush_barriers();

        self.instructions.push(Instruction::SignalPrePresent {
            swapchain_id,
            stage_mask,
        });

        self.should_flush_submit = true;
    }

    fn pre_present_acquire_queue_family_ownership(
        &mut self,
        last_accesses: &[ResourceAccess],
        swapchain_id: Id<Swapchain>,
    ) {
        if !self.has_flushed_submit {
            self.flush_submit();
        }

        self.instructions.push(Instruction::WaitPrePresent {
            swapchain_id,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });

        let last_access = last_accesses[swapchain_id.index() as usize];

        self.barriers.push(super::MemoryBarrier {
            src_stage_mask: PipelineStages::empty(),
            src_access_mask: AccessFlags::empty(),
            dst_stage_mask: PipelineStages::empty(),
            dst_access_mask: AccessFlags::empty(),
            old_layout: last_access.image_layout,
            new_layout: ImageLayout::PresentSrc,
            src_queue_family_index: last_access.queue_family_index,
            dst_queue_family_index: self.present_queue.as_ref().unwrap().queue_family_index(),
            resource: swapchain_id.erase(),
        });

        self.instructions.push(Instruction::SignalPresent {
            swapchain_id,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });
    }

    fn flush_barriers(&mut self) {
        if self.prev_barrier_index != self.barriers.len() {
            self.instructions.push(Instruction::PipelineBarrier {
                barrier_range: self.prev_barrier_index as BarrierIndex
                    ..self.barriers.len() as BarrierIndex,
            });
            self.prev_barrier_index = self.barriers.len();
        }
    }

    fn flush_submit(&mut self) {
        self.flush_barriers();
        self.instructions.push(Instruction::FlushSubmit);
        self.has_flushed_submit = true;
        self.should_flush_submit = false;
    }

    fn submit(&mut self, queue: Arc<Queue>) {
        self.instructions.push(Instruction::Submit);

        let prev_instruction_range_end = self
            .submissions
            .last()
            .map(|s| s.instruction_range.end)
            .unwrap_or(0);
        self.submissions.push(Submission {
            queue,
            initial_barrier_range: self.initial_barrier_range.clone(),
            instruction_range: prev_instruction_range_end..self.instructions.len(),
        });
    }
}

impl<W: ?Sized> ExecutableTaskGraph<W> {
    /// Decompiles the graph back into a modifiable form.
    #[inline]
    pub fn decompile(mut self) -> TaskGraph<W> {
        self.drop_graph = false;

        // SAFETY: We unset the `drop_graph` flag which ensures that the graph isn't dropped by the
        // `Drop` implementation.
        unsafe { ManuallyDrop::take(&mut self.graph) }
    }
}

/// Parameters to [compile] a [`TaskGraph`].
///
/// [compile]: TaskGraph::compile
#[derive(Clone, Debug)]
pub struct CompileInfo<'a> {
    /// The queues to work with.
    ///
    /// You must supply at least one queue and all queues must be from unique queue families.
    ///
    /// The default value is empty, which must be overridden.
    pub queues: &'a [&'a Arc<Queue>],

    /// The queue to use for swapchain presentation, if any.
    ///
    /// You must supply this queue if the task graph uses any swapchains. It can be the same queue
    /// as one in the [`queues`] field, or a different one.
    ///
    /// The default value is `None`.
    ///
    /// [`queues`]: Self::queues
    pub present_queue: Option<&'a Arc<Queue>>,

    /// The flight which will be executed.
    ///
    /// The default value is `Id::INVALID`, which must be overridden.
    pub flight_id: Id<Flight>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CompileInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CompileInfo<'_> {
    /// Returns a default `CompileInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            queues: &[],
            present_queue: None,
            flight_id: Id::INVALID,
            _ne: crate::NE,
        }
    }
}

/// Error that can happen when [compiling] a [`TaskGraph`].
///
/// [compiling]: TaskGraph::compile
pub struct CompileError<W: ?Sized> {
    pub graph: TaskGraph<W>,
    pub kind: CompileErrorKind,
}

/// The kind of [`CompileError`] that occurred.
#[derive(Debug, PartialEq, Eq)]
pub enum CompileErrorKind {
    Unconnected,
    Cycle,
    InsufficientQueues,
    VulkanError(VulkanError),
}

impl<W: ?Sized> CompileError<W> {
    fn new(graph: TaskGraph<W>, kind: CompileErrorKind) -> Self {
        CompileError { graph, kind }
    }
}

impl<W: ?Sized> fmt::Debug for CompileError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.kind, f)
    }
}

impl<W: ?Sized> fmt::Display for CompileError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            CompileErrorKind::Unconnected => f.write_str("the graph is not weakly connected"),
            CompileErrorKind::Cycle => f.write_str("the graph contains a directed cycle"),
            CompileErrorKind::InsufficientQueues => {
                f.write_str("the given queues are not sufficient for the requirements of a task")
            }
            CompileErrorKind::VulkanError(_) => f.write_str("a runtime error occurred"),
        }
    }
}

impl<W: ?Sized> Error for CompileError<W> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            CompileErrorKind::VulkanError(err) => Some(err),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::AttachmentInfo,
        resource::{AccessTypes, ImageLayoutType},
        tests::test_queues,
    };
    use std::marker::PhantomData;
    use vulkano::{
        buffer::BufferCreateInfo, image::ImageCreateInfo, swapchain::SwapchainCreateInfo,
        sync::Sharing,
    };

    #[test]
    fn unconnected1() {
        let (resources, queues) = test_queues!();

        // ┌───┐
        // │ A │
        // └───┘
        // ┄┄┄┄┄
        // ┌───┐
        // │ B │
        // └───┘
        let mut graph = TaskGraph::<()>::new(&resources);
        graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        graph
            .create_task_node("B", QueueFamilyType::Compute, PhantomData)
            .build();

        let err = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap_err();

        assert_eq!(err.kind, CompileErrorKind::Unconnected);
    }

    #[test]
    fn unconnected2() {
        let (resources, queues) = test_queues!();

        // ┌───┐
        // │ A ├─┐
        // └───┘ │
        // ┌───┐ │
        // │ B ├┐│
        // └───┘││
        // ┄┄┄┄┄││┄┄┄┄┄┄
        //      ││ ┌───┐
        //      │└►│ C │
        //      │  └───┘
        //      │  ┌───┐
        //      └─►│ D │
        //         └───┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Compute, PhantomData)
            .build();
        let d = graph
            .create_task_node("D", QueueFamilyType::Compute, PhantomData)
            .build();
        graph.add_edge(a, c).unwrap();
        graph.add_edge(b, d).unwrap();

        let err = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap_err();

        assert_eq!(err.kind, CompileErrorKind::Unconnected);
    }

    #[test]
    fn unconnected3() {
        let (resources, queues) = test_queues!();

        // ┌───┐  ┌───┐  ┌───┐
        // │ A ├─►│ B ├─►│ C │
        // └───┘  └───┘  └───┘
        // ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
        // ┌───┐  ┌───┐  ┌───┐   ┌───┐
        // │ D ├┬►│ E ├┬►│ F ├──►│   │
        // └───┘│ └───┘│ └───┘┌─►│ G │
        //      │      └──────┘┌►│   │
        //      └──────────────┘ └───┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
            .build();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        let d = graph
            .create_task_node("D", QueueFamilyType::Compute, PhantomData)
            .build();
        let e = graph
            .create_task_node("E", QueueFamilyType::Compute, PhantomData)
            .build();
        let f = graph
            .create_task_node("F", QueueFamilyType::Compute, PhantomData)
            .build();
        let g = graph
            .create_task_node("G", QueueFamilyType::Compute, PhantomData)
            .build();
        graph.add_edge(d, e).unwrap();
        graph.add_edge(d, g).unwrap();
        graph.add_edge(e, f).unwrap();
        graph.add_edge(e, g).unwrap();
        graph.add_edge(f, g).unwrap();

        let err = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap_err();

        assert_eq!(err.kind, CompileErrorKind::Unconnected);
    }

    #[test]
    fn cycle1() {
        let (resources, queues) = test_queues!();

        //   ┌───┐  ┌───┐  ┌───┐
        // ┌►│ A ├─►│ B ├─►│ C ├┐
        // │ └───┘  └───┘  └───┘│
        // └────────────────────┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
            .build();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();

        let err = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap_err();

        assert_eq!(err.kind, CompileErrorKind::Cycle);
    }

    #[test]
    fn cycle2() {
        let (resources, queues) = test_queues!();

        //   ┌───┐  ┌───┐  ┌───┐
        // ┌►│ A ├┬►│ B ├─►│ C ├┐
        // │ └───┘│ └───┘  └───┘│
        // │┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄
        // │      │ ┌───┐  ┌───┐│ ┌───┐
        // │      └►│ D ├─►│ E ├┴►│ F ├┐
        // │        └───┘  └───┘  └───┘│
        // └───────────────────────────┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
            .build();
        let d = graph
            .create_task_node("D", QueueFamilyType::Compute, PhantomData)
            .build();
        let e = graph
            .create_task_node("E", QueueFamilyType::Compute, PhantomData)
            .build();
        let f = graph
            .create_task_node("F", QueueFamilyType::Compute, PhantomData)
            .build();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(a, d).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, f).unwrap();
        graph.add_edge(d, e).unwrap();
        graph.add_edge(e, f).unwrap();
        graph.add_edge(f, a).unwrap();

        let err = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap_err();

        assert_eq!(err.kind, CompileErrorKind::Cycle);
    }

    #[test]
    fn cycle3() {
        let (resources, queues) = test_queues!();

        // ┌─────┐
        // │┌───┐└►┌───┐  ┌───┐
        // ││ A ├┬►│ B ├─►│ C ├┬──────┐
        // │└───┘│ └───┘┌►└───┘│      │
        // │┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄│┄
        // │     │ ┌───┐│ ┌───┐│ ┌───┐│
        // │     └►│ D ├┴►│ E │└►│ F ├│┐
        // │     ┌►└───┘  └───┘  └───┘││
        // │     └────────────────────┘│
        // └───────────────────────────┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
            .build();
        let d = graph
            .create_task_node("D", QueueFamilyType::Compute, PhantomData)
            .build();
        let e = graph
            .create_task_node("E", QueueFamilyType::Compute, PhantomData)
            .build();
        let f = graph
            .create_task_node("F", QueueFamilyType::Compute, PhantomData)
            .build();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(a, d).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(c, f).unwrap();
        graph.add_edge(d, c).unwrap();
        graph.add_edge(d, e).unwrap();
        graph.add_edge(f, b).unwrap();

        let err = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap_err();

        assert_eq!(err.kind, CompileErrorKind::Cycle);
    }

    #[test]
    fn initial_pipeline_barrier() {
        let (resources, queues) = test_queues!();

        let mut graph = TaskGraph::<()>::new(&resources);
        let buffer = graph.add_buffer(&BufferCreateInfo::default());
        let image = graph.add_image(&ImageCreateInfo::default());
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer, AccessTypes::VERTEX_SHADER_UNIFORM_READ)
            .image_access(
                image,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer, AccessTypes::VERTEX_SHADER_SAMPLED_READ)
            .image_access(
                image,
                AccessTypes::FRAGMENT_SHADER_STORAGE_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(node1, node2).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            InitialPipelineBarrier {
                barriers: [
                    {
                        dst_stage_mask: VERTEX_SHADER,
                        dst_access_mask: UNIFORM_READ | SHADER_SAMPLED_READ,
                        new_layout: Undefined,
                        resource: buffer,
                    },
                    {
                        dst_stage_mask: FRAGMENT_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ | SHADER_STORAGE_READ,
                        new_layout: General,
                        resource: image,
                    },
                ],
            },
            ExecuteTask { node: node1 },
            ExecuteTask { node: node2 },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn barrier1() {
        let (resources, queues) = test_queues!();

        let mut graph = TaskGraph::<()>::new(&resources);
        let buffer = graph.add_buffer(&BufferCreateInfo::default());
        let image = graph.add_image(&ImageCreateInfo::default());
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer, AccessTypes::FRAGMENT_SHADER_STORAGE_WRITE)
            .image_access(
                image,
                AccessTypes::FRAGMENT_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer, AccessTypes::INDIRECT_COMMAND_READ)
            .image_access(
                image,
                AccessTypes::FRAGMENT_SHADER_COLOR_INPUT_ATTACHMENT_READ,
                ImageLayoutType::General,
            )
            .build();
        let node3 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer, AccessTypes::RAY_TRACING_SHADER_STORAGE_READ)
            .image_access(
                image,
                AccessTypes::RAY_TRACING_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: node1 },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: FRAGMENT_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: DRAW_INDIRECT | RAY_TRACING_SHADER,
                        dst_access_mask: INDIRECT_COMMAND_READ | SHADER_STORAGE_READ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer,
                    },
                    {
                        src_stage_mask: FRAGMENT_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: FRAGMENT_SHADER | RAY_TRACING_SHADER,
                        dst_access_mask: INPUT_ATTACHMENT_READ | SHADER_SAMPLED_READ,
                        old_layout: General,
                        new_layout: General,
                        resource: image,
                    },
                ],
            },
            ExecuteTask { node: node2 },
            ExecuteTask { node: node3 },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn barrier2() {
        let (resources, queues) = test_queues!();

        let mut graph = TaskGraph::<()>::new(&resources);
        let buffer = graph.add_buffer(&BufferCreateInfo::default());
        let image = graph.add_image(&ImageCreateInfo::default());
        let node1 = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
            .image_access(
                image,
                AccessTypes::COPY_TRANSFER_WRITE,
                ImageLayoutType::General,
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer, AccessTypes::COPY_TRANSFER_WRITE)
            .image_access(
                image,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .build();
        let node3 = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer, AccessTypes::INDIRECT_COMMAND_READ)
            .image_access(
                image,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: node1 },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: COPY,
                        dst_access_mask: TRANSFER_WRITE,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer,
                    },
                    {
                        src_stage_mask: COPY,
                        src_access_mask: TRANSFER_WRITE,
                        dst_stage_mask: COMPUTE_SHADER,
                        dst_access_mask: SHADER_STORAGE_WRITE,
                        old_layout: General,
                        new_layout: General,
                        resource: image,
                    },
                ],
            },
            ExecuteTask { node: node2 },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COPY,
                        src_access_mask: TRANSFER_WRITE,
                        dst_stage_mask: DRAW_INDIRECT,
                        dst_access_mask: INDIRECT_COMMAND_READ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: COMPUTE_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: General,
                        new_layout: General,
                        resource: image,
                    },
                ],
            },
            ExecuteTask { node: node3 },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn semaphore1() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        // ┌───┐
        // │ A ├─┐
        // └───┘ │
        // ┌───┐ │
        // │ B ├┐│
        // └───┘││
        // ┄┄┄┄┄││┄┄┄┄┄┄
        //      │└►┌───┐
        //      └─►│ C │
        //         └───┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Compute, PhantomData)
            .build();
        graph.add_edge(a, c).unwrap();
        graph.add_edge(b, c).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: b },
            // TODO: This semaphore is redundant.
            SignalSemaphore {
                semaphore_index: semaphore1,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            ExecuteTask { node: a },
            SignalSemaphore {
                semaphore_index: semaphore2,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore1,
                stage_mask: ALL_COMMANDS,
            },
            WaitSemaphore {
                semaphore_index: semaphore2,
                stage_mask: ALL_COMMANDS,
            },
            ExecuteTask { node: c },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn semaphore2() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        // ┌───┐
        // │ A ├┐
        // └───┘│
        // ┄┄┄┄┄│┄┄┄┄┄┄
        //      │ ┌───┐
        //      ├►│ B │
        //      │ └───┘
        //      │ ┌───┐
        //      └►│ C │
        //        └───┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Compute, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Compute, PhantomData)
            .build();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(a, c).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: a },
            // TODO: This semaphore is redundant.
            SignalSemaphore {
                semaphore_index: semaphore1,
                stage_mask: ALL_COMMANDS,
            },
            SignalSemaphore {
                semaphore_index: semaphore2,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore2,
                stage_mask: ALL_COMMANDS,
            },
            ExecuteTask { node: c },
            FlushSubmit,
            WaitSemaphore {
                semaphore_index: semaphore1,
                stage_mask: ALL_COMMANDS,
            },
            ExecuteTask { node: b },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn semaphore3() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        // ┌───┐                ┌───┐
        // │ A ├───────┬───────►│ E │
        // └───┘       │      ┌►└───┘
        // ┌───┐       │      │
        // │ B ├┐      │      │
        // └───┘│      │      │
        // ┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄
        //      │ ┌───┐└►┌───┐│
        //      └►│ C ├─►│ D ├┘
        //        └───┘  └───┘
        let mut graph = TaskGraph::<()>::new(&resources);
        let a = graph
            .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
            .build();
        let b = graph
            .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
            .build();
        let c = graph
            .create_task_node("C", QueueFamilyType::Compute, PhantomData)
            .build();
        let d = graph
            .create_task_node("D", QueueFamilyType::Compute, PhantomData)
            .build();
        let e = graph
            .create_task_node("E", QueueFamilyType::Graphics, PhantomData)
            .build();
        graph.add_edge(a, d).unwrap();
        graph.add_edge(a, e).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, e).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        // TODO: This could be brought down to 3 submissions with task reordering.
        assert_matches_instructions!(
            graph,
            ExecuteTask { node: b },
            SignalSemaphore {
                semaphore_index: semaphore1,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore1,
                stage_mask: ALL_COMMANDS,
            },
            ExecuteTask { node: c },
            FlushSubmit,
            Submit,
            ExecuteTask { node: a },
            SignalSemaphore {
                semaphore_index: semaphore2,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore2,
                stage_mask: ALL_COMMANDS,
            },
            ExecuteTask { node: d },
            SignalSemaphore {
                semaphore_index: semaphore3,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore3,
                stage_mask: ALL_COMMANDS,
            },
            ExecuteTask { node: e },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn render_pass1() {
        let (resources, queues) = test_queues!();

        let mut graph = TaskGraph::<()>::new(&resources);
        let color_image = graph.add_image(&ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_READ | AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        graph.add_edge(node1, node2).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            BeginRenderPass,
            ExecuteTask { node: node1 },
            NextSubpass,
            ExecuteTask { node: node2 },
            EndRenderPass,
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn render_pass2() {
        let (resources, queues) = test_queues!();

        let mut graph = TaskGraph::<()>::new(&resources);
        let color_image = graph.add_image(&ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_READ | AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .input_attachment(
                color_image,
                AccessTypes::FRAGMENT_SHADER_COLOR_INPUT_ATTACHMENT_READ,
                ImageLayoutType::General,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::General,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        graph.add_edge(node1, node2).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            BeginRenderPass,
            ExecuteTask { node: node1 },
            NextSubpass,
            ClearAttachments {
                node: node2,
                attachments: [color_image, depth_image],
            },
            ExecuteTask { node: node2 },
            EndRenderPass,
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn render_pass3() {
        let (resources, queues) = test_queues!();

        let mut graph = TaskGraph::<()>::new(&resources);
        let color_image = graph.add_image(&ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_READ | AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        let node3 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .input_attachment(
                color_image,
                AccessTypes::FRAGMENT_SHADER_COLOR_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            BeginRenderPass,
            ExecuteTask { node: node1 },
            ClearAttachments {
                node: node2,
                attachments: [color_image, depth_image],
            },
            ExecuteTask { node: node2 },
            NextSubpass,
            ExecuteTask { node: node3 },
            EndRenderPass,
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn queue_family_ownership_transfer1() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        let mut graph = TaskGraph::<()>::new(&resources);
        let buffer1 = graph.add_buffer(&BufferCreateInfo::default());
        let buffer2 = graph.add_buffer(&BufferCreateInfo::default());
        let image1 = graph.add_image(&ImageCreateInfo::default());
        let image2 = graph.add_image(&ImageCreateInfo::default());
        let compute_node = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer1, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
            .buffer_access(buffer2, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
            .image_access(
                image1,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                image2,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::Optimal,
            )
            .build();
        let graphics_node = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer1, AccessTypes::INDEX_READ)
            .buffer_access(buffer2, AccessTypes::VERTEX_SHADER_UNIFORM_READ)
            .image_access(
                image1,
                AccessTypes::VERTEX_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .image_access(
                image2,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(compute_node, graphics_node).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: compute_node },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer1,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer2,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: General,
                        new_layout: General,
                        resource: image1,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ShaderReadOnlyOptimal,
                        new_layout: General,
                        resource: image2,
                    },
                ],
            },
            SignalSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: INDEX_INPUT,
                        dst_access_mask: INDEX_READ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer1,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: VERTEX_SHADER,
                        dst_access_mask: UNIFORM_READ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer2,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: VERTEX_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: General,
                        new_layout: General,
                        resource: image1,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: FRAGMENT_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: ShaderReadOnlyOptimal,
                        new_layout: General,
                        resource: image2,
                    },
                ],
            },
            ExecuteTask { node: graphics_node },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn queue_family_ownership_transfer2() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        let mut graph = TaskGraph::<()>::new(&resources);
        let queue_family_indices = queues
            .iter()
            .map(|q| q.queue_family_index())
            .collect::<Vec<_>>();
        let sharing = Sharing::Concurrent(&queue_family_indices);
        let buffer1 = graph.add_buffer(&BufferCreateInfo {
            sharing,
            ..Default::default()
        });
        let buffer2 = graph.add_buffer(&BufferCreateInfo {
            sharing,
            ..Default::default()
        });
        let image1 = graph.add_image(&ImageCreateInfo {
            sharing,
            ..Default::default()
        });
        let image2 = graph.add_image(&ImageCreateInfo {
            sharing,
            ..Default::default()
        });
        let compute_node = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer1, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
            .buffer_access(buffer2, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
            .image_access(
                image1,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                image2,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::Optimal,
            )
            .build();
        let graphics_node = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .buffer_access(buffer1, AccessTypes::INDEX_READ)
            .buffer_access(buffer2, AccessTypes::VERTEX_SHADER_UNIFORM_READ)
            .image_access(
                image1,
                AccessTypes::VERTEX_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .image_access(
                image2,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(compute_node, graphics_node).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: compute_node },
            SignalSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: FRAGMENT_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: ShaderReadOnlyOptimal,
                        new_layout: General,
                        resource: image2,
                    },
                ],
            },
            ExecuteTask { node: graphics_node },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn queue_family_ownership_transfer3() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        let mut graph = TaskGraph::<()>::new(&resources);
        let buffer1 = graph.add_buffer(&BufferCreateInfo::default());
        let buffer2 = graph.add_buffer(&BufferCreateInfo::default());
        let image1 = graph.add_image(&ImageCreateInfo::default());
        let image2 = graph.add_image(&ImageCreateInfo::default());
        let color_image = graph.add_image(&ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let compute_node = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer1, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
            .buffer_access(buffer2, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
            .image_access(
                image1,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                image2,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::Optimal,
            )
            .build();
        let graphics_node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        let graphics_node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .input_attachment(
                depth_image,
                AccessTypes::FRAGMENT_SHADER_DEPTH_STENCIL_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .buffer_access(buffer1, AccessTypes::INDEX_READ)
            .buffer_access(buffer2, AccessTypes::VERTEX_SHADER_UNIFORM_READ)
            .image_access(
                image1,
                AccessTypes::VERTEX_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .image_access(
                image2,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(compute_node, graphics_node1).unwrap();
        graph.add_edge(graphics_node1, graphics_node2).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask { node: compute_node },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer1,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer2,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: General,
                        new_layout: General,
                        resource: image1,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ShaderReadOnlyOptimal,
                        new_layout: General,
                        resource: image2,
                    },
                ],
            },
            SignalSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: INDEX_INPUT,
                        dst_access_mask: INDEX_READ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer1,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: VERTEX_SHADER,
                        dst_access_mask: UNIFORM_READ,
                        old_layout: Undefined,
                        new_layout: Undefined,
                        resource: buffer2,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: VERTEX_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: General,
                        new_layout: General,
                        resource: image1,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: FRAGMENT_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: ShaderReadOnlyOptimal,
                        new_layout: General,
                        resource: image2,
                    },
                ],
            },
            BeginRenderPass,
            ExecuteTask { node: graphics_node1 },
            NextSubpass,
            ExecuteTask { node: graphics_node2 },
            EndRenderPass,
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn queue_family_ownership_transfer4() {
        let (resources, queues) = test_queues!();

        if !has_compute_only_queue(&queues) {
            return;
        }

        let mut graph = TaskGraph::<()>::new(&resources);
        let queue_family_indices = queues
            .iter()
            .map(|q| q.queue_family_index())
            .collect::<Vec<_>>();
        let sharing = Sharing::Concurrent(&queue_family_indices);
        let buffer1 = graph.add_buffer(&BufferCreateInfo {
            sharing,
            ..Default::default()
        });
        let buffer2 = graph.add_buffer(&BufferCreateInfo {
            sharing,
            ..Default::default()
        });
        let image1 = graph.add_image(&ImageCreateInfo {
            sharing,
            ..Default::default()
        });
        let image2 = graph.add_image(&ImageCreateInfo {
            sharing,
            ..Default::default()
        });
        let color_image = graph.add_image(&ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let compute_node = graph
            .create_task_node("", QueueFamilyType::Compute, PhantomData)
            .buffer_access(buffer1, AccessTypes::COMPUTE_SHADER_STORAGE_WRITE)
            .buffer_access(buffer2, AccessTypes::COMPUTE_SHADER_STORAGE_READ)
            .image_access(
                image1,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                image2,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::Optimal,
            )
            .build();
        let graphics_node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        let graphics_node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                color_image,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .input_attachment(
                depth_image,
                AccessTypes::FRAGMENT_SHADER_DEPTH_STENCIL_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .buffer_access(buffer1, AccessTypes::INDEX_READ)
            .buffer_access(buffer2, AccessTypes::VERTEX_SHADER_UNIFORM_READ)
            .image_access(
                image1,
                AccessTypes::VERTEX_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .image_access(
                image2,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();
        graph.add_edge(compute_node, graphics_node1).unwrap();
        graph.add_edge(graphics_node1, graphics_node2).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            ExecuteTask {
                node: compute_node,
            },
            SignalSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            FlushSubmit,
            Submit,
            WaitSemaphore {
                semaphore_index: semaphore,
                stage_mask: ALL_COMMANDS,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: FRAGMENT_SHADER,
                        dst_access_mask: SHADER_SAMPLED_READ,
                        old_layout: ShaderReadOnlyOptimal,
                        new_layout: General,
                        resource: image2,
                    },
                ],
            },
            BeginRenderPass,
            ExecuteTask { node: graphics_node1 },
            NextSubpass,
            ExecuteTask { node: graphics_node2 },
            EndRenderPass,
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn swapchain1() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();

        let present_queue = queues.iter().find(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::GRAPHICS)
        });

        let mut graph = TaskGraph::<()>::new(&resources);
        let swapchain1 = graph.add_swapchain(&SwapchainCreateInfo::default());
        let swapchain2 = graph.add_swapchain(&SwapchainCreateInfo::default());
        let node = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .image_access(
                swapchain1.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                swapchain2.current_image_id(),
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .build();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                present_queue: Some(present_queue.unwrap()),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            WaitAcquire {
                swapchain_id: swapchain1,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            WaitAcquire {
                swapchain_id: swapchain2,
                stage_mask: COMPUTE_SHADER,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: ,
                        dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        dst_access_mask: COLOR_ATTACHMENT_WRITE,
                        old_layout: Undefined,
                        new_layout: ColorAttachmentOptimal,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: COMPUTE_SHADER,
                        dst_access_mask: SHADER_STORAGE_WRITE,
                        old_layout: Undefined,
                        new_layout: General,
                        resource: swapchain2,
                    },
                ],
            },
            ExecuteTask { node: node },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: General,
                        new_layout: PresentSrc,
                        resource: swapchain2,
                    },
                ],
            },
            SignalPresent {
                swapchain_id: swapchain1,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            SignalPresent {
                swapchain_id: swapchain2,
                stage_mask: COMPUTE_SHADER,
            },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn swapchain2() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();

        let present_queue = queues.iter().find(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS)
        });

        if !present_queue.is_some() {
            return;
        }

        let mut graph = TaskGraph::<()>::new(&resources);
        let queue_family_indices = queues
            .iter()
            .map(|q| q.queue_family_index())
            .collect::<Vec<_>>();
        let concurrent_sharing = Sharing::Concurrent(&queue_family_indices);
        let swapchain1 = graph.add_swapchain(&SwapchainCreateInfo::default());
        let swapchain2 = graph.add_swapchain(&SwapchainCreateInfo {
            image_sharing: concurrent_sharing,
            ..Default::default()
        });
        let swapchain3 = graph.add_swapchain(&SwapchainCreateInfo::default());
        let swapchain4 = graph.add_swapchain(&SwapchainCreateInfo {
            image_sharing: concurrent_sharing,
            ..Default::default()
        });
        let node = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .image_access(
                swapchain1.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                swapchain2.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                swapchain3.current_image_id(),
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .image_access(
                swapchain4.current_image_id(),
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::Optimal,
            )
            .build();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                present_queue: Some(present_queue.unwrap()),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            WaitAcquire {
                swapchain_id: swapchain1,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            WaitAcquire {
                swapchain_id: swapchain2,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            WaitAcquire {
                swapchain_id: swapchain3,
                stage_mask: COMPUTE_SHADER,
            },
            WaitAcquire {
                swapchain_id: swapchain4,
                stage_mask: COMPUTE_SHADER,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: ,
                        dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        dst_access_mask: COLOR_ATTACHMENT_WRITE,
                        old_layout: Undefined,
                        new_layout: ColorAttachmentOptimal,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: ,
                        dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        dst_access_mask: COLOR_ATTACHMENT_WRITE,
                        old_layout: Undefined,
                        new_layout: ColorAttachmentOptimal,
                        resource: swapchain2,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: COMPUTE_SHADER,
                        dst_access_mask: SHADER_STORAGE_WRITE,
                        old_layout: Undefined,
                        new_layout: General,
                        resource: swapchain3,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: ,
                        dst_stage_mask: COMPUTE_SHADER,
                        dst_access_mask: SHADER_STORAGE_WRITE,
                        old_layout: Undefined,
                        new_layout: General,
                        resource: swapchain4,
                    },
                ],
            },
            ExecuteTask { node: node },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain2,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: General,
                        new_layout: PresentSrc,
                        resource: swapchain3,
                    },
                    {
                        src_stage_mask: COMPUTE_SHADER,
                        src_access_mask: SHADER_STORAGE_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: General,
                        new_layout: PresentSrc,
                        resource: swapchain4,
                    },
                ],
            },
            SignalPrePresent {
                swapchain_id: swapchain1,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            SignalPrePresent {
                swapchain_id: swapchain3,
                stage_mask: COMPUTE_SHADER,
            },
            SignalPresent {
                swapchain_id: swapchain2,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            SignalPresent {
                swapchain_id: swapchain4,
                stage_mask: COMPUTE_SHADER,
            },
            FlushSubmit,
            Submit,
            WaitPrePresent {
                swapchain_id: swapchain1,
                stage_mask: ALL_COMMANDS,
            },
            SignalPresent {
                swapchain_id: swapchain1,
                stage_mask: ALL_COMMANDS,
            },
            WaitPrePresent {
                swapchain_id: swapchain3,
                stage_mask: ALL_COMMANDS,
            },
            SignalPresent {
                swapchain_id: swapchain3,
                stage_mask: ALL_COMMANDS,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: General,
                        new_layout: PresentSrc,
                        resource: swapchain3,
                    },
                ],
            },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn swapchain3() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();

        let present_queue = queues.iter().find(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::GRAPHICS)
        });

        let mut graph = TaskGraph::<()>::new(&resources);
        let swapchain = graph.add_swapchain(&SwapchainCreateInfo {
            image_format: Format::R8G8B8A8_UNORM,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                swapchain.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        let node3 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .input_attachment(
                depth_image,
                AccessTypes::FRAGMENT_SHADER_DEPTH_STENCIL_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                present_queue: Some(present_queue.unwrap()),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            WaitAcquire {
                swapchain_id: swapchain,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: ,
                        dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        dst_access_mask: COLOR_ATTACHMENT_WRITE,
                        old_layout: Undefined,
                        new_layout: ColorAttachmentOptimal,
                        resource: swapchain,
                    },
                ],
            },
            BeginRenderPass,
            ExecuteTask { node: node1 },
            NextSubpass,
            ExecuteTask { node: node2 },
            NextSubpass,
            ExecuteTask { node: node3 },
            EndRenderPass,
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain,
                    },
                ],
            },
            SignalPresent {
                swapchain_id: swapchain,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            FlushSubmit,
            Submit,
        );
    }

    #[test]
    fn swapchain4() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();

        let present_queue = queues.iter().find(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS)
        });

        if !present_queue.is_some() {
            return;
        }

        let mut graph = TaskGraph::<()>::new(&resources);
        let queue_family_indices = queues
            .iter()
            .map(|q| q.queue_family_index())
            .collect::<Vec<_>>();
        let concurrent_sharing = Sharing::Concurrent(&queue_family_indices);
        let swapchain1 = graph.add_swapchain(&SwapchainCreateInfo {
            image_format: Format::R8G8B8A8_UNORM,
            ..SwapchainCreateInfo::default()
        });
        let swapchain2 = graph.add_swapchain(&SwapchainCreateInfo {
            image_format: Format::R8G8B8A8_UNORM,
            image_sharing: concurrent_sharing,
            ..Default::default()
        });
        let depth_image = graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            ..Default::default()
        });
        let framebuffer = graph.add_framebuffer();
        let node1 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        let node2 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .color_attachment(
                swapchain1.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .color_attachment(
                swapchain2.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 1,
                    ..Default::default()
                },
            )
            .depth_stencil_attachment(
                depth_image,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        let node3 = graph
            .create_task_node("", QueueFamilyType::Graphics, PhantomData)
            .framebuffer(framebuffer)
            .input_attachment(
                depth_image,
                AccessTypes::FRAGMENT_SHADER_DEPTH_STENCIL_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .build();
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        let graph = unsafe {
            graph.compile(&CompileInfo {
                queues: &queues.iter().collect::<Vec<_>>(),
                present_queue: Some(present_queue.unwrap()),
                ..Default::default()
            })
        }
        .unwrap();

        assert_matches_instructions!(
            graph,
            WaitAcquire {
                swapchain_id: swapchain1,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            WaitAcquire {
                swapchain_id: swapchain2,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: ,
                        dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        dst_access_mask: COLOR_ATTACHMENT_WRITE,
                        old_layout: Undefined,
                        new_layout: ColorAttachmentOptimal,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: ,
                        dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        dst_access_mask: COLOR_ATTACHMENT_WRITE,
                        old_layout: Undefined,
                        new_layout: ColorAttachmentOptimal,
                        resource: swapchain2,
                    },
                ],
            },
            BeginRenderPass,
            ExecuteTask { node: node1 },
            NextSubpass,
            ExecuteTask { node: node2 },
            NextSubpass,
            ExecuteTask { node: node3 },
            EndRenderPass,
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain1,
                    },
                    {
                        src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                        src_access_mask: COLOR_ATTACHMENT_WRITE,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain2,
                    },
                ],
            },
            SignalPrePresent {
                swapchain_id: swapchain1,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            SignalPresent {
                swapchain_id: swapchain2,
                stage_mask: COLOR_ATTACHMENT_OUTPUT,
            },
            FlushSubmit,
            Submit,
            WaitPrePresent {
                swapchain_id: swapchain1,
                stage_mask: ALL_COMMANDS,
            },
            SignalPresent {
                swapchain_id: swapchain1,
                stage_mask: ALL_COMMANDS,
            },
            PipelineBarrier {
                barriers: [
                    {
                        src_stage_mask: ,
                        src_access_mask: ,
                        dst_stage_mask: ,
                        dst_access_mask: ,
                        old_layout: ColorAttachmentOptimal,
                        new_layout: PresentSrc,
                        resource: swapchain1,
                    },
                ],
            },
            FlushSubmit,
            Submit,
        );
    }

    fn has_compute_only_queue(queues: &[Arc<Queue>]) -> bool {
        let queue_family_properties = queues[0]
            .device()
            .physical_device()
            .queue_family_properties();

        queues.iter().any(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS)
        })
    }

    struct MatchingState {
        submission_index: usize,
        instruction_index: usize,
        semaphores: foldhash::HashMap<&'static str, SemaphoreIndex>,
    }

    macro_rules! assert_matches_instructions {
        (
            $graph:ident,
            $($arg:tt)+
        ) => {
            let mut state = MatchingState {
                submission_index: 0,
                instruction_index: 0,
                semaphores: Default::default(),
            };
            assert_matches_instructions!(@ $graph, state, $($arg)+);
        };
        (
            @
            $graph:ident,
            $state:ident,
            InitialPipelineBarrier {
                barriers: [
                    $({
                        dst_stage_mask: $($dst_stage:ident)|*,
                        dst_access_mask: $($dst_access:ident)|*,
                        new_layout: $new_layout:ident,
                        resource: $resource:ident,
                    },)*
                ],
            },
            $($arg:tt)*
        ) => {
            let submission = &$graph.submissions[$state.submission_index];
            let barrier_range = &submission.initial_barrier_range;
            let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
            let barriers = &$graph.barriers[barrier_range];

            #[allow(unused_mut)]
            let mut barrier_count = 0;
            $(
                let barrier = barriers
                    .iter()
                    .find(|barrier| barrier.resource == $resource.erase())
                    .unwrap();
                assert_eq!(barrier.src_stage_mask, PipelineStages::empty());
                assert_eq!(barrier.src_access_mask, AccessFlags::empty());
                assert_eq!(
                    barrier.dst_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$dst_stage)*,
                );
                assert_eq!(
                    barrier.dst_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$dst_access)*,
                );
                assert_eq!(barrier.old_layout, ImageLayout::Undefined);
                assert_eq!(barrier.new_layout, ImageLayout::$new_layout);
                barrier_count += 1;
            )*
            assert_eq!(barriers.len(), barrier_count);

            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            WaitAcquire {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            WaitSemaphore {
                semaphore_index: $semaphore_index:ident,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::WaitSemaphore {
                    stage_mask,
                    ..
                } if stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            let Instruction::WaitSemaphore { semaphore_index, .. } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };

            assert_eq!(
                semaphore_index,
                $state.semaphores.get(stringify!($semaphore_index)).unwrap(),
            );

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            ExecuteTask {
                node: $node:expr $(,)?
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::ExecuteTask { node_index } if node_index == $node.index(),
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            PipelineBarrier {
                barriers: [
                    $({
                        src_stage_mask: $($src_stage:ident)|*,
                        src_access_mask: $($src_access:ident)|*,
                        dst_stage_mask: $($dst_stage:ident)|*,
                        dst_access_mask: $($dst_access:ident)|*,
                        old_layout: $old_layout:ident,
                        new_layout: $new_layout:ident,
                        resource: $resource:ident,
                    },)+
                ],
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::PipelineBarrier { .. },
            ));
            let Instruction::PipelineBarrier { barrier_range } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };
            let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
            let barriers = &$graph.barriers[barrier_range];

            let mut barrier_count = 0;
            $(
                let barrier = barriers
                    .iter()
                    .find(|barrier| barrier.resource == $resource.erase())
                    .unwrap();
                assert_eq!(
                    barrier.src_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$src_stage)*,
                );
                assert_eq!(
                    barrier.src_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$src_access)*,
                );
                assert_eq!(
                    barrier.dst_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$dst_stage)*,
                );
                assert_eq!(
                    barrier.dst_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$dst_access)*,
                );
                assert_eq!(barrier.old_layout, ImageLayout::$old_layout);
                assert_eq!(barrier.new_layout, ImageLayout::$new_layout);
                barrier_count += 1;
            )+
            assert_eq!(barriers.len(), barrier_count);

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            BeginRenderPass,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::BeginRenderPass { .. },
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            NextSubpass,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::NextSubpass,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            EndRenderPass,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::EndRenderPass,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            ClearAttachments {
                node: $node:ident,
                attachments: [$($resource:ident),+ $(,)?],
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::ClearAttachments { node_index, .. } if node_index == $node.index(),
            ));
            let Instruction::ClearAttachments { clear_attachment_range, .. } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };
            let clear_attachments = &$graph.clear_attachments[clear_attachment_range.clone()];

            let mut clear_attachment_count = 0;
            $(
                assert!(clear_attachments.contains(&$resource.erase()));
                clear_attachment_count += 1;
            )+
            assert_eq!(clear_attachments.len(), clear_attachment_count);

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            SignalSemaphore {
                semaphore_index: $semaphore_index:ident,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::SignalSemaphore {
                    stage_mask,
                    ..
                } if stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            let Instruction::SignalSemaphore { semaphore_index, .. } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };

            assert!($state.semaphores.get(&stringify!($semaphore_index)).is_none());
            $state.semaphores.insert(stringify!($semaphore_index), *semaphore_index);

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            SignalPrePresent {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            WaitPrePresent {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            SignalPresent {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            FlushSubmit,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::FlushSubmit,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            Submit,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::Submit,
            ));
            $state.submission_index += 1;
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
        ) => {
            assert_eq!($graph.submissions.len(), $state.submission_index);
            assert_eq!($graph.instructions.len(), $state.instruction_index);
        };
    }
    use assert_matches_instructions;
}
