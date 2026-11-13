//! The task graph data structure and associated types.

pub use self::{
    compile::{CompileError, CompileErrorKind, CompileInfo},
    execute::{ExecuteError, ResourceMap},
};
use crate::{
    linear_map::LinearMap,
    resource::{self, AccessTypes, Flight, HostAccessType, ImageLayoutType},
    slotmap::{self, declare_key, Iter, IterMut, SlotMap},
    Id, InvalidSlotError, Object, ObjectType, QueueFamilyType, Task,
};
use ash::vk;
use concurrent_slotmap::SlotId;
use foldhash::HashMap;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    error::Error,
    fmt, hint,
    iter::{self, FusedIterator},
    mem::ManuallyDrop,
    ops::Range,
    sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo},
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{
        sampler::ComponentMapping, Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageUsage,
        SampleCount,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    swapchain::{Swapchain, SwapchainCreateInfo},
    sync::{semaphore::Semaphore, AccessFlags, PipelineStages},
};

mod compile;
mod execute;

/// The task graph is a [directed acyclic graph] consisting of [`Task`] nodes, with edges
/// representing happens-before relations.
///
/// [directed acyclic graph]: https://en.wikipedia.org/wiki/Directed_acyclic_graph
pub struct TaskGraph<W: ?Sized> {
    nodes: Nodes<W>,
    resources: Resources,
}

struct Nodes<W: ?Sized> {
    inner: SlotMap<NodeId, Node<W>>,
}

struct Node<W: ?Sized> {
    // TODO:
    #[allow(unused)]
    name: Cow<'static, str>,
    inner: NodeInner<W>,
    in_edges: Vec<NodeIndex>,
    out_edges: Vec<NodeIndex>,
}

enum NodeInner<W: ?Sized> {
    Task(TaskNode<W>),
    // TODO:
    #[allow(unused)]
    Semaphore,
}

type NodeIndex = u32;

pub(crate) struct Resources {
    inner: SlotMap<Id, ResourceInfo>,
    physical_resources: Arc<resource::Resources>,
    physical_map: HashMap<Id, Id>,
    host_reads: Vec<Id<Buffer>>,
    host_writes: Vec<Id<Buffer>>,
    framebuffers: SlotMap<Id<Framebuffer>, ()>,
}

struct ResourceInfo {
    format: Format,
    samples: SampleCount,
    usage: ImageUsage,
}

impl<W: ?Sized> TaskGraph<W> {
    /// Creates a new `TaskGraph`.
    #[must_use]
    pub fn new(physical_resources: &Arc<resource::Resources>) -> Self {
        TaskGraph {
            nodes: Nodes {
                inner: SlotMap::with_key(),
            },
            resources: Resources {
                inner: SlotMap::with_key(),
                physical_resources: physical_resources.clone(),
                physical_map: HashMap::default(),
                host_reads: Vec::new(),
                host_writes: Vec::new(),
                framebuffers: SlotMap::with_key(),
            },
        }
    }

    /// Creates a new [`TaskNode`] from the given `task` and adds it to the graph. Returns a
    /// builder allowing you to add resource accesses to the task node.
    ///
    /// `queue_family_type` is the type of queue family the task can be executed on.
    #[must_use]
    pub fn create_task_node(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        queue_family_type: QueueFamilyType,
        task: impl Task<World = W>,
    ) -> TaskNodeBuilder<'_> {
        let id = self.nodes.add_node(
            name.into(),
            NodeInner::Task(TaskNode::new(queue_family_type, task)),
        );

        // SAFETY: We just inserted this task node.
        let task_node = unsafe { self.nodes.task_node_unchecked_mut(id.index()) };

        TaskNodeBuilder {
            id,
            accesses: &mut task_node.accesses,
            attachments: &mut task_node.attachments,
            resources: &mut self.resources,
        }
    }

    /// Removes the task node corresponding to `id` from the graph.
    pub fn remove_task_node(&mut self, id: NodeId) -> Result<TaskNode<W>> {
        self.task_node(id)?;

        let task = match self.nodes.remove_node(id).inner {
            NodeInner::Task(task) => task,
            // We checked that the node is a task above.
            _ => unreachable!(),
        };

        Ok(task)
    }

    /// Adds an edge starting at the node corresponding to `from` and ending at the node
    /// corresponding to `to`.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result {
        let [from_node, to_node] = self.nodes.node_many_mut([from, to])?;
        let out_edges = &mut from_node.out_edges;
        let in_edges = &mut to_node.in_edges;

        if !out_edges.contains(&to.index()) {
            out_edges.push(to.index());
            in_edges.push(from.index());

            Ok(())
        } else {
            Err(TaskGraphError::DuplicateEdge)
        }
    }

    /// Removes an edge starting at the node corresponding to `from` and ending at the node
    /// corresponding to `to`.
    pub fn remove_edge(&mut self, from: NodeId, to: NodeId) -> Result {
        let [from_node, to_node] = self.nodes.node_many_mut([from, to])?;
        let out_edges = &mut from_node.out_edges;
        let in_edges = &mut to_node.in_edges;

        if let Some(index) = out_edges.iter().position(|&i| i == to.index()) {
            out_edges.remove(index);
            let edge_index = in_edges.iter().position(|&i| i == from.index()).unwrap();
            in_edges.remove(edge_index);

            Ok(())
        } else {
            Err(TaskGraphError::InvalidEdge)
        }
    }

    /// Returns a reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node(&self, id: NodeId) -> Result<&TaskNode<W>> {
        self.nodes.task_node(id)
    }

    /// Returns a mutable reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node_mut(&mut self, id: NodeId) -> Result<&mut TaskNode<W>> {
        self.nodes.task_node_mut(id)
    }

    /// Returns an iterator over all [`TaskNode`]s.
    #[inline]
    pub fn task_nodes(&self) -> TaskNodes<'_, W> {
        TaskNodes {
            inner: self.nodes.nodes(),
        }
    }

    /// Returns an iterator over all [`TaskNode`]s that allows you to mutate them.
    #[inline]
    pub fn task_nodes_mut(&mut self) -> TaskNodesMut<'_, W> {
        TaskNodesMut {
            inner: self.nodes.nodes_mut(),
        }
    }

    /// Add a [virtual buffer resource] to the task graph.
    #[must_use]
    pub fn add_buffer(&mut self, create_info: &BufferCreateInfo<'_>) -> Id<Buffer> {
        self.resources.add_buffer(create_info)
    }

    /// Add a [virtual image resource] to the task graph.
    #[must_use]
    pub fn add_image(&mut self, create_info: &ImageCreateInfo<'_>) -> Id<Image> {
        self.resources.add_image(create_info)
    }

    /// Add a [virtual swapchain resource] to the task graph.
    #[must_use]
    pub fn add_swapchain(&mut self, create_info: &SwapchainCreateInfo<'_>) -> Id<Swapchain> {
        self.resources.add_swapchain(create_info)
    }

    /// Adds a host buffer access to the task graph.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    pub fn add_host_buffer_access(&mut self, id: Id<Buffer>, access_type: HostAccessType) {
        self.resources.add_host_buffer_access(id, access_type)
    }

    /// Adds a [virtual framebuffer] to the task graph.
    #[must_use]
    pub fn add_framebuffer(&mut self) -> Id<Framebuffer> {
        self.resources.add_framebuffer()
    }
}

impl<W: ?Sized> Nodes<W> {
    fn add_node(&mut self, name: Cow<'static, str>, inner: NodeInner<W>) -> NodeId {
        self.inner.insert(Node {
            name,
            inner,
            in_edges: Vec::new(),
            out_edges: Vec::new(),
        })
    }

    fn remove_node(&mut self, id: NodeId) -> Node<W> {
        let node = self.inner.remove(id).unwrap();

        // NOTE(Marc): We must not leave any broken edges because the rest of the code relies on
        // this being impossible.

        for &in_node_index in &node.in_edges {
            // SAFETY: By our own invariant, node indices in the edges vectors must be valid.
            let out_edges = &mut unsafe { self.node_unchecked_mut(in_node_index) }.out_edges;
            let edge_index = out_edges.iter().position(|&i| i == id.index()).unwrap();
            out_edges.remove(edge_index);
        }

        for &out_node_index in &node.out_edges {
            // SAFETY: By our own invariant, node indices in the edges vectors must be valid.
            let in_edges = &mut unsafe { self.node_unchecked_mut(out_node_index) }.in_edges;
            let edge_index = in_edges.iter().position(|&i| i == id.index()).unwrap();
            in_edges.remove(edge_index);
        }

        node
    }

    fn reserved_len(&self) -> u32 {
        self.inner.reserved_len()
    }

    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn task_node(&self, id: NodeId) -> Result<&TaskNode<W>> {
        let node = self.node(id)?;

        if let NodeInner::Task(task_node) = &node.inner {
            Ok(task_node)
        } else {
            Err(TaskGraphError::InvalidNodeType)
        }
    }

    unsafe fn task_node_unchecked(&self, index: NodeIndex) -> &TaskNode<W> {
        // SAFETY: The caller must ensure that the `index` is valid.
        let node = unsafe { self.node_unchecked(index) };

        if let NodeInner::Task(task_node) = &node.inner {
            task_node
        } else {
            // SAFETY: The caller must ensure that the node is a `TaskNode`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn task_node_mut(&mut self, id: NodeId) -> Result<&mut TaskNode<W>> {
        let node = self.node_mut(id)?;

        if let NodeInner::Task(task_node) = &mut node.inner {
            Ok(task_node)
        } else {
            Err(TaskGraphError::InvalidNodeType)
        }
    }

    unsafe fn task_node_unchecked_mut(&mut self, index: NodeIndex) -> &mut TaskNode<W> {
        // SAFETY: The caller must ensure that the `index` is valid.
        let node = unsafe { self.node_unchecked_mut(index) };

        if let NodeInner::Task(task_node) = &mut node.inner {
            task_node
        } else {
            // SAFETY: The caller must ensure that the node is a `TaskNode`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn node(&self, id: NodeId) -> Result<&Node<W>> {
        self.inner.get(id).ok_or(TaskGraphError::InvalidNode)
    }

    unsafe fn node_unchecked(&self, index: NodeIndex) -> &Node<W> {
        // SAFETY: The generation's state tag is `OCCUPIED_TAG`.
        let id = NodeId(unsafe { SlotId::new_unchecked(index, SlotId::OCCUPIED_TAG) });

        // SAFETY: The caller must ensure that the `index` is valid.
        unsafe { self.inner.get_unchecked(id) }
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node<W>> {
        self.inner.get_mut(id).ok_or(TaskGraphError::InvalidNode)
    }

    unsafe fn node_unchecked_mut(&mut self, index: NodeIndex) -> &mut Node<W> {
        // SAFETY: The generation's state tag is `OCCUPIED_TAG`.
        let id = NodeId(unsafe { SlotId::new_unchecked(index, SlotId::OCCUPIED_TAG) });

        // SAFETY: The caller must ensure that the `index` is valid.
        unsafe { self.inner.get_unchecked_mut(id) }
    }

    fn node_many_mut<const N: usize>(&mut self, ids: [NodeId; N]) -> Result<[&mut Node<W>; N]> {
        self.inner
            .get_many_mut(ids)
            .ok_or(TaskGraphError::InvalidNode)
    }

    fn nodes(&self) -> Iter<'_, NodeId, Node<W>> {
        self.inner.iter()
    }

    fn nodes_mut(&mut self) -> IterMut<'_, NodeId, Node<W>> {
        self.inner.iter_mut()
    }
}

impl Resources {
    fn add_buffer(&mut self, create_info: &BufferCreateInfo<'_>) -> Id<Buffer> {
        let mut tag = Buffer::TAG | Id::VIRTUAL_BIT;

        if create_info.sharing.is_exclusive() {
            tag |= Id::EXCLUSIVE_BIT;
        }

        let resource_info = ResourceInfo {
            format: Format::UNDEFINED,
            samples: SampleCount::Sample1,
            usage: ImageUsage::empty(),
        };

        let id = self.inner.insert_with_tag(resource_info, tag);

        unsafe { id.parametrize() }
    }

    fn add_image(&mut self, create_info: &ImageCreateInfo<'_>) -> Id<Image> {
        let mut tag = Image::TAG | Id::VIRTUAL_BIT;

        if create_info.sharing.is_exclusive() {
            tag |= Id::EXCLUSIVE_BIT;
        }

        let resource_info = ResourceInfo {
            format: create_info.format,
            samples: create_info.samples,
            usage: create_info.usage,
        };

        let id = self.inner.insert_with_tag(resource_info, tag);

        unsafe { id.parametrize() }
    }

    fn add_swapchain(&mut self, create_info: &SwapchainCreateInfo<'_>) -> Id<Swapchain> {
        let mut tag = Swapchain::TAG | Id::VIRTUAL_BIT;

        if create_info.image_sharing.is_exclusive() {
            tag |= Id::EXCLUSIVE_BIT;
        }

        let resource_info = ResourceInfo {
            format: create_info.image_format,
            samples: SampleCount::Sample1,
            usage: create_info.image_usage,
        };

        let id = self.inner.insert_with_tag(resource_info, tag);

        unsafe { id.parametrize() }
    }

    fn add_physical_buffer(
        &mut self,
        physical_id: Id<Buffer>,
    ) -> Result<Id<Buffer>, InvalidSlotError> {
        let physical_resources = self.physical_resources.clone();
        let buffer_state = physical_resources.try_buffer(physical_id)?;
        let buffer = buffer_state.buffer();
        let virtual_id = self.add_buffer(&BufferCreateInfo {
            sharing: buffer.sharing(),
            ..Default::default()
        });
        self.physical_map
            .insert(physical_id.erase(), virtual_id.erase());

        Ok(virtual_id)
    }

    fn add_physical_image(
        &mut self,
        physical_id: Id<Image>,
    ) -> Result<Id<Image>, InvalidSlotError> {
        let physical_resources = self.physical_resources.clone();
        let image_state = physical_resources.try_image(physical_id)?;
        let image = image_state.image();
        let virtual_id = self.add_image(&ImageCreateInfo {
            sharing: image.sharing(),
            ..Default::default()
        });
        self.physical_map
            .insert(physical_id.erase(), virtual_id.erase());

        Ok(virtual_id)
    }

    fn add_physical_swapchain(
        &mut self,
        id: Id<Swapchain>,
    ) -> Result<Id<Swapchain>, InvalidSlotError> {
        let physical_resources = self.physical_resources.clone();
        let swapchain_state = physical_resources.try_swapchain(id)?;
        let swapchain = swapchain_state.swapchain();
        let virtual_id = self.add_swapchain(&SwapchainCreateInfo {
            image_sharing: swapchain.image_sharing(),
            ..Default::default()
        });
        self.physical_map.insert(id.erase(), virtual_id.erase());

        Ok(virtual_id)
    }

    fn add_host_buffer_access(&mut self, mut id: Id<Buffer>, access_type: HostAccessType) {
        if id.is_virtual() {
            self.get(id.erase()).expect("invalid buffer ID");
        } else if let Some(&virtual_id) = self.physical_map.get(&id.erase()) {
            id = unsafe { virtual_id.parametrize() };
        } else {
            id = self.add_physical_buffer(id).expect("invalid buffer ID");
        }

        let host_accesses = match access_type {
            HostAccessType::Read => &mut self.host_reads,
            HostAccessType::Write => &mut self.host_writes,
        };

        if !host_accesses.contains(&id) {
            host_accesses.push(id);
        }
    }

    fn add_framebuffer(&mut self) -> Id<Framebuffer> {
        let tag = Framebuffer::TAG | Id::VIRTUAL_BIT;

        self.framebuffers.insert_with_tag((), tag)
    }

    fn reserved_len(&self) -> u32 {
        self.inner.reserved_len()
    }

    fn len(&self) -> u32 {
        self.inner.len()
    }

    pub(crate) fn physical_map(&self) -> &HashMap<Id, Id> {
        &self.physical_map
    }

    fn get(&self, id: Id) -> Result<&ResourceInfo, InvalidSlotError> {
        self.inner.get(id).ok_or(InvalidSlotError::new(id))
    }

    fn iter(&self) -> Iter<'_, Id, ResourceInfo> {
        self.inner.iter()
    }

    pub(crate) fn contains_host_buffer_access(
        &self,
        mut id: Id<Buffer>,
        access_type: HostAccessType,
    ) -> bool {
        if !id.is_virtual() {
            if let Some(&virtual_id) = self.physical_map.get(&id.erase()) {
                id = unsafe { virtual_id.parametrize() };
            } else {
                return false;
            }
        }

        let host_accesses = match access_type {
            HostAccessType::Read => &self.host_reads,
            HostAccessType::Write => &self.host_writes,
        };

        host_accesses.contains(&id)
    }

    fn framebuffer_mut(&mut self, id: Id<Framebuffer>) -> Option<&mut ()> {
        self.framebuffers.get_mut(id)
    }
}

impl<W: ?Sized> fmt::Debug for TaskGraph<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME:
        f.debug_struct("TaskGraph").finish_non_exhaustive()
    }
}

unsafe impl<W: ?Sized> DeviceOwned for TaskGraph<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.resources.physical_resources.device()
    }
}

declare_key! {
    /// The ID type used to refer to a node within a [`TaskGraph`].
    pub struct NodeId;
}

impl NodeId {
    fn index(self) -> NodeIndex {
        self.0.index()
    }
}

/// A node within a [`TaskGraph`] that represents a [`Task`] to be executed along with its resource
/// accesses.
pub struct TaskNode<W: ?Sized> {
    accesses: ResourceAccesses,
    attachments: Option<Attachments>,
    queue_family_type: QueueFamilyType,
    queue_family_index: u32,
    dependency_level_index: u32,
    subpass: Option<Subpass>,
    task: Box<dyn Task<World = W>>,
}

pub(crate) struct ResourceAccesses {
    inner: LinearMap<Id, ResourceAccess>,
}

#[derive(Clone, Copy, Default)]
struct ResourceAccess {
    stage_mask: PipelineStages,
    access_mask: AccessFlags,
    image_layout: ImageLayout,
    queue_family_index: u32,
}

pub(crate) struct Attachments {
    framebuffer_id: Id<Framebuffer>,
    input_attachments: LinearMap<Id, AttachmentInfo<'static>>,
    color_attachments: LinearMap<Id, AttachmentInfo<'static>>,
    depth_stencil_attachment: Option<(Id, AttachmentInfo<'static>)>,
}

impl<W: ?Sized> TaskNode<W> {
    fn new(queue_family_type: QueueFamilyType, task: impl Task<World = W>) -> Self {
        TaskNode {
            accesses: ResourceAccesses::new(),
            attachments: None,
            queue_family_type,
            queue_family_index: 0,
            dependency_level_index: 0,
            subpass: None,
            task: Box::new(task),
        }
    }

    /// Returns the queue family type the task node was created with.
    #[inline]
    #[must_use]
    pub fn queue_family_type(&self) -> QueueFamilyType {
        self.queue_family_type
    }

    /// Returns a reference to the task the task node was created with.
    #[inline]
    #[must_use]
    pub fn task(&self) -> &dyn Task<World = W> {
        &*self.task
    }

    /// Returns a mutable reference to the task the task node was created with.
    #[inline]
    #[must_use]
    pub fn task_mut(&mut self) -> &mut dyn Task<World = W> {
        &mut *self.task
    }

    /// Returns the subpass that the task node is part of, if any.
    #[inline]
    pub fn subpass(&self) -> Option<&Subpass> {
        self.subpass.as_ref()
    }
}

impl ResourceAccesses {
    const fn new() -> Self {
        ResourceAccesses {
            inner: LinearMap::new(),
        }
    }

    fn get(&self, id: Id) -> Option<&ResourceAccess> {
        self.inner.get(&id)
    }

    fn buffer_mut(
        &mut self,
        resources: &mut Resources,
        id: Id<Buffer>,
    ) -> (Id<Buffer>, Option<&mut ResourceAccess>) {
        let (id, access) = self
            .get_mut(resources, id.erase())
            .expect("invalid buffer ID");

        (unsafe { id.parametrize() }, access)
    }

    fn image_mut(
        &mut self,
        resources: &mut Resources,
        id: Id<Image>,
    ) -> (Id<Image>, Option<&mut ResourceAccess>) {
        let (id, access) = self
            .get_mut(resources, id.erase())
            .expect("invalid image ID");

        (unsafe { id.parametrize() }, access)
    }

    fn get_mut(
        &mut self,
        resources: &mut Resources,
        mut id: Id,
    ) -> Result<(Id, Option<&mut ResourceAccess>), InvalidSlotError> {
        if id.is_virtual() {
            resources.get(id)?;
        } else if let Some(&virtual_id) = resources.physical_map.get(&id) {
            id = virtual_id;
        } else {
            id = match id.object_type() {
                ObjectType::Buffer => resources
                    .add_physical_buffer(unsafe { id.parametrize() })?
                    .erase(),
                ObjectType::Image => resources
                    .add_physical_image(unsafe { id.parametrize() })?
                    .erase(),
                ObjectType::Swapchain => resources
                    .add_physical_swapchain(unsafe { id.parametrize() })?
                    .erase(),
                _ => unreachable!(),
            };
        }

        let access = self.inner.get_mut(&id);

        Ok((id, access))
    }

    fn iter(&self) -> impl Iterator<Item = (Id, &ResourceAccess)> + use<'_> {
        self.inner.iter().map(|(id, access)| (*id, access))
    }
}

impl Attachments {
    fn keys(&self) -> impl Iterator<Item = &Id> + use<'_> {
        self.input_attachments
            .keys()
            .chain(self.color_attachments.keys())
            .chain(self.depth_stencil_attachment.iter().map(|(id, _)| id))
    }

    fn iter(&self) -> impl Iterator<Item = (&Id, &AttachmentInfo<'static>)> + use<'_> {
        self.input_attachments
            .iter()
            .chain(self.color_attachments.iter())
            .chain(
                self.depth_stencil_attachment
                    .iter()
                    .map(|(id, attachment_state)| (id, attachment_state)),
            )
    }
}

const INPUT_ATTACHMENT_ACCESS_FLAGS: AccessFlags = AccessFlags::INPUT_ATTACHMENT_READ;
const COLOR_ATTACHMENT_ACCESS_FLAGS: AccessFlags =
    AccessFlags::COLOR_ATTACHMENT_READ.union(AccessFlags::COLOR_ATTACHMENT_WRITE);
const DEPTH_STENCIL_ATTACHMENT_ACCESS_FLAGS: AccessFlags =
    AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
const ATTACHMENT_ACCESS_FLAGS: AccessFlags = INPUT_ATTACHMENT_ACCESS_FLAGS
    .union(COLOR_ATTACHMENT_ACCESS_FLAGS)
    .union(DEPTH_STENCIL_ATTACHMENT_ACCESS_FLAGS);

const COLOR_ASPECTS: ImageAspects = ImageAspects::COLOR
    .union(ImageAspects::PLANE_0)
    .union(ImageAspects::PLANE_1)
    .union(ImageAspects::PLANE_2);
const DEPTH_STENCIL_ASPECTS: ImageAspects = ImageAspects::DEPTH.union(ImageAspects::STENCIL);

/// A builder used to add resource accesses and attachments to a [`TaskNode`].
pub struct TaskNodeBuilder<'a> {
    id: NodeId,
    accesses: &'a mut ResourceAccesses,
    attachments: &'a mut Option<Attachments>,
    resources: &'a mut Resources,
}

impl TaskNodeBuilder<'_> {
    /// Adds a buffer access to the task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_types` contains any access type that's not a valid buffer access type.
    #[track_caller]
    pub fn buffer_access(&mut self, id: Id<Buffer>, access_types: AccessTypes) -> &mut Self {
        assert!(access_types.are_valid_buffer_access_types());

        let (id, access) = self.accesses.buffer_mut(self.resources, id);

        if let Some(access) = access {
            access.stage_mask |= access_types.stage_mask();
            access.access_mask |= access_types.access_mask();
        } else {
            self.accesses.inner.insert(
                id.erase(),
                ResourceAccess {
                    stage_mask: access_types.stage_mask(),
                    access_mask: access_types.access_mask(),
                    image_layout: ImageLayout::Undefined,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            );
        }

        self
    }

    /// Adds an image access to the task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_types` contains any access type that's not a valid image access type.
    /// - Panics if `access_types` contains attachment access types as well as other access types.
    /// - Panics if `access_types` contains both color and depth/stencil access types.
    /// - Panics if an access for `id` was already added and its image layout doesn't equal
    ///   `access_types.image_layout(layout_type)`.
    #[track_caller]
    pub fn image_access(
        &mut self,
        id: Id<Image>,
        access_types: AccessTypes,
        layout_type: ImageLayoutType,
    ) -> &mut Self {
        assert!(access_types.are_valid_image_access_types());

        let (id, access) = self.accesses.image_mut(self.resources, id);

        let access_mask = access_types.access_mask();

        if access_mask.intersects(ATTACHMENT_ACCESS_FLAGS) {
            assert!(
                ATTACHMENT_ACCESS_FLAGS.contains(access_mask),
                "an image access that contains attachment access types must not contain any other \
                access types",
            );

            assert!(
                !(access_mask.intersects(COLOR_ATTACHMENT_ACCESS_FLAGS)
                    && access_mask.intersects(DEPTH_STENCIL_ATTACHMENT_ACCESS_FLAGS)),
                "an image access can't contain both color and depth/stencil attachment access \
                types",
            );
        }

        if let Some(access) = access {
            assert_eq!(access.image_layout, access_types.image_layout(layout_type));

            access.stage_mask |= access_types.stage_mask();
            access.access_mask |= access_types.access_mask();
        } else {
            self.accesses.inner.insert(
                id.erase(),
                ResourceAccess {
                    stage_mask: access_types.stage_mask(),
                    access_mask: access_types.access_mask(),
                    image_layout: access_types.image_layout(layout_type),
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            );
        }

        self
    }

    /// Sets the framebuffer that the task node's attachments will belong to. You must call this
    /// method before adding attachments to the task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid framebuffer ID.
    /// - Panics if the framebuffer was already set.
    #[track_caller]
    pub fn framebuffer(&mut self, id: Id<Framebuffer>) -> &mut Self {
        assert!(self.resources.framebuffer_mut(id).is_some());

        assert!(
            self.attachments.is_none(),
            "a task node must use at most one framebuffer",
        );

        *self.attachments = Some(Attachments {
            framebuffer_id: id,
            color_attachments: LinearMap::new(),
            input_attachments: LinearMap::new(),
            depth_stencil_attachment: None,
        });

        self
    }

    /// Adds an input attachment to the task node.
    ///
    /// # Panics
    ///
    /// - Panics if the framebuffer wasn't set beforehand.
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_types` contains any access type that's not an input attachment access
    ///   type.
    /// - Panics if an input attachment for `id` was already added.
    /// - Panics if an input attachment using the `attachment_info.index` was already added.
    /// - Panics if a color or depth/stencil attachment using `id` was added but the attachment
    ///   infos don't match.
    /// - Panics if an access for `id` was already added and its image layout doesn't equal
    ///   `access_types.image_layout(layout_type)`.
    #[track_caller]
    pub fn input_attachment(
        &mut self,
        id: Id<Image>,
        mut access_types: AccessTypes,
        layout_type: ImageLayoutType,
        attachment_info: &AttachmentInfo<'_>,
    ) -> &mut Self {
        let attachments = self
            .attachments
            .as_mut()
            .expect("the framebuffer must be set before adding attachments");

        let (id, _access) = self.accesses.image_mut(self.resources, id);

        assert!(INPUT_ATTACHMENT_ACCESS_FLAGS.contains(access_types.access_mask()));

        assert!(
            attachments.input_attachments.get(&id.erase()).is_none(),
            "a task node must use an image as at most one input attachment",
        );

        let input_attachment_index = attachment_info.index;

        assert!(
            !attachments
                .input_attachments
                .values()
                .any(|info| info.index == input_attachment_index),
            "the task node already has an input attachment that uses the input attachment index \
            `{input_attachment_index}`",
        );

        assert!(
            !attachments
                .color_attachments
                .get(&id.erase())
                .is_some_and(|attachment_info2| attachment_info2 == attachment_info),
            "the task node also uses the image as a color attachment but the attachment infos \
            don't match",
        );

        assert!(
            !attachments
                .depth_stencil_attachment
                .as_ref()
                .is_some_and(|(_, attachment_info2)| attachment_info2 == attachment_info),
            "the task node also uses the image as a depth/stencil attachment but the attachment \
            infos dont' match",
        );

        let resource_info = self.resources.get(id.erase()).unwrap();
        let format = if attachment_info.format == Format::UNDEFINED {
            resource_info.format
        } else {
            attachment_info.format
        };

        // Clear operations (whether using `AttachmentLoadOp::Clear` or the `clear_attachments`
        // command) are attachment writes.
        if attachment_info.clear {
            if format.aspects().intersects(COLOR_ASPECTS) {
                access_types |= AccessTypes::COLOR_ATTACHMENT_WRITE;
            } else if format.aspects().intersects(DEPTH_STENCIL_ASPECTS) {
                access_types |= AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE;
            } else {
                unreachable!();
            }
        }

        attachments.input_attachments.insert(
            id.erase(),
            AttachmentInfo {
                format,
                _ne: crate::NE,
                ..*attachment_info
            },
        );

        self.image_access(id, access_types, layout_type)
    }

    /// Adds a color attachment to the task node.
    ///
    /// # Panics
    ///
    /// - Panics if the framebuffer wasn't set beforehand.
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_types` contains any access type that's not a color attachment access
    ///   type.
    /// - Panics if a color attachment using `id` was already added.
    /// - Panics if a color attachment using the `attachment_info.index` was already added.
    /// - Panics if an input attachment using `id` was added but the attachment infos don't match.
    /// - Panics if `attachment_info.format` is not a color format.
    /// - Panics if an access for `id` was already added and its image layout doesn't equal
    ///   `access_types.image_layout(layout_type)`.
    #[track_caller]
    pub fn color_attachment(
        &mut self,
        id: Id<Image>,
        mut access_types: AccessTypes,
        layout_type: ImageLayoutType,
        attachment_info: &AttachmentInfo<'_>,
    ) -> &mut Self {
        let attachments = self
            .attachments
            .as_mut()
            .expect("the framebuffer must be set before adding attachments");

        let (id, _access) = self.accesses.image_mut(self.resources, id);

        assert!(COLOR_ATTACHMENT_ACCESS_FLAGS.contains(access_types.access_mask()));

        assert!(
            attachments.color_attachments.get(&id.erase()).is_none(),
            "a task node must use an image as at most one color attachment",
        );

        let location = attachment_info.index;

        assert!(
            !attachments
                .color_attachments
                .values()
                .any(|info| info.index == location),
            "the task node already has a color attachment that uses the location `{location}`",
        );

        assert!(
            !attachments
                .input_attachments
                .get(&id.erase())
                .is_some_and(|attachment_info2| attachment_info2 == attachment_info),
            "the task node also uses the image as an input attachment but the attachment infos \
            don't match",
        );

        let resource_info = self.resources.get(id.erase()).unwrap();
        let format = if attachment_info.format == Format::UNDEFINED {
            resource_info.format
        } else {
            attachment_info.format
        };

        assert!(
            format.aspects().intersects(COLOR_ASPECTS),
            "an image can only be used as a color attachment if it has a color format",
        );

        // Clear operations (whether using `AttachmentLoadOp::Clear` or the `clear_attachments`
        // command) are attachment writes.
        if attachment_info.clear {
            access_types |= AccessTypes::COLOR_ATTACHMENT_WRITE;
        }

        attachments.color_attachments.insert(
            id.erase(),
            AttachmentInfo {
                format,
                _ne: crate::NE,
                ..*attachment_info
            },
        );

        self.image_access(id, access_types, layout_type)
    }

    /// Adds a depth/stencil attachment to the task node.
    ///
    /// # Panics
    ///
    /// - Panics if the framebuffer wasn't set beforehand.
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_types` contains any access type that's not a depth/stencil attachment
    ///   access type.
    /// - Panics if a depth/stencil attachment was already added.
    /// - Panics if an input attachment using `id` was added but the attachment infos don't match.
    /// - Panics if `attachment_info.format` is not a depth/stencil format.
    /// - Panics if an access for `id` was already added and its image layout doesn't equal
    ///   `access_types.image_layout(layout_type)`.
    #[track_caller]
    pub fn depth_stencil_attachment(
        &mut self,
        id: Id<Image>,
        mut access_types: AccessTypes,
        layout_type: ImageLayoutType,
        attachment_info: &AttachmentInfo<'_>,
    ) -> &mut Self {
        let attachments = self
            .attachments
            .as_mut()
            .expect("the framebuffer must be set before adding attachments");

        let (id, _access) = self.accesses.image_mut(self.resources, id);

        assert!(DEPTH_STENCIL_ATTACHMENT_ACCESS_FLAGS.contains(access_types.access_mask()));

        assert!(
            attachments.depth_stencil_attachment.is_none(),
            "a task node must have at most one depth/stencil attachment",
        );

        assert!(
            !attachments
                .input_attachments
                .get(&id.erase())
                .is_some_and(|attachment_info2| attachment_info2 == attachment_info),
            "the task node also uses the image as an input attachment but the attachment infos \
            don't match",
        );

        let resource_info = self.resources.get(id.erase()).unwrap();
        let format = if attachment_info.format == Format::UNDEFINED {
            resource_info.format
        } else {
            attachment_info.format
        };

        assert!(
            format.aspects().intersects(DEPTH_STENCIL_ASPECTS),
            "an image can only be used as a depth/stencil attachment if it has a depth/stencil \
            format",
        );

        // Clear operations (whether using `AttachmentLoadOp::Clear` or the `clear_attachments`
        // command) are attachment writes.
        if attachment_info.clear {
            access_types |= AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        attachments.depth_stencil_attachment = Some((
            id.erase(),
            AttachmentInfo {
                format,
                _ne: crate::NE,
                ..*attachment_info
            },
        ));

        self.image_access(id, access_types, layout_type)
    }

    /// Finishes building the task node and returns the ID of the built node.
    #[inline]
    pub fn build(&mut self) -> NodeId {
        self.id
    }
}

/// Parameters describing a rendering attachment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttachmentInfo<'a> {
    /// The input attachment index or location to use.
    ///
    /// For input attachments, this corresponds to the input attachment index you read from the
    /// shader. For color attachments, this corresponds to the location you read or write from the
    /// shader. For depth/stencil attachments, this is ignored.
    ///
    /// The default value is `0`.
    pub index: u32,

    /// If set to `true`, the attachment is cleared before the task is executed. The clear value is
    /// set by the task in [`Task::clear_values`].
    ///
    /// The default value is `false`.
    pub clear: bool,

    /// The format of the attachment.
    ///
    /// If left as the default, the format of the image is used.
    ///
    /// If this is set to a format that is different from the image, the image must be created with
    /// the `MUTABLE_FORMAT` flag.
    ///
    /// On [portability subset] devices, if `format` does not have the same number of components
    /// and bits per component as the parent image's format, the
    /// [`image_view_format_reinterpretation`] feature must be enabled on the device.
    ///
    /// The default value is `Format::UNDEFINED`.
    ///
    /// [portability subset]: vulkano::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`image_view_format_reinterpretation`]: vulkano::device::DeviceFeatures::image_view_format_reinterpretation
    pub format: Format,

    /// How to map components of each pixel.
    ///
    /// On [portability subset] devices, if `component_mapping` is not the identity mapping, the
    /// [`image_view_format_swizzle`] feature must be enabled on the device.
    ///
    /// The default value is [`ComponentMapping::identity()`].
    ///
    /// [portability subset]: vulkano::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`image_view_format_swizzle`]: crate::device::DeviceFeatures::image_view_format_swizzle
    pub component_mapping: ComponentMapping,

    /// The mip level to render to.
    ///
    /// The default value is `0`.
    pub mip_level: u32,

    /// The base array layer to render to.
    ///
    /// The default value is `0`.
    pub base_array_layer: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AttachmentInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AttachmentInfo<'_> {
    /// Returns a default `AttachmentInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            index: 0,
            clear: false,
            format: Format::UNDEFINED,
            component_mapping: ComponentMapping::identity(),
            mip_level: 0,
            base_array_layer: 0,
            _ne: crate::NE,
        }
    }
}

/// A [`TaskGraph`] that has been compiled into an executable form.
pub struct ExecutableTaskGraph<W: ?Sized> {
    graph: ManuallyDrop<TaskGraph<W>>,
    flight_id: Id<Flight>,
    use_swapchain_maintenance1: bool,
    instructions: Vec<Instruction>,
    submissions: Vec<Submission>,
    barriers: Vec<MemoryBarrier>,
    render_passes: RefCell<Vec<RenderPassState>>,
    clear_attachments: Vec<Id>,
    semaphores: RefCell<Vec<Semaphore>>,
    swapchains: SmallVec<[Id<Swapchain>; 1]>,
    present_queue: Option<Arc<Queue>>,
    last_accesses: Vec<ResourceAccess>,
    last_frame: Cell<Option<u64>>,
    drop_graph: bool,
}

// FIXME: Initial queue family ownership transfers
#[derive(Debug)]
struct Submission {
    queue: Arc<Queue>,
    initial_barrier_range: Range<BarrierIndex>,
    instruction_range: Range<InstructionIndex>,
}

type InstructionIndex = usize;

#[derive(Clone, Debug)]
enum Instruction {
    WaitAcquire {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    WaitSemaphore {
        semaphore_index: SemaphoreIndex,
        stage_mask: PipelineStages,
    },
    ExecuteTask {
        node_index: NodeIndex,
    },
    PipelineBarrier {
        barrier_range: Range<BarrierIndex>,
    },
    // TODO:
    // SetEvent {
    //     event_index: EventIndex,
    //     barrier_range: Range<BarrierIndex>,
    // },
    // WaitEvent {
    //     event_index: EventIndex,
    //     barrier_range: Range<BarrierIndex>,
    // },
    BeginRenderPass {
        render_pass_index: RenderPassIndex,
    },
    NextSubpass,
    EndRenderPass,
    ClearAttachments {
        node_index: NodeIndex,
        render_pass_index: RenderPassIndex,
        clear_attachment_range: Range<ClearAttachmentIndex>,
    },
    SignalSemaphore {
        semaphore_index: SemaphoreIndex,
        stage_mask: PipelineStages,
    },
    SignalPrePresent {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    WaitPrePresent {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    SignalPresent {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    FlushSubmit,
    Submit,
}

type RenderPassIndex = usize;

type SemaphoreIndex = usize;

type BarrierIndex = u32;

type ClearAttachmentIndex = usize;

#[derive(Clone, Debug)]
struct MemoryBarrier {
    src_stage_mask: PipelineStages,
    src_access_mask: AccessFlags,
    dst_stage_mask: PipelineStages,
    dst_access_mask: AccessFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    resource: Id,
}

#[derive(Debug)]
struct RenderPassState {
    render_pass: Arc<RenderPass>,
    attachments: LinearMap<Id, AttachmentState>,
    framebuffers: Vec<Arc<Framebuffer>>,
    clear_node_indices: Vec<NodeIndex>,
}

#[derive(Debug)]
struct AttachmentState {
    index: u32,
    format: Format,
    component_mapping: ComponentMapping,
    mip_level: u32,
    base_array_layer: u32,
}

impl<W: ?Sized> ExecutableTaskGraph<W> {
    /// Returns a reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node(&self, id: NodeId) -> Result<&TaskNode<W>> {
        self.graph.task_node(id)
    }

    /// Returns a mutable reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node_mut(&mut self, id: NodeId) -> Result<&mut TaskNode<W>> {
        self.graph.task_node_mut(id)
    }

    /// Returns an iterator over all [`TaskNode`]s.
    #[inline]
    pub fn task_nodes(&self) -> TaskNodes<'_, W> {
        self.graph.task_nodes()
    }

    /// Returns an iterator over all [`TaskNode`]s that allows you to mutate them.
    #[inline]
    pub fn task_nodes_mut(&mut self) -> TaskNodesMut<'_, W> {
        self.graph.task_nodes_mut()
    }

    /// Returns the flight ID that the task graph was compiled with.
    #[inline]
    pub fn flight_id(&self) -> Id<Flight> {
        self.flight_id
    }
}

impl<W: ?Sized> fmt::Debug for ExecutableTaskGraph<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ExecutableTaskGraph");

        debug
            .field("graph", &self.graph)
            .field("flight_id", &self.flight_id)
            .field("instructions", &self.instructions)
            .field("submissions", &self.submissions)
            .field("barriers", &self.barriers)
            .field("render_passes", &self.render_passes)
            .field("clear_attachments", &self.clear_attachments)
            .field("semaphores", &self.semaphores)
            .field("swapchains", &self.swapchains)
            .field("present_queue", &self.present_queue)
            .finish_non_exhaustive()
    }
}

impl<W: ?Sized> Drop for ExecutableTaskGraph<W> {
    fn drop(&mut self) {
        if let Some(last_frame) = self.last_frame.get() {
            let resources = &self.graph.resources.physical_resources;
            let mut batch = resources.create_deferred_batch();

            for semaphore in self.semaphores.get_mut().drain(..) {
                batch.destroy_object(semaphore);
            }

            for render_pass_state in self.render_passes.get_mut() {
                for framebuffer in render_pass_state.framebuffers.drain(..) {
                    batch.destroy_object(framebuffer);
                }
            }

            // SAFETY: We only defer the destruction of objects that are graph-local and
            // `last_frame` is the last frame that the graph executed.
            unsafe { batch.enqueue_with_frames(iter::once((self.flight_id, last_frame))) };
        }

        if self.drop_graph {
            // SAFETY: We are being dropped which ensures that the graph cannot be used again.
            unsafe { ManuallyDrop::drop(&mut self.graph) };
        }
    }
}

unsafe impl<W: ?Sized> DeviceOwned for ExecutableTaskGraph<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.graph.device()
    }
}

/// An iterator over all [`TaskNode`]s within a [`TaskGraph`].
///
/// This type is created by the [`task_nodes`] method on [`TaskGraph`].
///
/// [`task_nodes`]: TaskGraph::task_nodes
pub struct TaskNodes<'a, W: ?Sized> {
    inner: slotmap::Iter<'a, NodeId, Node<W>>,
}

impl<W: ?Sized> fmt::Debug for TaskNodes<'_, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskNodes").finish_non_exhaustive()
    }
}

impl<'a, W: ?Sized> Iterator for TaskNodes<'a, W> {
    type Item = &'a TaskNode<W>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next()?;

            if let NodeInner::Task(task_node) = &node.inner {
                break Some(task_node);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<W: ?Sized> DoubleEndedIterator for TaskNodes<'_, W> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next_back()?;

            if let NodeInner::Task(task_node) = &node.inner {
                break Some(task_node);
            }
        }
    }
}

impl<W: ?Sized> FusedIterator for TaskNodes<'_, W> {}

/// An iterator over all [`TaskNode`]s within a [`TaskGraph`] that allows you to mutate them.
///
/// This type is created by the [`task_nodes_mut`] method on [`TaskGraph`].
///
/// [`task_nodes_mut`]: TaskGraph::task_nodes_mut
pub struct TaskNodesMut<'a, W: ?Sized> {
    inner: slotmap::IterMut<'a, NodeId, Node<W>>,
}

impl<W: ?Sized> fmt::Debug for TaskNodesMut<'_, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskNodesMut").finish_non_exhaustive()
    }
}

impl<'a, W: ?Sized> Iterator for TaskNodesMut<'a, W> {
    type Item = &'a mut TaskNode<W>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next()?;

            if let NodeInner::Task(task_node) = &mut node.inner {
                break Some(task_node);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<W: ?Sized> DoubleEndedIterator for TaskNodesMut<'_, W> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next_back()?;

            if let NodeInner::Task(task_node) = &mut node.inner {
                break Some(task_node);
            }
        }
    }
}

impl<W: ?Sized> FusedIterator for TaskNodesMut<'_, W> {}

type Result<T = (), E = TaskGraphError> = ::std::result::Result<T, E>;

/// Error that can happen when doing operations on a [`TaskGraph`].
#[derive(Debug, PartialEq, Eq)]
pub enum TaskGraphError {
    InvalidNode,
    InvalidNodeType,
    InvalidEdge,
    DuplicateEdge,
}

impl fmt::Display for TaskGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::InvalidNode => "a node with the given ID does not exist",
            Self::InvalidNodeType => {
                "the node with the given ID has a type that is incompatible with the operation"
            }
            Self::InvalidEdge => "an edge between the given nodes does not exist",
            Self::DuplicateEdge => "an edge between the given nodes already exists",
        };

        f.write_str(msg)
    }
}

impl Error for TaskGraphError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_queues;
    use std::marker::PhantomData;

    #[test]
    fn basic_usage1() {
        let (resources, _) = test_queues!();
        let mut graph = TaskGraph::<()>::new(&resources);

        let x = graph
            .create_task_node("X", QueueFamilyType::Graphics, PhantomData)
            .build();
        let y = graph
            .create_task_node("Y", QueueFamilyType::Graphics, PhantomData)
            .build();

        graph.add_edge(x, y).unwrap();
        assert!(graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(graph.nodes.node(y).unwrap().in_edges.contains(&x.index()));
        assert_eq!(graph.add_edge(x, y), Err(TaskGraphError::DuplicateEdge));

        graph.remove_edge(x, y).unwrap();
        assert!(!graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(!graph.nodes.node(y).unwrap().in_edges.contains(&x.index()));

        assert_eq!(graph.remove_edge(x, y), Err(TaskGraphError::InvalidEdge));

        graph.add_edge(y, x).unwrap();
        assert!(graph.nodes.node(y).unwrap().out_edges.contains(&x.index()));
        assert!(graph.nodes.node(x).unwrap().in_edges.contains(&y.index()));
        assert_eq!(graph.add_edge(y, x), Err(TaskGraphError::DuplicateEdge));

        graph.remove_edge(y, x).unwrap();
        assert!(!graph.nodes.node(y).unwrap().out_edges.contains(&x.index()));
        assert!(!graph.nodes.node(x).unwrap().in_edges.contains(&y.index()));

        assert_eq!(graph.remove_edge(y, x), Err(TaskGraphError::InvalidEdge));
    }

    #[test]
    fn basic_usage2() {
        let (resources, _) = test_queues!();
        let mut graph = TaskGraph::<()>::new(&resources);

        let x = graph
            .create_task_node("X", QueueFamilyType::Graphics, PhantomData)
            .build();
        let y = graph
            .create_task_node("Y", QueueFamilyType::Graphics, PhantomData)
            .build();
        let z = graph
            .create_task_node("Z", QueueFamilyType::Graphics, PhantomData)
            .build();

        assert!(graph.task_node(x).is_ok());
        assert!(graph.task_node(y).is_ok());
        assert!(graph.task_node(z).is_ok());
        assert!(graph.task_node_mut(x).is_ok());
        assert!(graph.task_node_mut(y).is_ok());
        assert!(graph.task_node_mut(z).is_ok());

        graph.add_edge(x, y).unwrap();
        graph.add_edge(y, z).unwrap();
        assert!(graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(graph.nodes.node(z).unwrap().in_edges.contains(&y.index()));

        graph.remove_task_node(y).unwrap();
        assert!(!graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(!graph.nodes.node(z).unwrap().in_edges.contains(&y.index()));

        assert!(matches!(
            graph.remove_task_node(y),
            Err(TaskGraphError::InvalidNode),
        ));
    }

    #[test]
    fn self_referential_node() {
        let (resources, _) = test_queues!();
        let mut graph = TaskGraph::<()>::new(&resources);

        let x = graph
            .create_task_node("X", QueueFamilyType::Graphics, PhantomData)
            .build();

        assert_eq!(graph.add_edge(x, x), Err(TaskGraphError::InvalidNode));
    }
}
