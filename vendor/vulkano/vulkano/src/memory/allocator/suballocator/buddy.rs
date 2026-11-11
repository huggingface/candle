use super::{
    AllocationType, Region, Suballocation, SuballocationNode, SuballocationType, Suballocator,
    SuballocatorError,
};
use crate::{
    memory::{
        allocator::{align_up, array_vec::ArrayVec, AllocationHandle, DeviceLayout},
        is_aligned, DeviceAlignment,
    },
    DeviceSize,
};
use std::{
    cmp,
    collections::VecDeque,
    fmt::{Debug, Formatter, Result as FmtResult},
    hint,
    iter::FusedIterator,
    marker::PhantomData,
    num::NonZero,
    ptr::NonNull,
};

/// A [suballocator] whose structure forms a binary tree of power-of-two-sized suballocations.
///
/// You generally shouldn't use this allocator. It has some of the worst [fragmentation] overall,
/// worse than both [`TlsfAllocator`] and [`FreeListAllocator`]. This is because it can only
/// allocate sizes that are powers of two. For example, if you need an allocation size of 70MiB,
/// you would use a whole 128MiB instead, wasting 45% of the memory. It is recommended that you use
/// `TlsfAllocator` instead. However, this allocator does have its niche: when all your allocations
/// are dynamically- but power-of-two-sized, then there is no [internal fragmentation], and since
/// this algorithm somewhat reduces [external fragmentation], the overall fragmentation can beat
/// `TlsfAllocator` and `FreeListAllocator`. The algorithm is also fast in practice. Please test
/// and profile to make sure that this is actually better.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// Say you have a [region] of size 256MiB, and you want to allocate 14MiB. Assuming there are no
/// existing allocations, the `BuddyAllocator` would split the 256MiB root *node* into two 128MiB
/// nodes. These two nodes are called *buddies*. The allocator would then proceed to split the left
/// node recursively until it wouldn't be able to fit the allocation anymore. In this example, that
/// would happen after 4 splits and end up with a node size of 16MiB. Since the allocation
/// requested was 14MiB, 2MiB would become internal fragmentation and be unusable for the lifetime
/// of the allocation. When an allocation is freed, this process is done backwards, checking if the
/// buddy of each node on the way up is free, and if so, they are coalesced.
///
/// Each possible node size has an *order*, with the smallest node size being of order 0 and the
/// largest of the highest order. With this notion, node sizes are proportional to 2<sup>*n*</sup>
/// where *n* is the order. The highest order is determined from the size of the region and a
/// constant minimum node size, which we chose to be 16B: log(*region&nbsp;size*&nbsp;/&nbsp;16),
/// or equivalently, log(*region&nbsp;size*)&nbsp;-&nbsp;4 (assuming
/// *region&nbsp;size*&nbsp;&ge;&nbsp;16).
///
/// It's safe to say that this algorithm works best if you have some level of control over your
/// allocation sizes, so that you don't end up allocating twice as much memory. An example of this
/// would be when you need to allocate regions for other allocators, such as for an allocation pool
/// or the [`BumpAllocator`].
///
/// # Efficiency
///
/// The time complexity of both allocation and deallocation is *O*(log(*n*)). Resetting is *O*(1).
///
/// [suballocator]: Suballocator
/// [fragmentation]: super::super#fragmentation
/// [`TlsfAllocator`]: super::TlsfAllocator
/// [`FreeListAllocator`]: super::FreeListAllocator
/// [internal fragmentation]: super::super#internal-fragmentation
/// [external fragmentation]: super::super#external-fragmentation
/// [the `Suballocator` implementation]: Self#impl-Suballocator-for-BuddyAllocator
/// [region]: Suballocator#regions
/// [`BumpAllocator`]: super::BumpAllocator
#[derive(Debug)]
pub struct BuddyAllocator {
    // Total memory remaining in the region.
    free_size: DeviceSize,
    suballocations: SuballocationTree,
}

impl BuddyAllocator {
    pub(super) const MIN_NODE_SIZE: DeviceSize = 16;

    /// Arbitrary maximum number of orders, used to avoid a 2D `Vec`. Together with a minimum node
    /// size of 16, this is enough for a 32GiB region.
    const MAX_ORDERS: usize = 32;
}

unsafe impl Suballocator for BuddyAllocator {
    type Suballocations<'a> = Suballocations<'a>;

    /// Creates a new `BuddyAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.size` is not a power of two.
    /// - Panics if `region.size` is not in the range \[16B,&nbsp;32GiB\].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        const EMPTY_FREE_LIST: Vec<NonNull<SuballocationTreeNode>> = Vec::new();

        assert!(region.size().is_power_of_two());
        assert!(region.size() >= BuddyAllocator::MIN_NODE_SIZE);

        // The division can't discard any bits because `region.size()` is a power of two.
        let max_order = (region.size() / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;

        assert!(max_order < BuddyAllocator::MAX_ORDERS);

        // This can't overflow because `max_order` is less than `BuddyAllocator::MAX_ORDERS`.
        let orders = max_order + 1;

        let mut suballocations = SuballocationTree {
            region,
            root_ptr: NonNull::dangling(),
            head_ptr: NonNull::dangling(),
            tail_ptr: NonNull::dangling(),
            len: 0,
            free_list: ArrayVec::new(orders, [EMPTY_FREE_LIST; BuddyAllocator::MAX_ORDERS]),
            node_allocator: slabbin::SlabAllocator::new(32),
        };

        // This can't discard any bits because `max_order` is confined to the range
        // [0, BuddyAllocator::MAX_ORDERS).
        unsafe { suballocations.init(max_order as u8) };

        BuddyAllocator {
            free_size: region.size(),
            suballocations,
        }
    }

    #[inline]
    fn allocate_buffer(
        &mut self,
        layout: DeviceLayout,
    ) -> Result<Suballocation, SuballocatorError> {
        self.allocate_inner(layout, AllocationType::Linear, DeviceAlignment::MIN)
    }

    #[inline]
    fn allocate(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        self.allocate_inner(layout, allocation_type, buffer_image_granularity)
    }

    #[inline]
    unsafe fn deallocate(&mut self, suballocation: Suballocation) {
        let node_ptr = suballocation
            .handle
            .as_ptr()
            .cast::<SuballocationTreeNode>();

        // SAFETY: The caller must guarantee that `suballocation` refers to a currently allocated
        // allocation of `self`, which means that `node_ptr` is the same one we gave out on
        // allocation, making it a valid pointer.
        let node_ptr = unsafe { NonNull::new_unchecked(node_ptr) };

        debug_assert!(self.suballocations.node_allocator.contains(node_ptr));

        // SAFETY: Same as the previous.
        let order = unsafe { self.suballocations.deallocate(node_ptr) };

        // This can't discard any bits because `order` is confined to the range
        // [0, BuddyAllocator::MAX_ORDERS).
        let size = BuddyAllocator::MIN_NODE_SIZE << order;

        // The sizes of suballocations allocated by `self` are constrained by that of its region,
        // so they can't possibly overflow when added up.
        self.free_size += size;
    }

    fn reset(&mut self) {
        self.suballocations.reset();
        self.free_size = self.region().size();
    }

    /// Returns the total amount of free space left in the [region] that is available to the
    /// allocator, which means that [internal fragmentation] is excluded.
    ///
    /// [region]: Suballocator#regions
    /// [internal fragmentation]: super::super#internal-fragmentation
    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size
    }

    #[inline]
    fn suballocations(&self) -> Self::Suballocations<'_> {
        self.suballocations.iter()
    }
}

impl BuddyAllocator {
    #[inline(always)]
    fn allocate_inner(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        /// Returns the largest power of two smaller or equal to the input, or zero if the input is
        /// zero.
        fn prev_power_of_two(val: DeviceSize) -> DeviceSize {
            const MAX_POWER_OF_TWO: DeviceSize = DeviceAlignment::MAX.as_devicesize();

            if let Some(val) = NonZero::new(val) {
                // This can't overflow because `val` is non-zero, which means it has fewer leading
                // zeroes than the total number of bits.
                MAX_POWER_OF_TWO >> val.leading_zeros()
            } else {
                0
            }
        }

        let mut size = layout.size();
        let mut alignment = layout.alignment();

        if buffer_image_granularity != DeviceAlignment::MIN {
            debug_assert!(is_aligned(self.region().offset(), buffer_image_granularity));

            if allocation_type == AllocationType::Unknown
                || allocation_type == AllocationType::NonLinear
            {
                // This can't overflow because `DeviceLayout` guarantees that `size` doesn't exceed
                // `DeviceLayout::MAX_SIZE`.
                size = align_up(size, buffer_image_granularity);
                alignment = cmp::max(alignment, buffer_image_granularity);
            }
        }

        // `DeviceLayout` guarantees that its size does not exceed `DeviceLayout::MAX_SIZE`,
        // which means it can't overflow when rounded up to the next power of two.
        let size = cmp::max(size, BuddyAllocator::MIN_NODE_SIZE).next_power_of_two();

        // SAFETY: `size` is a power of two.
        if let Some((node_ptr, offset, order)) = unsafe {
            self.suballocations
                .allocate(size, alignment, allocation_type)
        } {
            // This can't overflow because `order` is confined to the range
            // [0, BuddyAllocator::MAX_ORDERS).
            let size = BuddyAllocator::MIN_NODE_SIZE << order;

            // This can't overflow because suballocation sizes in the free-lists are constrained by
            // the remaining size of the region.
            self.free_size -= size;

            Ok(Suballocation {
                offset,
                allocation_type,
                size: layout.size(),
                handle: AllocationHandle::from_ptr(node_ptr.as_ptr().cast()),
            })
        } else {
            if prev_power_of_two(self.free_size()) >= layout.size() {
                // A node large enough could be formed if the region wasn't so fragmented.
                Err(SuballocatorError::FragmentedRegion)
            } else {
                Err(SuballocatorError::OutOfRegionMemory)
            }
        }
    }

    #[inline]
    fn region(&self) -> &Region {
        &self.suballocations.region
    }
}

struct SuballocationTree {
    region: Region,
    root_ptr: NonNull<SuballocationTreeNode>,
    head_ptr: NonNull<SuballocationTreeNode>,
    tail_ptr: NonNull<SuballocationTreeNode>,
    len: usize,
    // Every order has its own free-list. Each free-list is sorted by offset because we want to
    // find the first-fit as this strategy minimizes external fragmentation.
    free_list: ArrayVec<Vec<NonNull<SuballocationTreeNode>>, { BuddyAllocator::MAX_ORDERS }>,
    node_allocator: slabbin::SlabAllocator<SuballocationTreeNode>,
}

struct SuballocationTreeNode {
    parent_ptr: NonNull<SuballocationTreeNode>,
    ty: NodeType,
}

/// A manual enum because variant types aren't a thing.
#[repr(C)]
union NodeType {
    leaf: Leaf,
    branch: Branch,
}

// These fields are carefully ordered to minimize the size of the struct.
#[derive(Clone, Copy)]
#[repr(C)]
struct Leaf {
    tag: NodeTag,
    allocation_type: SuballocationType,
    order: u8,
    /// The offset minus `region.offset()` all divided by `BuddyAllocator::MIN_NODE_SIZE`. This
    /// ensures that the offset fits in a `u32`, otherwise `Self` would be wider by a `DeviceSize`.
    compressed_offset: u32,
    /// The previous leaf node.
    prev_ptr: NonNull<SuballocationTreeNode>,
    /// The next leaf node.
    next_ptr: NonNull<SuballocationTreeNode>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Branch {
    tag: NodeTag,
    /// The left subtree.
    left_ptr: NonNull<SuballocationTreeNode>,
    /// The right subtree.
    right_ptr: NonNull<SuballocationTreeNode>,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum NodeTag {
    Leaf,
    Branch,
}

unsafe impl Send for SuballocationTree {}
unsafe impl Sync for SuballocationTree {}

impl SuballocationTree {
    unsafe fn init(&mut self, max_order: u8) {
        let head_ptr = self.node_allocator.allocate();
        let root_ptr = self.node_allocator.allocate();
        let tail_ptr = self.node_allocator.allocate();

        let head = SuballocationTreeNode {
            parent_ptr: NonNull::dangling(),
            ty: NodeType::new_leaf(Leaf {
                tag: NodeTag::Leaf,
                prev_ptr: NonNull::dangling(),
                next_ptr: root_ptr,
                compressed_offset: 0,
                order: 0,
                allocation_type: SuballocationType::Unknown,
            }),
        };
        unsafe { head_ptr.write(head) };

        // The root node has the lowest offset and highest order, so it's the whole region.
        let root = SuballocationTreeNode {
            parent_ptr: NonNull::dangling(),
            ty: NodeType::new_leaf(Leaf {
                tag: NodeTag::Leaf,
                prev_ptr: head_ptr,
                next_ptr: tail_ptr,
                compressed_offset: 0,
                order: max_order,
                allocation_type: SuballocationType::Free,
            }),
        };
        unsafe { root_ptr.write(root) };

        let tail = SuballocationTreeNode {
            parent_ptr: NonNull::dangling(),
            ty: NodeType::new_leaf(Leaf {
                tag: NodeTag::Leaf,
                prev_ptr: root_ptr,
                next_ptr: NonNull::dangling(),
                // The division can't discard any bits because the region size is a power of two.
                // The cast can't discard any bits because the region size is at most `1 << 35`,
                // and `(1 << 35) / BuddyAllocator::MIN_NODE_SIZE` is `1 << 31`.
                compressed_offset: (self.region.size() / BuddyAllocator::MIN_NODE_SIZE) as u32,
                order: 0,
                allocation_type: SuballocationType::Unknown,
            }),
        };
        unsafe { tail_ptr.write(tail) };

        self.root_ptr = root_ptr;
        self.head_ptr = head_ptr;
        self.tail_ptr = tail_ptr;
        self.len = 1;
        self.free_list[usize::from(max_order)].push(root_ptr);
    }

    #[inline]
    unsafe fn allocate(
        &mut self,
        size: DeviceSize,
        alignment: DeviceAlignment,
        allocation_type: AllocationType,
    ) -> Option<(NonNull<SuballocationTreeNode>, DeviceSize, usize)> {
        debug_assert!(size.is_power_of_two());

        let min_order = (size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;

        // Start searching at the lowest possible order going up.
        let mut iter = self.free_list.iter_mut().enumerate().skip(min_order);
        let (order, mut node_ptr, node_ty, offset) = 'outer: loop {
            let (order, free_list) = iter.next()?;

            for (index, &(mut node_ptr)) in free_list.iter().enumerate() {
                let node = unsafe { node_ptr.as_mut() };

                // SAFETY: The free-lists only contain leaf nodes.
                let node_ty = unsafe { node.ty.leaf_unchecked_mut() };

                let offset = node_ty.offset(self.region.offset());

                if is_aligned(offset, alignment) {
                    free_list.remove(index);
                    node_ty.allocation_type = allocation_type.into();
                    break 'outer (order, node_ptr, node_ty, offset);
                }
            }
        };

        let mut prev_ptr = node_ty.prev_ptr;
        let mut next_ptr = node_ty.next_ptr;
        let compressed_offset = node_ty.compressed_offset;

        // Go in the opposite direction, splitting nodes from higher orders. The lowest order
        // doesn't need any splitting.
        for (order, free_list) in self
            .free_list
            .iter_mut()
            .enumerate()
            .skip(min_order)
            .take(order - min_order)
            .rev()
        {
            // This can't discard any bits because `order` is confined to the range
            // [0, BuddyAllocator::MAX_ORDERS).
            let size = BuddyAllocator::MIN_NODE_SIZE << order;

            let left_ptr = self.node_allocator.allocate();
            let right_ptr = self.node_allocator.allocate();

            let left = SuballocationTreeNode {
                parent_ptr: node_ptr,
                ty: NodeType::new_leaf(Leaf {
                    tag: NodeTag::Leaf,
                    prev_ptr,
                    next_ptr: right_ptr,
                    compressed_offset,
                    // This can't discard any bits because `order` is confined to the range
                    // [0, BuddyAllocator::MAX_ORDERS).
                    order: order as u8,
                    allocation_type: allocation_type.into(),
                }),
            };
            unsafe { left_ptr.write(left) };

            let right = SuballocationTreeNode {
                parent_ptr: node_ptr,
                ty: NodeType::new_leaf(Leaf {
                    tag: NodeTag::Leaf,
                    prev_ptr: left_ptr,
                    next_ptr,
                    // The division can't discard any bits because sizes are aligned to
                    // `BuddyAllocator::MIN_NODE_SIZE`. The cast can't discard any bits because the
                    // region size is at most `1 << 35`, and
                    // `(1 << 35) / BuddyAllocator::MIN_NODE_SIZE` is `1 << 31`. The addition can't
                    // overflow because suballocations are bounded by the region.
                    compressed_offset: compressed_offset
                        + (size / BuddyAllocator::MIN_NODE_SIZE) as u32,
                    // This can't discard any bits because `order` is confined to the range
                    // [0, BuddyAllocator::MAX_ORDERS).
                    order: order as u8,
                    allocation_type: SuballocationType::Free,
                }),
            };
            unsafe { right_ptr.write(right) };

            let next = unsafe { next_ptr.as_mut() };

            // SAFETY: The list of leaf nodes only contains leaf nodes.
            unsafe { next.ty.leaf_unchecked_mut() }.prev_ptr = right_ptr;

            unsafe { node_ptr.as_mut() }.ty = NodeType::new_branch(Branch {
                tag: NodeTag::Branch,
                left_ptr,
                right_ptr,
            });

            self.len += 1;

            node_ptr = left_ptr;
            next_ptr = right_ptr;

            unsafe { add_to_free_list(free_list, right_ptr, compressed_offset) };

            // Repeat splitting for the left child if required in the next loop turn.
        }

        let prev = unsafe { prev_ptr.as_mut() };

        // SAFETY: The list of leaf nodes only contains leaf nodes.
        unsafe { prev.ty.leaf_unchecked_mut() }.next_ptr = node_ptr;

        Some((node_ptr, offset, min_order))
    }

    #[inline]
    unsafe fn deallocate(&mut self, mut node_ptr: NonNull<SuballocationTreeNode>) -> usize {
        // SAFETY: Enforced by the caller.
        let node = unsafe { node_ptr.as_mut() };

        // SAFETY: The caller must ensure that `node_ptr` is a leaf node.
        let node_ty = unsafe { node.ty.leaf_unchecked_mut() };

        let mut parent_ptr = node.parent_ptr;
        let mut compressed_offset = node_ty.compressed_offset;
        let mut node_order = usize::from(node_ty.order);
        let mut prev_ptr = node_ty.prev_ptr;
        let mut next_ptr = node_ty.next_ptr;
        debug_assert_ne!(node_ty.allocation_type, SuballocationType::Free);
        node_ty.allocation_type = SuballocationType::Free;

        let min_order = node_order;

        // `- 1` because we must make sure not to dereference the parent of the highest-order node
        // as that's a dangling pointer. The `- 1` can't overflow because we always have at least
        // one node (the root node). Also, `min_order` is confined to the range
        // [0, self.free_list.len()), so the second subtraction can't overflow either.
        let max_orders = self.free_list.len() - 1 - min_order;

        debug_assert!(!self.free_list[node_order].contains(&node_ptr));

        // Try to coalesce nodes while incrementing the order.
        for (order, free_list) in self
            .free_list
            .iter_mut()
            .enumerate()
            .skip(min_order)
            .take(max_orders)
        {
            // `+ 1` because we need to free the parent. This can't overflow because `order` is
            // confined to the range [0, BuddyAllocator::MAX_ORDERS - 1).
            node_order = order + 1;

            let parent = unsafe { parent_ptr.as_mut() };

            // SAFETY: The parent of a node is always a branch node.
            let parent_ty = unsafe { parent.ty.branch_unchecked() };

            let (buddy_ptr, is_left) = if node_ptr == parent_ty.left_ptr {
                (parent_ty.right_ptr, true)
            } else if node_ptr == parent_ty.right_ptr {
                (parent_ty.left_ptr, false)
            } else {
                // SAFETY: The parent of a node always has the node as a child.
                unsafe { hint::unreachable_unchecked() }
            };

            let buddy = unsafe { buddy_ptr.as_ref() };

            // If the buddy isn't a free node, we can't coalesce, so we add the node to the
            // free-list.
            if !buddy.ty.is_free() {
                let prev = unsafe { prev_ptr.as_mut() };
                let prev_ty = unsafe { prev.ty.leaf_unchecked_mut() };
                prev_ty.next_ptr = node_ptr;

                let next = unsafe { next_ptr.as_mut() };
                let next_ty = unsafe { next.ty.leaf_unchecked_mut() };
                next_ty.prev_ptr = node_ptr;

                unsafe { add_to_free_list(free_list, node_ptr, compressed_offset) };

                return min_order;
            };

            // SAFETY: We checked that the buddy is a leaf node above.
            let buddy_ty = unsafe { buddy.ty.leaf_unchecked() };

            let buddy_compressed_offset = buddy_ty.compressed_offset;

            let Ok(index) = free_list.binary_search_by_key(&buddy_compressed_offset, |&ptr| {
                let node = unsafe { ptr.as_ref() };

                // SAFETY: The free-lists only contain leaf nodes.
                unsafe { node.ty.leaf_unchecked() }.compressed_offset
            }) else {
                // SAFETY: We checked that the buddy is a free node above, which means it must be
                // in the free-list.
                unsafe { hint::unreachable_unchecked() }
            };

            free_list.remove(index);

            let node = unsafe { node_ptr.as_ref() };

            // SAFETY: We start out with `node_ptr` being a leaf node, and upon iteration, we set
            // it to a leaf node.
            let node_ty = unsafe { node.ty.leaf_unchecked() };

            (prev_ptr, next_ptr) = if is_left {
                (node_ty.prev_ptr, buddy_ty.next_ptr)
            } else {
                (buddy_ty.prev_ptr, node_ty.next_ptr)
            };

            compressed_offset = cmp::min(compressed_offset, buddy_compressed_offset);

            parent.ty = NodeType::new_leaf(Leaf {
                tag: NodeTag::Leaf,
                prev_ptr,
                next_ptr,
                compressed_offset,
                // The addition can't overflow and the cast can't discard any bits because `order`
                // is confined to the range [0, BuddyAllocator::MAX_ORDERS - 1).
                order: (order + 1) as u8,
                allocation_type: SuballocationType::Free,
            });

            unsafe { self.node_allocator.deallocate(node_ptr) };
            unsafe { self.node_allocator.deallocate(buddy_ptr) };

            self.len -= 1;

            node_ptr = parent_ptr;
            parent_ptr = parent.parent_ptr;
        }

        let prev = unsafe { prev_ptr.as_mut() };
        let prev_ty = unsafe { prev.ty.leaf_unchecked_mut() };
        prev_ty.next_ptr = node_ptr;

        let next = unsafe { next_ptr.as_mut() };
        let next_ty = unsafe { next.ty.leaf_unchecked_mut() };
        next_ty.prev_ptr = node_ptr;

        let free_list = unsafe { self.free_list.get_unchecked_mut(node_order) };
        unsafe { add_to_free_list(free_list, node_ptr, compressed_offset) };

        min_order
    }

    fn reset(&mut self) {
        self.free_list.iter_mut().for_each(Vec::clear);
        unsafe { self.node_allocator.reset() };

        // The division can't discard any bits because the region size is a power of two. The cast
        // can't discard any bits because the region size is at most `1 << 35`, and
        // `log((1 << 35) / BuddyAllocator::MIN_NODE_SIZE)` is 31.
        let max_order = (self.region.size() / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as u8;

        unsafe { self.init(max_order) };
    }

    #[inline]
    fn iter(&self) -> Suballocations<'_> {
        let head = unsafe { self.head_ptr.as_ref() };
        let tail = unsafe { self.tail_ptr.as_ref() };
        let head_ptr = unsafe { head.ty.leaf_unchecked() }.next_ptr;
        let tail_ptr = unsafe { tail.ty.leaf_unchecked() }.prev_ptr;

        Suballocations {
            region_offset: self.region.offset(),
            head_ptr,
            tail_ptr,
            len: self.len,
            marker: PhantomData,
        }
    }
}

#[inline]
unsafe fn add_to_free_list(
    free_list: &mut Vec<NonNull<SuballocationTreeNode>>,
    node_ptr: NonNull<SuballocationTreeNode>,
    compressed_offset: u32,
) {
    let (Ok(index) | Err(index)) = free_list.binary_search_by_key(&compressed_offset, |&ptr| {
        // SAFETY: The caller must ensure that `free_list` is a free-list.
        let node = unsafe { ptr.as_ref() };

        // SAFETY: The free-lists only contain leaf nodes.
        unsafe { node.ty.leaf_unchecked() }.compressed_offset
    });
    free_list.insert(index, node_ptr);
}

impl Debug for SuballocationTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        struct Nodes<'a>(&'a SuballocationTree);

        impl Debug for Nodes<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let mut entries = Vec::with_capacity(self.0.len);
                let mut queue = VecDeque::with_capacity(BuddyAllocator::MAX_ORDERS);
                queue.push_back(self.0.root_ptr);

                // Breath-first traversal because it's easier to make out which nodes are children
                // of which nodes this way.
                while let Some(node_ptr) = queue.pop_front() {
                    let node = unsafe { node_ptr.as_ref() };

                    entries.push((node_ptr, node));

                    if let Some(branch) = node.ty.branch() {
                        queue.push_back(branch.left_ptr);
                        queue.push_back(branch.right_ptr);
                    }
                }

                f.debug_map().entries(entries).finish()
            }
        }

        f.debug_struct("SuballocationTree")
            .field("region", &self.region)
            .field("nodes", &Nodes(self))
            .field("len", &self.len)
            .field("free_list", &self.free_list)
            .finish()
    }
}

impl Debug for SuballocationTreeNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.ty.tag() {
            NodeTag::Leaf => {
                let ty = unsafe { self.ty.leaf_unchecked() };

                f.debug_struct("Leaf")
                    .field("parent_ptr", &self.parent_ptr)
                    .field("allocation_type", &ty.allocation_type)
                    .field("order", &ty.order)
                    .field("compressed_offset", &ty.compressed_offset)
                    .field("prev_ptr", &ty.prev_ptr)
                    .field("next_ptr", &ty.next_ptr)
                    .finish()
            }
            NodeTag::Branch => {
                let ty = unsafe { self.ty.branch_unchecked() };

                f.debug_struct("Branch")
                    .field("parent_ptr", &self.parent_ptr)
                    .field("left_ptr", &ty.left_ptr)
                    .field("right_ptr", &ty.right_ptr)
                    .finish()
            }
        }
    }
}

impl NodeType {
    #[inline]
    fn new_leaf(leaf: Leaf) -> Self {
        let this = Self { leaf };
        assert!(this.is_leaf());

        this
    }

    #[inline]
    fn new_branch(branch: Branch) -> Self {
        let this = Self { branch };
        assert!(this.is_branch());

        this
    }

    #[inline]
    fn leaf(&self) -> Option<&Leaf> {
        if self.is_leaf() {
            // SAFETY: We checked that the tag is that of a leaf.
            Some(unsafe { self.leaf_unchecked() })
        } else {
            None
        }
    }

    #[inline]
    unsafe fn leaf_unchecked(&self) -> &Leaf {
        debug_assert!(self.is_leaf());

        unsafe { &self.leaf }
    }

    #[inline]
    unsafe fn leaf_unchecked_mut(&mut self) -> &mut Leaf {
        debug_assert!(self.is_leaf());

        unsafe { &mut self.leaf }
    }

    #[inline]
    fn branch(&self) -> Option<&Branch> {
        if self.is_branch() {
            // SAFETY: We checked that the tag is that of a branch.
            Some(unsafe { self.branch_unchecked() })
        } else {
            None
        }
    }

    #[inline]
    unsafe fn branch_unchecked(&self) -> &Branch {
        debug_assert!(self.is_branch());

        unsafe { &self.branch }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.tag() as u8 == NodeTag::Leaf as u8
    }

    #[inline]
    fn is_branch(&self) -> bool {
        self.tag() as u8 == NodeTag::Branch as u8
    }

    #[inline]
    fn tag(&self) -> NodeTag {
        // SAFETY: The union is marked `#[repr(C)]`, and its fields are all marked `#[repr(C)]` and
        // have `NodeTag` as the first field.
        unsafe { *<*const _>::cast::<NodeTag>(self) }
    }

    #[inline]
    fn is_free(&self) -> bool {
        if let Some(leaf) = self.leaf() {
            leaf.allocation_type == SuballocationType::Free
        } else {
            false
        }
    }
}

impl Leaf {
    #[inline]
    fn offset(&self, region_offset: DeviceSize) -> DeviceSize {
        region_offset + DeviceSize::from(self.compressed_offset) * BuddyAllocator::MIN_NODE_SIZE
    }
}

#[derive(Clone)]
pub struct Suballocations<'a> {
    region_offset: DeviceSize,
    head_ptr: NonNull<SuballocationTreeNode>,
    tail_ptr: NonNull<SuballocationTreeNode>,
    len: usize,
    marker: PhantomData<&'a SuballocationTree>,
}

unsafe impl Send for Suballocations<'_> {}
unsafe impl Sync for Suballocations<'_> {}

impl Iterator for Suballocations<'_> {
    type Item = SuballocationNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            let head = unsafe { self.head_ptr.as_ref() };

            // SAFETY: We start out with `self.left_ptr` being a leaf node, and upon iteration, we
            // set it to the next leaf node.
            let head_ty = unsafe { head.ty.leaf_unchecked() };

            self.head_ptr = head_ty.next_ptr;
            self.len -= 1;

            Some(SuballocationNode {
                offset: head_ty.offset(self.region_offset),
                size: BuddyAllocator::MIN_NODE_SIZE << head_ty.order,
                allocation_type: head_ty.allocation_type,
            })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for Suballocations<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            let tail = unsafe { self.tail_ptr.as_ref() };

            // SAFETY: We start out with `self.right_ptr` being a leaf node, and upon iteration, we
            // set it to the previous leaf node.
            let tail_ty = unsafe { tail.ty.leaf_unchecked() };

            self.tail_ptr = tail_ty.prev_ptr;
            self.len -= 1;

            Some(SuballocationNode {
                offset: tail_ty.offset(self.region_offset),
                size: BuddyAllocator::MIN_NODE_SIZE << tail_ty.order,
                allocation_type: tail_ty.allocation_type,
            })
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Suballocations<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl FusedIterator for Suballocations<'_> {}
