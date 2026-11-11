//! Garbage collection facilities.

use crate::{
    assert_unsafe_precondition,
    descriptor_set::{
        AccelerationStructureId, SampledImageId, SamplerId, StorageBufferId, StorageImageId,
    },
    resource::{Flight, Resources},
    Id,
};
use concurrent_slotmap::{hyaline, SlotId, SlotMap};
use smallvec::SmallVec;
use std::{
    cell::UnsafeCell,
    fmt,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::ControlFlow,
    ptr,
    sync::{
        atomic::{
            AtomicU32,
            Ordering::{Acquire, Relaxed, Release},
        },
        Arc,
    },
};
use vulkano::{buffer::Buffer, image::Image};

/// A batch of deferred functions to be enqueued together.
///
/// You should batch as many deferred functions together as you can for optimal performance.
///
/// This type is created by the [`create_deferred_batch`] method on [`Resources`].
///
/// [`create_deferred_batch`]: Resources::create_deferred_batch
pub struct DeferredBatch<'a> {
    resources: &'a Resources,
    node_index: u32,
    biased_frames: *mut SmallVec<[(Id<Flight>, u64); 4]>,
    deferreds: *mut Vec<Deferred>,
    guard: hyaline::Guard<'a>,
    drop_guard: bool,
}

impl<'a> DeferredBatch<'a> {
    pub(crate) fn new(resources: &'a Resources) -> Self {
        let guard = resources.pin();
        let garbage_queue = resources.garbage_queue();
        let node_allocator = garbage_queue.node_allocator();
        let (node_index, node) = node_allocator.allocate(&guard);

        DeferredBatch {
            resources,
            node_index,
            biased_frames: node.biased_frames.get(),
            deferreds: node.deferreds.get(),
            guard,
            drop_guard: false,
        }
    }

    /// Defers the destruction of the buffer corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid physical resource ID.
    #[inline]
    #[track_caller]
    pub fn destroy_buffer(&mut self, id: Id<Buffer>) -> &mut Self {
        self.resources
            .try_invalidate_buffer(id, &self.guard)
            .expect("invalid buffer ID");

        self.defer(move |resources| {
            // SAFETY: Since this method is the only way to remove the buffer and future attempts
            // to remove it will fail because we invalidated the buffer above, there is no way the
            // buffer can be removed besides this deferred function.
            unsafe { resources.remove_invalidated_buffer_unchecked(id) };
        })
    }

    /// Defers the destruction of the image corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid physical resource ID.
    #[inline]
    #[track_caller]
    pub fn destroy_image(&mut self, id: Id<Image>) -> &mut Self {
        self.resources
            .try_invalidate_image(id, &self.guard)
            .expect("invalid image ID");

        self.defer(move |resources| {
            // SAFETY: Same as in the `destroy_buffer` method above.
            unsafe { resources.remove_invalidated_image_unchecked(id) };
        })
    }

    /// Defers the destruction of the sampler descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    #[track_caller]
    pub fn destroy_sampler(&mut self, id: SamplerId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_sampler(id, &self.guard)
            .expect("invalid sampler ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: Same as in the `destroy_buffer` method above.
            unsafe { bcx.global_set().remove_invalidated_sampler_unchecked(id) };
        })
    }

    /// Defers the destruction of the sampled image descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    #[track_caller]
    pub fn destroy_sampled_image(&mut self, id: SampledImageId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_sampled_image(id, &self.guard)
            .expect("invalid sampled image ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: Same as in the `destroy_buffer` method above.
            unsafe {
                bcx.global_set()
                    .remove_invalidated_sampled_image_unchecked(id)
            };
        })
    }

    /// Defers the destruction of the storage image descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    #[track_caller]
    pub fn destroy_storage_image(&mut self, id: StorageImageId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_storage_image(id, &self.guard)
            .expect("invalid storage image ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: Same as in the `destroy_buffer` method above.
            unsafe {
                bcx.global_set()
                    .remove_invalidated_storage_image_unchecked(id)
            };
        })
    }

    /// Defers the destruction of the storage buffer descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    #[track_caller]
    pub fn destroy_storage_buffer(&mut self, id: StorageBufferId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_storage_buffer(id, &self.guard)
            .expect("invalid storage buffer ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: Same as in the `destroy_buffer` method above.
            unsafe {
                bcx.global_set()
                    .remove_invalidated_storage_buffer_unchecked(id)
            };
        })
    }

    /// Defers the destruction of the acceleration structure descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    #[track_caller]
    pub fn destroy_acceleration_structure(&mut self, id: AccelerationStructureId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_acceleration_structure(id, &self.guard)
            .expect("invalid acceleration structure ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: Same as in the `destroy_buffer` method above.
            unsafe {
                bcx.global_set()
                    .remove_invalidated_acceleration_structure_unchecked(id)
            };
        })
    }

    /// Defers the destruction of the given `object`.
    #[inline]
    pub fn destroy_object(&mut self, object: impl Send + 'static) -> &mut Self {
        self.defer(|_| drop(object))
    }

    /// Defers the given function.
    #[inline]
    pub fn defer(&mut self, f: impl FnOnce(&Resources) + Send + 'static) -> &mut Self {
        if cfg!(debug_assertions) {
            // This whole block is conditional on debug assertions because this adds a word of
            // memory overhead and the inline storage of `Deferred` is limited to 3 words.

            let addr = ptr::from_ref(self.resources) as usize;

            self.deferreds_mut().push(Deferred::new(move |resources| {
                assert_unsafe_precondition!(
                    ptr::from_ref(resources) as usize == addr,
                    "`Deferred::call_in_place` must be called with the same `Resources` collection
                    as the one for which the deferred function was issued",
                );

                f(resources);
            }));
        } else {
            self.deferreds_mut().push(Deferred::new(f));
        }

        self
    }

    pub(crate) fn deferreds_mut(&mut self) -> &mut Vec<Deferred> {
        // SAFETY: By our own invariant, `self.deferreds` is valid so long as one of the `enqueue`
        // methods hasn't been called.
        unsafe { &mut *self.deferreds }
    }

    /// Enqueues the deferred functions for after all flights have been waited on.
    pub fn enqueue(self) {
        // See the `Drop` implementation.
    }

    /// Enqueues the deferred functions for after the flights corresponding to the given
    /// `flight_ids` have been waited on.
    ///
    /// # Safety
    ///
    /// - There must be no other flights that use any resource or descriptor that is being
    ///   destroyed by the deferred functions.
    ///
    /// # Panics
    ///
    /// - Panics if `flight_ids` contains an invalid ID.
    /// - Panics if `flight_ids` panics.
    ///
    /// # Leaks
    ///
    /// If the method panics for any reason (including if the provided iterator panics), then the
    /// deferred functions will never be called. For any destruction that was deferred, the
    /// corresponding resource or descriptor will be leaked.
    pub unsafe fn enqueue_with_flights(self, flight_ids: impl IntoIterator<Item = Id<Flight>>) {
        let guard = self.guard.clone();

        let biased_frames = flight_ids.into_iter().flat_map(|flight_id| {
            let flight = self
                .resources
                .try_flight_protected(flight_id, &guard)
                .expect("invalid flight ID");
            let biased_frame = flight.biased_started_frame();

            (!flight.is_biased_frame_complete(biased_frame)).then_some((flight_id, biased_frame))
        });

        let mut this = ManuallyDrop::new(self);

        // SAFETY: We have wrapped `self` in a `ManuallyDrop`, which means the drop glue cannot
        // drop the guard naturally.
        unsafe { this.set_drop_guard() };

        // SAFETY:
        // * We own `self`, which ensures that this method isn't called again.
        // * We have wrapped `self` in a `ManuallyDrop` to ensure that the `Drop` implementation
        //   doesn't call this method.
        // * The caller must ensure that `flight_ids` constitutes the correct set of flights for
        //   our deferred functions.
        unsafe { this.enqueue_inner(biased_frames) };
    }

    /// Enqueues the deferred functions for after the given `frames` have been waited on.
    ///
    /// `frames` provides mappings from a flight ID to the corresponding [frame] that should be
    /// waited on.
    ///
    /// # Safety
    ///
    /// - There must be no other flights that use any resource or descriptor that is being
    ///   destroyed by the deferred functions.
    /// - The given flights must not use any resource or descriptor that is being destroyed by the
    ///   deferred functions past the corresponding frame.
    ///
    /// # Panics
    ///
    /// - Panics if `frames` contains an invalid ID.
    /// - Panics if `frames` panics.
    ///
    /// # Leaks
    ///
    /// If the method panics for any reason (including if the provided iterator panics), then the
    /// deferred functions will never be called. For any destruction that was deferred, the
    /// corresponding resource or descriptor will be leaked.
    pub unsafe fn enqueue_with_frames(self, frames: impl IntoIterator<Item = (Id<Flight>, u64)>) {
        let guard = self.guard.clone();

        let biased_frames = frames.into_iter().flat_map(|(flight_id, frame)| {
            let flight = self
                .resources
                .try_flight_protected(flight_id, &guard)
                .expect("invalid flight ID");
            let biased_frame = frame + u64::from(flight.frame_count()) + 1;

            (!flight.is_biased_frame_complete(biased_frame)).then_some((flight_id, biased_frame))
        });

        let mut this = ManuallyDrop::new(self);

        // SAFETY: We have wrapped `self` in a `ManuallyDrop`, which means the drop glue cannot
        // drop the guard naturally.
        unsafe { this.set_drop_guard() };

        // SAFETY:
        // * We own `self`, which ensures that this method isn't called again.
        // * We have wrapped `self` in a `ManuallyDrop` to ensure that the `Drop` implementation
        //   doesn't call this method.
        // * The caller must ensure that `frames` constitutes the correct set of frames for our
        //   deferred functions.
        unsafe { this.enqueue_inner(biased_frames) };
    }

    unsafe fn set_drop_guard(&mut self) {
        self.drop_guard = true;
    }

    unsafe fn enqueue_inner(&mut self, biased_frames: impl IntoIterator<Item = (Id<Flight>, u64)>) {
        struct ClearGuard<'a, 'b>(&'a mut DeferredBatch<'b>);

        impl Drop for ClearGuard<'_, '_> {
            #[cold]
            fn drop(&mut self) {
                self.0.deferreds_mut().clear();

                let node_allocator = self.0.resources.garbage_queue().node_allocator();

                // SAFETY: By our own invariant, the node denoted by `self.node_index` must have
                // been allocated by us, and since we haven't pushed it to any garbage queue, the
                // node is still unlinked and therefore safe to deallocate.
                unsafe { node_allocator.deallocate(self.0.node_index, &self.0.guard) };

                if self.0.drop_guard {
                    // SAFETY: The caller must ensure that this method is not called again.
                    unsafe { ptr::drop_in_place(&mut self.0.guard) };
                }
            }
        }

        let this = ClearGuard(self);

        // SAFETY: By our own invariant, the node denoted by `self.node_index` must have been
        // allocated by us, which means that no other threads can be accessing this node's data
        // while we haven't pushed it to a garbage queue.
        let node_biased_frames = unsafe { &mut *this.0.biased_frames };
        node_biased_frames.clear();
        node_biased_frames.extend(biased_frames);

        mem::forget(this);

        if self.deferreds_mut().is_empty() {
            let node_allocator = self.resources.garbage_queue().node_allocator();

            // SAFETY: By our own invariant, the node denoted by `self.node_index` must have been
            // allocated by us, and since we haven't pushed it to any garbage queue, the node is
            // still unlinked and therefore safe to deallocate.
            unsafe { node_allocator.deallocate(self.node_index, &self.guard) };

            self.guard.flush();

            if self.drop_guard {
                // SAFETY: The caller must ensure that this method is not called again.
                unsafe { ptr::drop_in_place(&mut self.guard) };
            }

            return;
        }

        match node_biased_frames.as_slice() {
            [] => {
                // SAFETY: There are no frames to wait on; we can collect immediately.
                unsafe { collect(self.resources, self.deferreds_mut()) };

                let node_allocator = self.resources.garbage_queue().node_allocator();

                // SAFETY: By our own invariant, the node denoted by `self.node_index` must have
                // been allocated by us, and since we haven't pushed it to any garbage queue, the
                // node is still unlinked and therefore safe to deallocate.
                unsafe { node_allocator.deallocate(self.node_index, &self.guard) };
            }
            &[(flight_id, _)] => {
                // SAFETY: The caller must ensure that `frames` contained valid flight IDs when
                // extending the node's `frames` above. Since we have owned a `hyaline::Guard` from
                // then until now, the flight cannot have been dropped yet even if it were to be
                // removed between then and now.
                let flight = unsafe {
                    self.resources
                        .flight_unchecked_protected(flight_id, &self.guard)
                };

                let garbage_queue = flight.garbage_queue();

                // SAFETY:
                // * By our own invariant, the node denoted by `self.node_index` must have been
                //   allocated by us.
                // * The caller must ensure that the node's data is not accessed mutably again.
                // * The caller must ensure that this method is not called again.
                // * The caller must ensure that `frames` constitutes the correct set of frames for
                //   our deferred functions.
                unsafe { garbage_queue.push(self.node_index, &self.guard) };
            }
            _ => {
                let garbage_queue = self.resources.garbage_queue();

                // SAFETY: Same as the `push` above.
                unsafe { garbage_queue.push(self.node_index, &self.guard) };
            }
        }

        self.guard.flush();

        if self.drop_guard {
            // SAFETY: The caller must ensure that this method is not called again.
            unsafe { ptr::drop_in_place(&mut self.guard) };
        }
    }
}

impl Drop for DeferredBatch<'_> {
    /// Dropping a `DeferredBatch` does the same thing as calling [`enqueue`].
    ///
    /// [`enqueue`]: Self::enqueue
    fn drop(&mut self) {
        let guard = self.guard.clone();
        let flights = self.resources.flights_protected(&guard);
        let biased_frames = flights.flat_map(|(flight_id, flight)| {
            let biased_frame = flight.biased_started_frame();

            (!flight.is_biased_frame_complete(biased_frame)).then_some((flight_id, biased_frame))
        });

        // SAFETY:
        // * We're dropping `self`, which ensures that this method isn't called again.
        // * `frames` includes the current frame of every flight, so there can be no flight that
        //   uses any of the resources/descriptors being destroyed that isn't waited on.
        unsafe { self.enqueue_inner(biased_frames) };
    }
}

const DATA_WORDS: usize = 3;

/// A deferred function.
///
/// This is essentially a `dyn FnOnce(&Resources)` except sized and stored inline.
pub(crate) struct Deferred {
    call: unsafe fn(*mut (), &Resources),
    data: MaybeUninit<Data>,
}

type Data = [usize; DATA_WORDS];

// SAFETY: `Deferred::new` requires that the function is `Send`, so it's safe to send the
// `Deferred` to another thread as well.
unsafe impl Send for Deferred {}

// SAFETY: We only give owning or mutable access to the contained function, so it's safe to share a
// reference to the `Deferred` as it can't be used for anything.
unsafe impl Sync for Deferred {}

impl Deferred {
    pub(crate) fn destroy(object: impl Send + 'static) -> Self {
        Self::new(|_| drop(object))
    }

    pub(crate) fn new<F: FnOnce(&Resources) + Send + 'static>(f: F) -> Self {
        if size_of::<F>() <= size_of::<Data>() && align_of::<F>() <= align_of::<Data>() {
            let mut data = MaybeUninit::<Data>::uninit();

            // SAFETY: The pointer is valid for writes and we checked that `Data` has a layout that
            // can fit `F`.
            unsafe { data.as_mut_ptr().cast::<F>().write(f) };

            Deferred {
                // SAFETY: The caller of `call` must ensure that `data` is our same data and that
                // `resources` is the collection for which the deferred function was issued.
                call: |data, resources| unsafe { data.cast::<F>().read() }(resources),
                data,
            }
        } else {
            const { assert!(size_of::<Box<F>>() <= size_of::<Data>()) };
            const { assert!(align_of::<Box<F>>() <= align_of::<Data>()) };

            let mut data = MaybeUninit::<Data>::uninit();

            // SAFETY: The pointer is valid for writes and we checked that `Data` has a layout that
            // can fit `Box<F>`.
            unsafe { data.as_mut_ptr().cast::<Box<F>>().write(Box::new(f)) };

            Deferred {
                // SAFETY: The caller of `call` must ensure that `data` is our same data and that
                // `resources` is the collection for which the deferred function was issued.
                call: |data, resources| unsafe { data.cast::<Box<F>>().read() }(resources),
                data,
            }
        }
    }

    #[inline]
    unsafe fn call_in_place(&mut self, resources: &Resources) {
        // SAFETY:
        // * The constructor of `Deferred` must ensure that `self.call` is safe to call with a
        //   pointer to `self.data`.
        // * The caller must ensure that the function is only called once.
        // * The caller must ensure that `resources` is the collection for which this deferred
        //   function was issued.
        unsafe { (self.call)(self.data.as_mut_ptr().cast(), resources) };
    }
}

impl fmt::Debug for Deferred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deferred").finish_non_exhaustive()
    }
}

pub(crate) struct GlobalQueue {
    inner: Queue,
}

impl GlobalQueue {
    pub(crate) unsafe fn new(hyaline_collector: hyaline::CollectorHandle) -> Self {
        GlobalQueue {
            // SAFETY: Enforced by the caller.
            inner: unsafe { Queue::with_collector(hyaline_collector) },
        }
    }

    fn node_allocator(&self) -> &NodeAllocator {
        &self.inner.node_allocator
    }

    pub(crate) fn register_local(&self) -> LocalQueue {
        LocalQueue {
            inner: Queue::with_allocator(self.inner.node_allocator.clone()),
        }
    }

    unsafe fn push(&self, node_index: u32, guard: &hyaline::Guard<'_>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.push(node_index, guard) };
    }

    pub(crate) unsafe fn collect(&self, resources: &Resources, guard: &hyaline::Guard<'_>) {
        let predicate = |biased_frames: &[(Id<Flight>, u64)]| {
            for &(flight_id, biased_frame) in biased_frames {
                let Ok(flight) = resources.try_flight_protected(flight_id, guard) else {
                    continue;
                };

                if !flight.is_biased_frame_complete(biased_frame) {
                    return ControlFlow::Continue(false);
                }
            }

            ControlFlow::Continue(true)
        };

        // SAFETY:
        // * The `predicate` correctly collects only if all frames are completed.
        // * The caller must ensure that `resources` is the collection for which the deferred
        //   functions were issued.
        // * The caller must ensure that `guard.global()` is that of `resources.global()`.
        unsafe { self.inner.collect(predicate, resources, guard) };
    }

    pub(crate) unsafe fn drop(&self, resources: &Resources, guard: &hyaline::Guard<'_>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.drop(resources, guard) };
    }
}

impl fmt::Debug for GlobalQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlobalQueue").finish_non_exhaustive()
    }
}

pub(crate) struct LocalQueue {
    inner: Queue,
}

impl LocalQueue {
    unsafe fn push(&self, node_index: u32, guard: &hyaline::Guard<'_>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.push(node_index, guard) };
    }

    pub(crate) unsafe fn collect(
        &self,
        resources: &Resources,
        flight: &Flight,
        guard: &hyaline::Guard<'_>,
    ) {
        let predicate = |biased_frames: &[(Id<Flight>, u64)]| {
            if let &[(_, biased_frame)] = biased_frames {
                if flight.is_biased_frame_complete(biased_frame) {
                    ControlFlow::Continue(true)
                } else {
                    ControlFlow::Break(())
                }
            } else {
                unreachable!();
            }
        };

        // SAFETY:
        // * The `predicate` correctly collects only if the frame is completed.
        // * The caller must ensure that `resources` is the collection for which the deferred
        //   functions were issued.
        // * The caller must ensure that `guard.global()` is that of `resources.global()`.
        unsafe { self.inner.collect(predicate, resources, guard) };
    }

    pub(crate) unsafe fn drop(&self, resources: &Resources, guard: &hyaline::Guard<'_>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.drop(resources, guard) };
    }
}

impl fmt::Debug for LocalQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalQueue").finish_non_exhaustive()
    }
}

#[allow(clippy::identity_op)]
const NIL: u32 = u32::MAX & !DELETED_BIT;
const DELETED_BIT: u32 = 1 << 31;

struct Queue {
    head: u32,
    node_allocator: Arc<NodeAllocator>,
}

impl Queue {
    #[cfg(test)]
    fn new() -> Self {
        unsafe { Self::with_collector(hyaline::CollectorHandle::new()) }
    }

    unsafe fn with_collector(hyaline_collector: hyaline::CollectorHandle) -> Self {
        // SAFETY: Enforced by the caller.
        let node_allocator = Arc::new(unsafe { NodeAllocator::new(hyaline_collector) });

        Self::with_allocator(node_allocator)
    }

    fn with_allocator(node_allocator: Arc<NodeAllocator>) -> Self {
        let guard = node_allocator.inner.pin();
        let (sentinel_index, sentinel_node) = node_allocator.allocate(&guard);
        sentinel_node.next.store(NIL, Relaxed);
        drop(guard);

        Queue {
            head: sentinel_index,
            node_allocator,
        }
    }

    unsafe fn push(&self, index: u32, guard: &hyaline::Guard<'_>) {
        // SAFETY: The caller must ensure that `index` is valid and that the node is not mutated
        // again.
        let node = unsafe { self.node_allocator.get_unchecked(index, guard) };

        node.next.store(NIL, Relaxed);

        // SAFETY: `self.head` is always valid.
        let head_node = unsafe { self.node_allocator.get_unchecked(self.head, guard) };

        let mut prev_node = head_node;
        let mut curr_index = prev_node.next.load(Acquire);

        loop {
            debug_assert!(!is_deleted(curr_index));

            if curr_index == NIL {
                match prev_node
                    .next
                    .compare_exchange(NIL, index, Release, Acquire)
                {
                    Ok(_) => {
                        // We successfully linked `node` after the tail node.
                        break;
                    }
                    Err(new_curr_index) if is_deleted(new_curr_index) => {
                        // Another thread deleted `prev_node`. We have to restart from the
                        // beginning, as our snapshot of the queue is now inconsistent.
                        prev_node = head_node;
                        curr_index = prev_node.next.load(Acquire);
                        continue;
                    }
                    Err(new_curr_index) => {
                        // Another thread linked another node after the tail node. Try to move onto
                        // the node in the next iteration and try again.
                        curr_index = new_curr_index;
                        continue;
                    }
                }
            }

            // SAFETY: We always push indices of existing nodes into the queue.
            let curr_node = unsafe { self.node_allocator.get_unchecked(curr_index, guard) };

            let next_index = curr_node.next.load(Acquire);

            // Try to unlink any deleted nodes we encounter. We cannot link a node after a deleted
            // node, and we also cannot unlink a node following a deleted node.
            if is_deleted(next_index) {
                match prev_node.next.compare_exchange(
                    curr_index,
                    next_index & !DELETED_BIT,
                    Release,
                    Relaxed,
                ) {
                    Ok(_) => {
                        // Our thread unlinked `curr_node`. Deallocate `curr_node` and move onto
                        // the next node.

                        // SAFETY: We successfully unlinked the node from the queue such that it is
                        // not reachable going forward and no other thread can be deallocating this
                        // same node.
                        unsafe { self.node_allocator.deallocate(curr_index, guard) };

                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                    Err(new_curr_index) if is_deleted(new_curr_index) => {
                        // Another thread deleted `prev_node`. We have to restart from the
                        // beginning, as our snapshot of the queue is now inconsistent.
                        prev_node = head_node;
                        curr_index = prev_node.next.load(Acquire);
                        continue;
                    }
                    Err(_) => {
                        // Another thread unlinked `curr_node`. Move onto the next node without
                        // deallocating `curr_node`.
                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                }
            }

            prev_node = curr_node;
            curr_index = next_index;
        }
    }

    unsafe fn collect(
        &self,
        mut collect_predicate: impl FnMut(&[(Id<Flight>, u64)]) -> ControlFlow<(), bool>,
        resources: &Resources,
        guard: &hyaline::Guard<'_>,
    ) {
        // SAFETY: `self.head` is always valid.
        let head_node = unsafe { self.node_allocator.get_unchecked(self.head, guard) };

        let mut prev_node = head_node;
        let mut curr_index = prev_node.next.load(Acquire);

        loop {
            debug_assert!(!is_deleted(curr_index));

            if curr_index == NIL {
                break;
            }

            // SAFETY: We always push indices of existing nodes into the queue.
            let curr_node = unsafe { self.node_allocator.get_unchecked(curr_index, guard) };

            let next_index = curr_node.next.load(Acquire);

            // Try to unlink any deleted nodes we encounter. We cannot collect and/or delete an
            // already deleted node, and we also cannot unlink a node following a deleted node.
            if is_deleted(next_index) {
                match prev_node.next.compare_exchange(
                    curr_index,
                    next_index & !DELETED_BIT,
                    Release,
                    Relaxed,
                ) {
                    Ok(_) => {
                        // Our thread unlinked `curr_node`. Deallocate `curr_node` and move onto
                        // the next node.

                        // SAFETY: We successfully unlinked the node from the queue such that it is
                        // not reachable going forward and no other thread can be deallocating this
                        // same node.
                        unsafe { self.node_allocator.deallocate(curr_index, guard) };

                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                    Err(new_curr_index) if is_deleted(new_curr_index) => {
                        // Another thread deleted `prev_node`. We have to restart from the
                        // beginning, as our snapshot of the queue is now inconsistent.
                        prev_node = head_node;
                        curr_index = prev_node.next.load(Acquire);
                        continue;
                    }
                    Err(_) => {
                        // Another thread unlinked `curr_node`. Move onto the next node without
                        // deallocating `curr_node`.
                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                }
            }

            // SAFETY: The caller of `Queue::push` must ensure that the pushed node is not accessed
            // mutably again such that we have shared access to the node's `frames`.
            let biased_frames = unsafe { &*curr_node.biased_frames.get() };

            match collect_predicate(biased_frames) {
                ControlFlow::Continue(false) => {
                    // The node cannot be collected but there might be more collectable nodes. Move
                    // onto the next node.
                    prev_node = curr_node;
                    curr_index = next_index;
                    continue;
                }
                ControlFlow::Continue(true) => {
                    // The node can be collected and there might be more collectable nodes. Collect
                    // the node and move onto the next.
                }
                ControlFlow::Break(()) => {
                    // The node cannot be collected and there cannot be more collectable nodes.
                    break;
                }
            }

            match curr_node.next.compare_exchange(
                next_index,
                next_index | DELETED_BIT,
                Release,
                Relaxed,
            ) {
                Ok(_) => {
                    // Our thread deleted `curr_node`. Collect it and move onto the next node.

                    // SAFETY: We successfully marked the node as deleted such that no other thread
                    // can be collecting this same node.
                    let deferreds = unsafe { &mut *curr_node.deferreds.get() };

                    // SAFETY:
                    // * The caller must ensure that `resources` is the correct collection for the
                    //   `deferreds`.
                    // * The caller must ensure that `frames` have been waited on.
                    unsafe { collect(resources, deferreds) };

                    prev_node = curr_node;
                    curr_index = next_index;
                }
                Err(_) => {
                    // Another thread deleted `curr_node` or unlinked `prev_node`. We have to
                    // restart from the beginning, as our snapshot of the queue is now
                    // inconsistent.
                    prev_node = head_node;
                    curr_index = prev_node.next.load(Acquire);
                }
            }
        }
    }

    unsafe fn drop(&self, resources: &Resources, guard: &hyaline::Guard<'_>) {
        // SAFETY: `self.head` is always valid.
        let head_node = unsafe { self.node_allocator.get_unchecked(self.head, guard) };

        let mut curr_index = head_node.next.load(Relaxed);

        while curr_index != NIL {
            // SAFETY: We always push indices of existing nodes into the queue.
            let curr_node = unsafe { self.node_allocator.get_unchecked(curr_index, guard) };

            let next_index = curr_node.next.load(Relaxed);

            // The caller must ensure that we have exclusive access to the queue, so we can simply
            // skip over deleted nodes without unlinking them.
            if is_deleted(next_index) {
                curr_index = next_index & !DELETED_BIT;
                continue;
            }

            // The caller must ensure that we have exclusive access to the queue, so we don't need
            // to use `compare_exchange` or stronger orderings here.
            curr_node.next.store(next_index | DELETED_BIT, Relaxed);

            // SAFETY: We marked the node as deleted such that the following `Collector::collect`
            // call cannot reentrantly collect this same node.
            let deferreds = unsafe { &mut *curr_node.deferreds.get() };

            // SAFETY:
            // * The caller must ensure that `resources` is the correct collection for the
            //   `deferreds`.
            // * The caller must ensure that all frames have been waited on.
            unsafe { collect(resources, deferreds) };

            curr_index = next_index;
        }
    }
}

fn is_deleted(index: u32) -> bool {
    index & DELETED_BIT != 0
}

unsafe fn collect(resources: &Resources, deferreds: &mut Vec<Deferred>) {
    struct ClearGuard<'a>(&'a mut Vec<Deferred>);

    impl Drop for ClearGuard<'_> {
        fn drop(&mut self) {
            self.0.clear();
        }
    }

    for deferred in ClearGuard(deferreds).0.iter_mut() {
        // SAFETY:
        // * The deferred cannot be called again because we call it once and ensure that the
        //   `deferreds` vector gets cleared afterward in all cases.
        // * The caller must ensure that `resources` is the collection for which the `deferreds`
        //   were issued.
        unsafe { deferred.call_in_place(resources) };
    }
}

struct Node {
    next: AtomicU32,
    biased_frames: UnsafeCell<SmallVec<[(Id<Flight>, u64); 4]>>,
    deferreds: UnsafeCell<Vec<Deferred>>,
}

unsafe impl Sync for Node {}

const MAX_NODES: u32 = 1 << 24;

struct NodeAllocator {
    inner: SlotMap<SlotId, MaybeUninit<Node>>,
}

impl NodeAllocator {
    unsafe fn new(hyaline_collector: hyaline::CollectorHandle) -> Self {
        NodeAllocator {
            // SAFETY: Enforced by the caller
            inner: unsafe { SlotMap::with_collector(MAX_NODES, hyaline_collector) },
        }
    }

    fn allocate<'a>(&'a self, guard: &'a hyaline::Guard<'a>) -> (u32, &'a Node) {
        let (id, node) = self.inner.revive_or_insert_with(guard, |_| {
            MaybeUninit::new(Node {
                next: AtomicU32::new(NIL),
                biased_frames: UnsafeCell::new(SmallVec::new()),
                deferreds: UnsafeCell::new(Vec::new()),
            })
        });

        // SAFETY: The node is initialized.
        (id.index(), unsafe { node.assume_init_ref() })
    }

    unsafe fn deallocate<'a>(&'a self, index: u32, guard: &'a hyaline::Guard<'a>) {
        // SAFETY: The generation's state tag is `OCCUPIED_TAG`.
        let id = unsafe { SlotId::new_unchecked(index, SlotId::OCCUPIED_TAG) };

        // SAFETY: The caller must ensure that the given node is not reachable anymore.
        unsafe { self.inner.remove_unchecked(id, guard) };
    }

    unsafe fn get_unchecked<'a>(&'a self, index: u32, guard: &'a hyaline::Guard<'a>) -> &'a Node {
        // SAFETY: The generation's state tag is `OCCUPIED_TAG`.
        let id = unsafe { SlotId::new_unchecked(index, SlotId::OCCUPIED_TAG) };

        // SAFETY: The caller must ensure that `index` is in bounds and that the slot is occupied.
        let node = unsafe { self.inner.get_unchecked(id, guard) };

        // SAFETY: The node is initialized.
        unsafe { node.assume_init_ref() }
    }
}

impl Drop for NodeAllocator {
    fn drop(&mut self) {
        for slot in self.inner.slots(&self.inner.pin()) {
            // SAFETY: The node is initialized, and we are being dropped which means that the node
            // cannot be accessed again.
            unsafe { slot.value_ptr().cast::<Node>().drop_in_place() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_queues;
    use std::{sync::Barrier, thread};

    const ITERATIONS: u32 = 1000;
    const THREADS: usize = 8;

    #[test]
    fn queue_basic_usage() {
        let (resources, _) = test_queues!();
        let queue = Queue::new();

        {
            let guard = &queue.node_allocator.inner.pin();
            unsafe { queue.push(queue.node_allocator.allocate(guard).0, guard) };
            unsafe { queue.push(queue.node_allocator.allocate(guard).0, guard) };
        }

        assert_eq!(queue_len(&queue), (2, 0));

        {
            let guard = &queue.node_allocator.inner.pin();
            let predicate = |_: &_| ControlFlow::Continue(true);
            unsafe { queue.collect(predicate, &resources, guard) };
        }

        assert_eq!(queue_len(&queue), (0, 2));

        {
            let guard = &queue.node_allocator.inner.pin();
            let predicate = |_: &_| ControlFlow::Continue(true);
            unsafe { queue.collect(predicate, &resources, guard) };
        }

        assert_eq!(queue_len(&queue), (0, 0));
    }

    #[test]
    fn queue_collect_stress() {
        let (resources, _) = test_queues!();
        let queue = Queue::new();

        // TODO: `Queue::push` is quadratic in isolation.
        {
            let guard = &queue.node_allocator.inner.pin();
            let (mut next_index, node) = queue.node_allocator.allocate(guard);
            node.next.store(NIL, Relaxed);

            for _ in 0..ITERATIONS - 1 {
                let (index, node) = queue.node_allocator.allocate(guard);
                node.next.store(next_index, Relaxed);
                next_index = index;
            }

            let head_node = unsafe { queue.node_allocator.get_unchecked(queue.head, guard) };
            head_node.next.store(next_index, Relaxed);
        }

        assert_eq!(queue_len(&queue), (ITERATIONS, 0));

        let barrier = Barrier::new(THREADS);

        thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    let guard = &queue.node_allocator.inner.pin();
                    let predicate = |_: &_| ControlFlow::Continue(true);

                    barrier.wait();

                    unsafe { queue.collect(predicate, &resources, guard) };
                });
            }
        });

        assert_eq!(queue_len(&queue), (0, 0));
    }

    #[test]
    fn queue_push_collect_stress() {
        let (resources, _) = test_queues!();
        let queue = Queue::new();
        let barrier = Barrier::new(THREADS);

        thread::scope(|scope| {
            for _ in 0..THREADS / 2 {
                scope.spawn(|| {
                    barrier.wait();

                    for _ in 0..ITERATIONS {
                        let guard = &queue.node_allocator.inner.pin();
                        let (index, _) = queue.node_allocator.allocate(guard);
                        unsafe { queue.push(index, guard) };
                    }
                });
            }

            for _ in 0..THREADS / 2 {
                scope.spawn(|| {
                    let predicate = |_: &_| ControlFlow::Continue(true);

                    barrier.wait();

                    for _ in 0..ITERATIONS {
                        let guard = &queue.node_allocator.inner.pin();
                        unsafe { queue.collect(predicate, &resources, guard) };
                    }
                });
            }
        });
    }

    fn queue_len(queue: &Queue) -> (u32, u32) {
        let guard = &queue.node_allocator.inner.pin();
        let head_node = unsafe { queue.node_allocator.get_unchecked(queue.head, guard) };
        let mut curr_index = head_node.next.load(Relaxed);
        let mut count = 0;
        let mut deleted_count = 0;

        while curr_index != NIL {
            let curr_node = unsafe { queue.node_allocator.get_unchecked(curr_index, guard) };

            let next_index = curr_node.next.load(Relaxed);

            if is_deleted(next_index) {
                deleted_count += 1;
            } else {
                count += 1;
            }

            curr_index = next_index & !DELETED_BIT;
        }

        (count, deleted_count)
    }
}
