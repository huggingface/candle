use super::{AccessCheckError, GpuFuture, SubmitAnyBuilder};
use crate::{
    buffer::Buffer,
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    swapchain::Swapchain,
    DeviceSize, Validated, VulkanError, VulkanObject,
};
use std::{ops::Range, sync::Arc};

/// Joins two futures together.
// TODO: handle errors
pub fn join<F, S>(first: F, second: S) -> JoinFuture<F, S>
where
    F: GpuFuture,
    S: GpuFuture,
{
    assert_eq!(first.device().handle(), second.device().handle());

    if !first.queue_change_allowed() && !second.queue_change_allowed() {
        assert_eq!(first.queue().unwrap(), second.queue().unwrap());
    }

    JoinFuture { first, second }
}

/// Two futures joined into one.
#[must_use]
pub struct JoinFuture<A, B> {
    first: A,
    second: B,
}

unsafe impl<A, B> DeviceOwned for JoinFuture<A, B>
where
    A: DeviceOwned,
    B: DeviceOwned,
{
    fn device(&self) -> &Arc<Device> {
        let device = self.first.device();
        debug_assert_eq!(self.second.device().handle(), device.handle());
        device
    }
}

unsafe impl<A, B> GpuFuture for JoinFuture<A, B>
where
    A: GpuFuture,
    B: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.first.cleanup_finished();
        self.second.cleanup_finished();
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        // Since each future remembers whether it has been flushed, there's no safety issue here
        // if we call this function multiple times.
        self.first.flush()?;
        self.second.flush()?;

        Ok(())
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        // TODO: review this function
        let first = unsafe { self.first.build_submission() }?;
        let second = unsafe { self.second.build_submission() }?;

        // In some cases below we have to submit previous command buffers already, this s done by
        // flushing previous. Since the implementation should remember being flushed it's
        // safe to call build_submission multiple times
        Ok(match (first, second) {
            (SubmitAnyBuilder::Empty, b) => b,
            (a, SubmitAnyBuilder::Empty) => a,
            (SubmitAnyBuilder::SemaphoresWait(mut a), SubmitAnyBuilder::SemaphoresWait(b)) => {
                a.extend(b);
                SubmitAnyBuilder::SemaphoresWait(a)
            }
            (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::CommandBuffer(_, _)) => {
                self.second.flush()?;
                SubmitAnyBuilder::SemaphoresWait(a)
            }
            (SubmitAnyBuilder::CommandBuffer(_, _), SubmitAnyBuilder::SemaphoresWait(b)) => {
                self.first.flush()?;
                SubmitAnyBuilder::SemaphoresWait(b)
            }
            (SubmitAnyBuilder::SemaphoresWait(a), SubmitAnyBuilder::QueuePresent(_)) => {
                self.second.flush()?;
                SubmitAnyBuilder::SemaphoresWait(a)
            }
            (SubmitAnyBuilder::QueuePresent(_), SubmitAnyBuilder::SemaphoresWait(b)) => {
                self.first.flush()?;
                SubmitAnyBuilder::SemaphoresWait(b)
            }
            (
                SubmitAnyBuilder::CommandBuffer(mut submit_info_a, fence_a),
                SubmitAnyBuilder::CommandBuffer(submit_info_b, fence_b),
            ) => {
                assert!(
                    fence_a.is_none() || fence_b.is_none(),
                    "Can't merge two queue submits that both have a fence"
                );

                submit_info_a
                    .wait_semaphores
                    .extend(submit_info_b.wait_semaphores);
                submit_info_a
                    .command_buffers
                    .extend(submit_info_b.command_buffers);
                submit_info_a
                    .signal_semaphores
                    .extend(submit_info_b.signal_semaphores);

                SubmitAnyBuilder::CommandBuffer(submit_info_a, fence_a.or(fence_b))
            }
            (SubmitAnyBuilder::QueuePresent(_), SubmitAnyBuilder::QueuePresent(_)) => {
                self.first.flush()?;
                self.second.flush()?;
                SubmitAnyBuilder::Empty
            }
            (SubmitAnyBuilder::CommandBuffer(_, _), SubmitAnyBuilder::QueuePresent(_)) => {
                unimplemented!()
            }
            (SubmitAnyBuilder::QueuePresent(_), SubmitAnyBuilder::CommandBuffer(_, _)) => {
                unimplemented!()
            }
        })
    }

    unsafe fn signal_finished(&self) {
        unsafe { self.first.signal_finished() };
        unsafe { self.second.signal_finished() };
    }

    fn queue_change_allowed(&self) -> bool {
        self.first.queue_change_allowed() && self.second.queue_change_allowed()
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        match (self.first.queue(), self.second.queue()) {
            (Some(q1), Some(q2)) => {
                if q1 == q2 {
                    Some(q1)
                } else if self.first.queue_change_allowed() {
                    Some(q2)
                } else if self.second.queue_change_allowed() {
                    Some(q1)
                } else {
                    None
                }
            }
            (Some(q), None) => Some(q),
            (None, Some(q)) => Some(q),
            (None, None) => None,
        }
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        let first = self
            .first
            .check_buffer_access(buffer, range.clone(), exclusive, queue);
        let second = self
            .second
            .check_buffer_access(buffer, range, exclusive, queue);
        debug_assert!(
            !(exclusive && first.is_ok() && second.is_ok()),
            "Two futures gave exclusive access to the same resource"
        );
        match (first, second) {
            (v, Err(AccessCheckError::Unknown)) => v,
            (Err(AccessCheckError::Unknown), v) => v,
            (Err(AccessCheckError::Denied(e1)), Err(AccessCheckError::Denied(_))) => {
                Err(AccessCheckError::Denied(e1))
            } // TODO: which one?
            (Ok(()), Err(AccessCheckError::Denied(_)))
            | (Err(AccessCheckError::Denied(_)), Ok(())) => panic!(
                "Contradictory information \
                                                                 between two futures"
            ),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        let first =
            self.first
                .check_image_access(image, range.clone(), exclusive, expected_layout, queue);
        let second =
            self.second
                .check_image_access(image, range, exclusive, expected_layout, queue);
        debug_assert!(
            !(exclusive && first.is_ok() && second.is_ok()),
            "Two futures gave exclusive access to the same resource"
        );
        match (first, second) {
            (v, Err(AccessCheckError::Unknown)) => v,
            (Err(AccessCheckError::Unknown), v) => v,
            (Err(AccessCheckError::Denied(e1)), Err(AccessCheckError::Denied(_))) => {
                Err(AccessCheckError::Denied(e1))
            } // TODO: which one?
            (Ok(()), Err(AccessCheckError::Denied(_)))
            | (Err(AccessCheckError::Denied(_)), Ok(())) => {
                panic!("Contradictory information between two futures")
            }
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        _before: bool,
    ) -> Result<(), AccessCheckError> {
        let first = self
            .first
            .check_swapchain_image_acquired(swapchain, image_index, false);
        let second = self
            .second
            .check_swapchain_image_acquired(swapchain, image_index, false);

        match (first, second) {
            (v, Err(AccessCheckError::Unknown)) => v,
            (Err(AccessCheckError::Unknown), v) => v,
            (Err(AccessCheckError::Denied(e1)), Err(AccessCheckError::Denied(_))) => {
                Err(AccessCheckError::Denied(e1))
            } // TODO: which one?
            (Ok(()), Err(AccessCheckError::Denied(_)))
            | (Err(AccessCheckError::Denied(_)), Ok(())) => Ok(()),
            (Ok(()), Ok(())) => Ok(()), // TODO: Double Acquired?
        }
    }
}
