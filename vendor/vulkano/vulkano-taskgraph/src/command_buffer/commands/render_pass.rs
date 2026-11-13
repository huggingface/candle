use crate::command_buffer::{RecordingCommandBuffer, Result};
use vulkano::command_buffer::{ClearAttachment, ClearRect};

/// # Commands for render passes
///
/// These commands require a graphics queue.
impl RecordingCommandBuffer<'_> {
    /// Clears specific regions of specific attachments of the framebuffer, panicking on a
    /// validation error.
    ///
    /// `attachments` specify the types of attachments and their clear values. `rects` specify the
    /// regions to clear.
    ///
    /// If the render pass instance this is recorded in uses multiview then
    /// [`ClearRect::base_array_layer`] must be zero and [`ClearRect::layer_count`] must be one.
    ///
    /// The rectangle area must be inside the render area ranges.
    ///
    /// This is a shortcut for `try_clear_attachments().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_clear_attachments`] returns a [`ValidationError`].
    ///
    /// [`try_clear_attachments`]: Self::try_clear_attachments
    #[track_caller]
    pub unsafe fn clear_attachments(
        &mut self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> &mut Self {
        unsafe { self.try_clear_attachments(attachments, rects) }.unwrap()
    }

    /// Clears specific regions of specific attachments of the framebuffer.
    ///
    /// `attachments` specify the types of attachments and their clear values. `rects` specify the
    /// regions to clear.
    ///
    /// If the render pass instance this is recorded in uses multiview then
    /// [`ClearRect::base_array_layer`] must be zero and [`ClearRect::layer_count`] must be one.
    ///
    /// The rectangle area must be inside the render area ranges.
    pub unsafe fn try_clear_attachments(
        &mut self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> Result<&mut Self> {
        Ok(unsafe { self.clear_attachments_unchecked(attachments, rects) })
    }

    pub unsafe fn clear_attachments_unchecked(
        &mut self,
        attachments: &[ClearAttachment],
        rects: &[ClearRect],
    ) -> &mut Self {
        unsafe { self.inner.clear_attachments_unchecked(attachments, rects) };

        self
    }
}
