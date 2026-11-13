//! Debug messenger called by intermediate layers or by the driver.
//!
//! When working on an application, it is recommended to register a debug messenger. For example if
//! you enable the validation layers provided by the official Vulkan SDK, they will warn you about
//! invalid API usages or performance problems by calling this callback. The callback can also
//! be called by the driver or by whatever intermediate layer is activated.
//!
//! Note that the vulkano library can also emit messages to warn you about performance issues.
//! TODO: ^ that's not the case yet, need to choose whether we keep this idea
//!
//! # Examples
//!
//! ```
//! # use std::sync::Arc;
//! # use vulkano::instance::Instance;
//! use vulkano::instance::debug::{
//!     DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
//! };
//!
//! # let instance: Arc<Instance> = return;
//! #
//! let _callback = DebugUtilsMessenger::new(
//!     &instance,
//!     &DebugUtilsMessengerCreateInfo::new(&unsafe {
//!         DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
//!             println!("Debug callback: {:?}", callback_data.message);
//!         })
//!     }),
//! )
//! .ok();
//! ```
//!
//! Note that you must keep the `_callback` object alive for as long as you want your callback to
//! be callable. If you don't store the return value of `DebugUtilsMessenger`'s constructor in a
//! variable, it will be immediately destroyed and your callback will not work.

use super::{Instance, InstanceExtensions};
use crate::{
    macros::{vulkan_bitflags, vulkan_enum},
    DebugWrapper, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk;
use std::{
    ffi::{c_void, CStr, CString},
    fmt::{Debug, Error as FmtError, Formatter},
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe, RefUnwindSafe},
    ptr, slice,
    sync::Arc,
};

/// Registration of a callback called by validation layers.
///
/// The callback can be called as long as this object is alive.
#[must_use = "The DebugUtilsMessenger object must be kept alive for as long as you want your callback to be called"]
pub struct DebugUtilsMessenger {
    handle: vk::DebugUtilsMessengerEXT,
    instance: DebugWrapper<Arc<Instance>>,
    _user_callback: Arc<DebugUtilsMessengerCallback>,
}

impl DebugUtilsMessenger {
    /// Initializes a debug callback, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        instance: &Arc<Instance>,
        create_info: &DebugUtilsMessengerCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        match Self::try_new(instance, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Initializes a debug callback.
    #[inline]
    pub fn try_new(
        instance: &Arc<Instance>,
        create_info: &DebugUtilsMessengerCreateInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::validate_new(instance, create_info)?;

        Ok(unsafe { Self::new_unchecked(instance, create_info) }?)
    }

    fn validate_new(
        instance: &Instance,
        create_info: &DebugUtilsMessengerCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_debug_utils {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        create_info
            .validate(instance)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        instance: &Arc<Instance>,
        create_info: &DebugUtilsMessengerCreateInfo<'_>,
    ) -> Result<Self, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.ext_debug_utils.create_debug_utils_messenger_ext)(
                    instance.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(DebugUtilsMessenger {
            handle,
            instance: DebugWrapper(instance.clone()),
            _user_callback: create_info.user_callback.clone(),
        })
    }
}

impl Drop for DebugUtilsMessenger {
    #[inline]
    fn drop(&mut self) {
        let fns = self.instance.fns();
        unsafe {
            (fns.ext_debug_utils.destroy_debug_utils_messenger_ext)(
                self.instance.handle(),
                self.handle,
                ptr::null(),
            )
        };
    }
}

impl Debug for DebugUtilsMessenger {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            instance,
            _user_callback: _,
        } = self;

        f.debug_struct("DebugUtilsMessenger")
            .field("handle", handle)
            .field("instance", instance)
            .finish_non_exhaustive()
    }
}

/// Parameters to create a `DebugUtilsMessenger`.
#[derive(Clone, Debug)]
pub struct DebugUtilsMessengerCreateInfo<'a> {
    /// The message severity types that the callback should be called for.
    ///
    /// The value must not be empty.
    ///
    /// The default value is `MessageSeverity::errors_and_warnings()`.
    pub message_severity: DebugUtilsMessageSeverity,

    /// The message types that the callback should be called for.
    ///
    /// The value must not be empty.
    ///
    /// The default value is `MessageType::general()`.
    pub message_type: DebugUtilsMessageType,

    /// The closure that should be called.
    ///
    /// The closure must not make any calls to the Vulkan API.
    /// If the closure panics, the panic is caught and ignored.
    pub user_callback: &'a Arc<DebugUtilsMessengerCallback>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> DebugUtilsMessengerCreateInfo<'a> {
    /// Returns a default `DebugUtilsMessengerCreateInfo` with the provided `user_callback`.
    #[inline]
    pub const fn new(user_callback: &'a Arc<DebugUtilsMessengerCallback>) -> Self {
        Self {
            message_severity: DebugUtilsMessageSeverity::ERROR
                .union(DebugUtilsMessageSeverity::WARNING),
            message_type: DebugUtilsMessageType::GENERAL,
            user_callback,
            _ne: crate::NE,
        }
    }

    #[inline]
    pub(crate) fn validate(&self, instance: &Instance) -> Result<(), Box<ValidationError>> {
        self.validate_raw(instance.api_version(), instance.enabled_extensions())
    }

    pub(crate) fn validate_raw(
        &self,
        instance_api_version: Version,
        instance_extensions: &InstanceExtensions,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            message_severity,
            message_type,
            user_callback: _,
            _ne: _,
        } = self;

        message_severity
            .validate_instance_raw(instance_api_version, instance_extensions)
            .map_err(|err| {
                err.add_context("message_severity").set_vuids(&[
                    "VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-parameter",
                ])
            })?;

        if message_severity.is_empty() {
            return Err(Box::new(ValidationError {
                context: "message_severity".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask"],
                ..Default::default()
            }));
        }

        message_type
            .validate_instance_raw(instance_api_version, instance_extensions)
            .map_err(|err| {
                err.add_context("message_type")
                    .set_vuids(&["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter"])
            })?;

        if message_type.is_empty() {
            return Err(Box::new(ValidationError {
                context: "message_type".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask"],
                ..Default::default()
            }));
        }

        // VUID-PFN_vkDebugUtilsMessengerCallbackEXT-None-04769
        // Can't be checked, creation is unsafe.

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        let &Self {
            message_type,
            message_severity,
            user_callback,
            _ne: _,
        } = self;

        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(message_severity.into())
            .message_type(message_type.into())
            .pfn_user_callback(Some(trampoline))
            .user_data(user_callback.as_ptr().cast_mut().cast())
    }
}

/// The callback function for debug messages.
pub struct DebugUtilsMessengerCallback(CallbackData);

type CallbackData = Box<
    dyn Fn(DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallbackData<'_>)
        + RefUnwindSafe
        + Send
        + Sync,
>;

impl DebugUtilsMessengerCallback {
    /// Returns a new `DebugUtilsMessengerCallback` wrapping the provided function.
    ///
    /// # Safety
    ///
    /// - `func` must not make any calls to the Vulkan API.
    pub unsafe fn new(
        func: impl Fn(
                DebugUtilsMessageSeverity,
                DebugUtilsMessageType,
                DebugUtilsMessengerCallbackData<'_>,
            ) + RefUnwindSafe
            + Send
            + Sync
            + 'static,
    ) -> Arc<Self> {
        Arc::new(Self(Box::new(func)))
    }

    pub(crate) fn as_ptr(&self) -> *const CallbackData {
        &raw const self.0
    }
}

impl Debug for DebugUtilsMessengerCallback {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("DebugUtilsMessengerCallback")
            .finish_non_exhaustive()
    }
}

pub(super) unsafe extern "system" fn trampoline(
    message_severity_vk: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types_vk: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data_vk: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data_vk: *mut c_void,
) -> vk::Bool32 {
    // Since we box the closure, the type system doesn't detect that the `UnwindSafe`
    // bound is enforced. Therefore we enforce it manually.
    let _ = catch_unwind(AssertUnwindSafe(move || {
        let vk::DebugUtilsMessengerCallbackDataEXT {
            flags: _,
            p_message_id_name,
            message_id_number,
            p_message,
            queue_label_count,
            p_queue_labels,
            cmd_buf_label_count,
            p_cmd_buf_labels,
            object_count,
            p_objects,
            ..
        } = unsafe { *callback_data_vk };

        let callback_data = DebugUtilsMessengerCallbackData {
            message_id_name: unsafe { p_message_id_name.as_ref() }.map(|p_message_id_name| {
                unsafe { CStr::from_ptr(p_message_id_name) }
                    .to_str()
                    .unwrap()
            }),
            message_id_number,
            message: unsafe { CStr::from_ptr(p_message) }.to_str().unwrap(),
            queue_labels: DebugUtilsMessengerCallbackLabelIter(
                // Some drivers give a null pointer for empty data.
                if p_queue_labels.is_null() {
                    &[]
                } else {
                    unsafe { slice::from_raw_parts(p_queue_labels, queue_label_count as usize) }
                }
                .iter(),
            ),
            cmd_buf_labels: DebugUtilsMessengerCallbackLabelIter(
                if p_cmd_buf_labels.is_null() {
                    &[]
                } else {
                    unsafe { slice::from_raw_parts(p_cmd_buf_labels, cmd_buf_label_count as usize) }
                }
                .iter(),
            ),
            objects: DebugUtilsMessengerCallbackObjectNameInfoIter(
                if p_objects.is_null() {
                    &[]
                } else {
                    unsafe { slice::from_raw_parts(p_objects, object_count as usize) }
                }
                .iter(),
            ),
        };

        let user_callback: &CallbackData = unsafe { &*user_data_vk.cast_const().cast() };

        user_callback(
            message_severity_vk.into(),
            message_types_vk.into(),
            callback_data,
        );
    }));

    vk::FALSE
}

/// The data of a message received by the user callback.
#[non_exhaustive]
pub struct DebugUtilsMessengerCallbackData<'a> {
    /// The particular message ID that is associated with the provided message.
    ///
    /// If the message is from a validation layer, then this may specify the part of the Vulkan
    /// specification that was validated.
    pub message_id_name: Option<&'a str>,

    /// The ID number of the message.
    pub message_id_number: i32,

    /// The message detailing the conditions.
    pub message: &'a str,

    /// Labels that were active in the current queue when the callback was triggered.
    pub queue_labels: DebugUtilsMessengerCallbackLabelIter<'a>,

    /// Labels that were active in the current command buffer when the callback was triggered.
    pub cmd_buf_labels: DebugUtilsMessengerCallbackLabelIter<'a>,

    /// Objects related to the message.
    pub objects: DebugUtilsMessengerCallbackObjectNameInfoIter<'a>,
}

/// The values of [`DebugUtilsLabel`], as returned to a messenger callback.
#[non_exhaustive]
pub struct DebugUtilsMessengerCallbackLabel<'a> {
    /// The name of the label.
    pub label_name: &'a str,

    /// An RGBA color value that is associated with the label, with values in the range
    /// `0.0..=1.0`.
    pub color: &'a [f32; 4],
}

#[derive(Clone, Debug)]
pub struct DebugUtilsMessengerCallbackLabelIter<'a>(slice::Iter<'a, vk::DebugUtilsLabelEXT<'a>>);

impl<'a> Iterator for DebugUtilsMessengerCallbackLabelIter<'a> {
    type Item = DebugUtilsMessengerCallbackLabel<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|label| DebugUtilsMessengerCallbackLabel {
            label_name: unsafe { label.label_name_as_c_str() }
                .unwrap()
                .to_str()
                .unwrap(),
            color: &label.color,
        })
    }
}

/// An object that triggered a callback.
#[non_exhaustive]
pub struct DebugUtilsMessengerCallbackObjectNameInfo<'a> {
    /// The type of object.
    pub object_type: vk::ObjectType,

    /// The handle of the object.
    pub object_handle: u64,

    /// The name of the object, if previously set.
    pub object_name: Option<&'a str>,
}

#[derive(Clone, Debug)]
pub struct DebugUtilsMessengerCallbackObjectNameInfoIter<'a>(
    slice::Iter<'a, vk::DebugUtilsObjectNameInfoEXT<'a>>,
);

impl<'a> Iterator for DebugUtilsMessengerCallbackObjectNameInfoIter<'a> {
    type Item = DebugUtilsMessengerCallbackObjectNameInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|info| {
            let &vk::DebugUtilsObjectNameInfoEXT {
                object_type,
                object_handle,
                p_object_name,
                ..
            } = info;

            DebugUtilsMessengerCallbackObjectNameInfo {
                object_type,
                object_handle,
                object_name: unsafe { p_object_name.as_ref() }.map(|p_object_name| {
                    unsafe { CStr::from_ptr(p_object_name) }.to_str().unwrap()
                }),
            }
        })
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Severity of message.
    DebugUtilsMessageSeverity = DebugUtilsMessageSeverityFlagsEXT(u32);

    /// An error that may cause undefined results, including an application crash.
    ERROR = ERROR,

    /// An unexpected use.
    WARNING = WARNING,

    /// An informational message that may be handy when debugging an application.
    INFO = INFO,

    /// Diagnostic information from the loader and layers.
    VERBOSE = VERBOSE,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Type of message.
    DebugUtilsMessageType = DebugUtilsMessageTypeFlagsEXT(u32);

    /// Specifies that some general event has occurred.
    GENERAL = GENERAL,

    /// Specifies that something has occurred during validation against the Vulkan specification
    VALIDATION = VALIDATION,

    /// Specifies a potentially non-optimal use of Vulkan
    PERFORMANCE = PERFORMANCE,
}

/// A label to associate with a span of work in a queue.
///
/// When debugging, labels can be useful to identify which queue, or where in a specific queue,
/// something happened.
#[derive(Clone, Debug)]
pub struct DebugUtilsLabel {
    /// The name of the label.
    ///
    /// The default value is empty.
    pub label_name: String,

    /// An RGBA color value that is associated with the label, with values in the range
    /// `0.0..=1.0`.
    ///
    /// If set to `[0.0; 4]`, the value is ignored.
    ///
    /// The default value is `[0.0; 4]`.
    pub color: [f32; 4],

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for DebugUtilsLabel {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DebugUtilsLabel {
    /// Returns a default `DebugUtilsLabel`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            label_name: String::new(),
            color: [0.0; 4],
            _ne: crate::NE,
        }
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a DebugUtilsLabelFields1Vk,
    ) -> vk::DebugUtilsLabelEXT<'a> {
        let &Self {
            label_name: _,
            color,
            _ne,
        } = self;

        let DebugUtilsLabelFields1Vk { label_name_vk } = fields1_vk;

        vk::DebugUtilsLabelEXT::default()
            .label_name(label_name_vk)
            .color(color)
    }

    pub(crate) fn to_vk_fields1(&self) -> DebugUtilsLabelFields1Vk {
        let label_name_vk = CString::new(self.label_name.as_str()).unwrap();

        DebugUtilsLabelFields1Vk { label_name_vk }
    }
}

pub(crate) struct DebugUtilsLabelFields1Vk {
    pub(crate) label_name_vk: CString,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Features of the validation layer to enable.
    ValidationFeatureEnable = ValidationFeatureEnableEXT(i32);

    /// The validation layer will use shader programs running on the GPU to provide additional
    /// validation.
    ///
    /// This must not be used together with `DebugPrintf`.
    GpuAssisted = GPU_ASSISTED,

    /// The validation layer will reserve and use one descriptor set slot for its own use.
    /// The limit reported by
    /// [`max_bound_descriptor_sets`](crate::device::DeviceProperties::max_bound_descriptor_sets)
    /// will be reduced by 1.
    ///
    /// `GpuAssisted` must also be enabled.
    GpuAssistedReserveBindingSlot = GPU_ASSISTED_RESERVE_BINDING_SLOT,

    /// The validation layer will report recommendations that are not strictly errors,
    /// but that may be considered good Vulkan practice.
    BestPractices = BEST_PRACTICES,

    /// The validation layer will process `debugPrintfEXT` operations in shaders, and send them
    /// to the debug callback.
    ///
    /// This must not be used together with `GpuAssisted`.
    DebugPrintf = DEBUG_PRINTF,

    /// The validation layer will report errors relating to synchronization, such as data races and
    /// the use of synchronization primitives.
    SynchronizationValidation = SYNCHRONIZATION_VALIDATION,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Features of the validation layer to disable.
    ValidationFeatureDisable = ValidationFeatureDisableEXT(i32);

    /// All validation is disabled.
    All = ALL,

    /// Shader validation is disabled.
    Shaders = SHADERS,

    /// Thread safety validation is disabled.
    ThreadSafety = THREAD_SAFETY,

    /// Stateless parameter validation is disabled.
    ApiParameters = API_PARAMETERS,

    /// Object lifetime validation is disabled.
    ObjectLifetimes = OBJECT_LIFETIMES,

    /// Core validation checks are disabled.
    ///
    /// This also disables shader validation and GPU-assisted validation.
    CoreChecks = CORE_CHECKS,

    /// Protection against duplicate non-dispatchable handles is disabled.
    UniqueHandles = UNIQUE_HANDLES,

    /// Results of shader validation will not be cached, and are validated from scratch each time.
    ShaderValidationCache = SHADER_VALIDATION_CACHE,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instance::{InstanceCreateInfo, InstanceExtensions},
        VulkanLibrary,
    };
    use std::thread;

    #[test]
    fn ensure_sendable() {
        // It's useful to be able to initialize a DebugUtilsMessenger on one thread
        // and keep it alive on another thread.
        let instance = {
            let library = match unsafe { VulkanLibrary::new() } {
                Ok(x) => x,
                Err(_) => return,
            };

            match Instance::try_new(
                &library,
                &InstanceCreateInfo {
                    enabled_extensions: &InstanceExtensions {
                        ext_debug_utils: true,
                        ..InstanceExtensions::empty()
                    },
                    ..Default::default()
                },
            ) {
                Ok(x) => x,
                Err(_) => return,
            }
        };

        let callback = DebugUtilsMessenger::new(
            &instance,
            &DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity::ERROR,
                message_type: DebugUtilsMessageType::GENERAL
                    | DebugUtilsMessageType::VALIDATION
                    | DebugUtilsMessageType::PERFORMANCE,
                ..DebugUtilsMessengerCreateInfo::new(&unsafe {
                    DebugUtilsMessengerCallback::new(|_, _, _| {})
                })
            },
        )
        .unwrap();
        thread::spawn(move || {
            drop(callback);
        });
    }
}
