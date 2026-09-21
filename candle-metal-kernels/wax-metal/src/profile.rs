//! The target profile.
//! Target triple, AIR / Metal-language versions, the resource limits in `!llvm.module.flags`,
//! and the platform fields in the `.metallib` container header.
//!
//! Resolved at build time from Cargo's target triple.
//! Measured on toolchain 32023.864 (Xcode SDK 26.2). The tests in this file pin those
//! measurements.
//!
//! A relatively small table suffices because the datalayout and the `air.max_*` limits are
//! identical everywhere measured. The AIR version, language version and container format code
//! are a function of the deployment target, not the SDK or GPU. And the platform appears only in
//! the triple's OS name and two container-header fields.

/// The Apple platform the AIR is built for.
///
/// Only the OS name in the triple and the container's platform fields differ between
/// these. The emitted code does not. Apple Silicon only. The SGMMA and SIMD-reduce
/// intrinsics the lowering emits are Apple-GPU features, so Intel Macs are out of scope.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Platform {
    MacOS,
    IOS,
    IOSSimulator,
    TvOS,
    TvOSSimulator,
    WatchOS,
    WatchOSSimulator,
    VisionOS,
    VisionOSSimulator,
}

use Platform::*;

impl Platform {
    /// The OS component of the target triple (`air64_v28-apple-`**`ios`**`26.0.0`).
    pub fn os_name(self) -> &'static str {
        match self {
            MacOS => "macosx",
            IOS | IOSSimulator => "ios",
            TvOS | TvOSSimulator => "tvos",
            WatchOS | WatchOSSimulator => "watchos",
            VisionOS | VisionOSSimulator => "xros",
        }
    }

    /// Simulator targets carry a `-simulator` triple suffix and their own container tag.
    pub fn is_simulator(self) -> bool {
        matches!(
            self,
            IOSSimulator | TvOSSimulator | WatchOSSimulator | VisionOSSimulator
        )
    }

    /// Whether the platform numbers its releases like iOS (iOS/tvOS: 15, 16, 17, 18, 26) or
    /// like macOS (12, 13, 14, 15, 26). watchOS and visionOS number differently again and
    /// were only measured at 26, so they read the newest era. See
    /// [`TargetProfile::with_versions`] to pin an older one by hand.
    fn numbers_like_ios(self) -> bool {
        !matches!(self, MacOS)
    }

    /// The container header's platform byte (offset 0x0B).
    fn container_tag(self) -> u8 {
        match self {
            MacOS => 0x81,
            IOS => 0x82,
            TvOS => 0x83,
            WatchOS => 0x84,
            IOSSimulator => 0x87,
            TvOSSimulator => 0x88,
            WatchOSSimulator => 0x89,
            VisionOS => 0x8b,
            VisionOSSimulator => 0x8c,
        }
    }

    /// The container header's first word (offset 0x04). macOS sets the high bit. Every other
    /// platform measured leaves it clear. Purpose unknown. Copied from `xcrun metallib`
    /// output, which is the only thing the loader agrees with.
    fn container_kind(self) -> u16 {
        match self {
            MacOS => 0x8001,
            _ => 0x0001,
        }
    }

    /// The platform this crate is being compiled for, or `None` when the host is not an
    /// Apple platform (a cross-build of the emitter itself: it still emits AIR, it just
    /// has no local Metal to run it on).
    pub fn host() -> Option<Self> {
        // `target_abi = "sim"` is what distinguishes aarch64-apple-ios-sim from
        // aarch64-apple-ios. Both report target_os = "ios".
        let sim = cfg!(target_abi = "sim");
        Some(match () {
            _ if cfg!(target_os = "macos") => MacOS,
            _ if cfg!(target_os = "ios") && sim => IOSSimulator,
            _ if cfg!(target_os = "ios") => IOS,
            _ if cfg!(target_os = "tvos") && sim => TvOSSimulator,
            _ if cfg!(target_os = "tvos") => TvOS,
            _ if cfg!(target_os = "watchos") && sim => WatchOSSimulator,
            _ if cfg!(target_os = "watchos") => WatchOS,
            _ if cfg!(target_os = "visionos") && sim => VisionOSSimulator,
            _ if cfg!(target_os = "visionos") => VisionOS,
            _ => return None,
        })
    }
}

/// One Metal release, and the AIR it wants. Selected by deployment target: a kernel
/// built for macOS 14 must declare AIR 2.6, because a macOS 14 Metal runtime will
/// reject a library that claims 2.8.
struct Era {
    /// First macOS major version in this era.
    macos: u32,
    /// First iOS/tvOS major version in this era.
    ios: u32,
    /// `!air.version` minor. The major has been 2 throughout.
    air_minor: u32,
    /// `!air.language_version` major.minor.
    lang: (u32, u32),
    /// The container header's format code (offset 0x08).
    container_code: u16,
}

/// Measured, oldest first. Note that AIR 2.5 and 2.6 share container code 7, so the
/// code is its own column rather than something derived from the AIR version.
const ERAS: &[Era] = &[
    Era {
        macos: 12,
        ios: 15,
        air_minor: 4,
        lang: (2, 4),
        container_code: 6,
    },
    Era {
        macos: 13,
        ios: 16,
        air_minor: 5,
        lang: (3, 0),
        container_code: 7,
    },
    Era {
        macos: 14,
        ios: 17,
        air_minor: 6,
        lang: (3, 1),
        container_code: 7,
    },
    Era {
        macos: 15,
        ios: 18,
        air_minor: 7,
        lang: (3, 2),
        container_code: 8,
    },
    Era {
        macos: 26,
        ios: 26,
        air_minor: 8,
        lang: (4, 0),
        container_code: 9,
    },
];

/// Deployment target used when the build does not name one: the newest era, which is
/// what the toolchain itself defaults to.
const DEFAULT_OS: (u32, u32) = (26, 0);

/// Identical on every platform and deployment target measured, so it is a constant
/// rather than a profile field.
pub const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64\
-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128\
-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// Resource limits, emitted as `air.max_*` module flags. Device-family dependent in
/// principle. Identical across everything measured here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    pub device_buffers: i64,
    pub constant_buffers: i64,
    pub threadgroup_buffers: i64,
    pub textures: i64,
    pub read_write_textures: i64,
    pub samplers: i64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            device_buffers: 31,
            constant_buffers: 31,
            threadgroup_buffers: 31,
            textures: 128,
            read_write_textures: 8,
            samplers: 16,
        }
    }
}

impl Limits {
    /// The flags in the order Apple's frontend emits them.
    pub fn flags(&self) -> [(&'static str, i64); 6] {
        [
            ("air.max_device_buffers", self.device_buffers),
            ("air.max_constant_buffers", self.constant_buffers),
            ("air.max_threadgroup_buffers", self.threadgroup_buffers),
            ("air.max_textures", self.textures),
            ("air.max_read_write_textures", self.read_write_textures),
            ("air.max_samplers", self.samplers),
        ]
    }
}

/// Everything target-specific the emitter and the container writer need.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TargetProfile {
    pub platform: Platform,
    /// Deployment target, major.minor. Drives the triple, the versions and the
    /// container's OS fields.
    pub os: (u32, u32),
    /// `!air.version`.
    pub air_version: (u32, u32, u32),
    /// `!air.language_version`, after the leading `"Metal"`.
    pub language_version: (u32, u32, u32),
    pub limits: Limits,
    container_code: u16,
}

impl TargetProfile {
    /// The profile for `platform` at deployment target `os_major.os_minor`.
    ///
    /// A version newer than anything in the table reads as the newest era, which is the
    /// useful direction to guess in: a new OS keeps loading the previous era's AIR.
    pub fn new(platform: Platform, os_major: u32, os_minor: u32) -> Self {
        let key = |e: &Era| {
            if platform.numbers_like_ios() {
                e.ios
            } else {
                e.macos
            }
        };
        let era = ERAS
            .iter()
            .rev()
            .find(|e| os_major >= key(e))
            .unwrap_or(&ERAS[0]);
        Self {
            platform,
            os: (os_major, os_minor),
            air_version: (2, era.air_minor, 0),
            language_version: (era.lang.0, era.lang.1, 0),
            limits: Limits::default(),
            container_code: era.container_code,
        }
    }

    /// Override the versions the era table would pick. For a platform whose release numbering
    /// is not in the table, or to reproduce a specific toolchain's output. The container's
    /// format code follows the AIR version.
    pub fn with_versions(
        mut self,
        air_version: (u32, u32, u32),
        language_version: (u32, u32, u32),
    ) -> Self {
        self.air_version = air_version;
        self.language_version = language_version;
        self.container_code = ERAS
            .iter()
            .find(|e| e.air_minor == air_version.1)
            .map(|e| e.container_code)
            .unwrap_or(self.container_code);
        self
    }

    /// The profile for the platform this crate was compiled for.
    ///
    /// The deployment target comes from `METAL_TILE_DEPLOYMENT_TARGET` ("18.0") or the
    /// platform's standard Apple variable (`IPHONEOS_DEPLOYMENT_TARGET` and friends) as they
    /// were set when this crate was compiled (see `build.rs`), else the newest era. Set one
    /// when shipping to an older OS, whose Metal runtime will refuse a library claiming a
    /// newer AIR version than it knows.
    ///
    /// Falls back to macOS when the host is not an Apple platform, so that a cross-build of
    /// the emitter still produces loadable AIR for the obvious target.
    pub fn host() -> Self {
        let platform = Platform::host().unwrap_or(MacOS);
        let (major, minor) = option_env!("METAL_TILE_DEPLOYMENT_TARGET")
            .and_then(parse_os_version)
            .unwrap_or(DEFAULT_OS);
        Self::new(platform, major, minor)
    }

    /// e.g. `air64_v28-apple-macosx26.0.0`, `air64_v28-apple-ios26.0.0-simulator`.
    pub fn triple(&self) -> String {
        format!(
            "air64_v{}-apple-{}{}.{}.0{}",
            self.air_tag(),
            self.platform.os_name(),
            self.os.0,
            self.os.1,
            if self.platform.is_simulator() {
                "-simulator"
            } else {
                ""
            }
        )
    }

    /// The `v28` in the triple: the AIR version with the dot removed.
    fn air_tag(&self) -> u32 {
        self.air_version.0 * 10 + self.air_version.1
    }

    pub fn datalayout(&self) -> &'static str {
        DATALAYOUT
    }

    /// The `SDK Version` module flag. Informational. The bitcode path omits the flag
    /// entirely and Metal loads those libraries fine, so it reports the deployment target
    /// rather than pretending to know which SDK is installed on the machine that will run
    /// the emitter.
    pub fn sdk_version(&self) -> (u32, u32) {
        self.os
    }

    /// The 12 bytes at offset 0x04 of a `.metallib` header: format word, constant 2,
    /// format code, platform tag, and the deployment target.
    pub(crate) fn container_header(&self) -> [u8; 12] {
        let mut h = [0u8; 12];
        h[0..2].copy_from_slice(&self.platform.container_kind().to_le_bytes());
        h[2..4].copy_from_slice(&2u16.to_le_bytes());
        h[4..6].copy_from_slice(&self.container_code.to_le_bytes());
        h[6] = 0;
        h[7] = self.platform.container_tag();
        h[8..10].copy_from_slice(&(self.os.0 as u16).to_le_bytes());
        h[10..12].copy_from_slice(&(self.os.1 as u16).to_le_bytes());
        h
    }
}

impl Default for TargetProfile {
    fn default() -> Self {
        Self::host()
    }
}

/// "26" or "26.2" -> (26, 2). Anything else is ignored rather than fatal: a typo in an
/// env var should not stop the emitter from producing loadable AIR.
fn parse_os_version(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().map_or(Some(0), |m| m.parse().ok())?;
    Some((major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Triples measured with `xcrun -sdk <sdk> metal -S`, one per era and platform.
    #[test]
    fn triples_match_the_toolchain() {
        let cases: &[(Platform, u32, u32, &str)] = &[
            (MacOS, 12, 0, "air64_v24-apple-macosx12.0.0"),
            (MacOS, 13, 3, "air64_v25-apple-macosx13.3.0"),
            (MacOS, 14, 0, "air64_v26-apple-macosx14.0.0"),
            (MacOS, 15, 4, "air64_v27-apple-macosx15.4.0"),
            (MacOS, 26, 0, "air64_v28-apple-macosx26.0.0"),
            (IOS, 15, 0, "air64_v24-apple-ios15.0.0"),
            (IOS, 17, 0, "air64_v26-apple-ios17.0.0"),
            (IOS, 26, 2, "air64_v28-apple-ios26.2.0"),
            (IOSSimulator, 26, 2, "air64_v28-apple-ios26.2.0-simulator"),
            (TvOS, 26, 2, "air64_v28-apple-tvos26.2.0"),
            (WatchOS, 26, 2, "air64_v28-apple-watchos26.2.0"),
            (VisionOS, 26, 2, "air64_v28-apple-xros26.2.0"),
        ];
        for &(platform, major, minor, want) in cases {
            let p = TargetProfile::new(platform, major, minor);
            assert_eq!(p.triple(), want, "{platform:?} {major}.{minor}");
        }
    }

    /// Container headers measured with `xcrun metallib`, as the 12 bytes after the
    /// magic.
    #[test]
    fn container_headers_match_the_toolchain() {
        let cases: &[(Platform, u32, u32, [u8; 12])] = &[
            // macOS 26.0: 0180 0200 0900 0081 1a00 0000
            (
                MacOS,
                26,
                0,
                [0x01, 0x80, 2, 0, 9, 0, 0, 0x81, 0x1a, 0, 0, 0],
            ),
            // macOS 15.4: 0180 0200 0800 0081 0f00 0400
            (
                MacOS,
                15,
                4,
                [0x01, 0x80, 2, 0, 8, 0, 0, 0x81, 0x0f, 0, 4, 0],
            ),
            // macOS 12.0: 0180 0200 0600 0081 0c00 0000
            (
                MacOS,
                12,
                0,
                [0x01, 0x80, 2, 0, 6, 0, 0, 0x81, 0x0c, 0, 0, 0],
            ),
            // iOS 26.2: 0100 0200 0900 0082 1a00 0200
            (IOS, 26, 2, [0x01, 0, 2, 0, 9, 0, 0, 0x82, 0x1a, 0, 2, 0]),
            // iOS 17.0: 0100 0200 0700 0082 1100 0000
            (IOS, 17, 0, [0x01, 0, 2, 0, 7, 0, 0, 0x82, 0x11, 0, 0, 0]),
            // iOS 26.2 simulator: 0100 0200 0900 0087 1a00 0200
            (
                IOSSimulator,
                26,
                2,
                [0x01, 0, 2, 0, 9, 0, 0, 0x87, 0x1a, 0, 2, 0],
            ),
            // tvOS 26.2: 0100 0200 0900 0083 1a00 0200
            (TvOS, 26, 2, [0x01, 0, 2, 0, 9, 0, 0, 0x83, 0x1a, 0, 2, 0]),
            // visionOS 26.2: 0100 0200 0900 008b 1a00 0200
            (
                VisionOS,
                26,
                2,
                [0x01, 0, 2, 0, 9, 0, 0, 0x8b, 0x1a, 0, 2, 0],
            ),
        ];
        for &(platform, major, minor, want) in cases {
            let got = TargetProfile::new(platform, major, minor).container_header();
            assert_eq!(got, want, "{platform:?} {major}.{minor}");
        }
    }

    #[test]
    fn versions_follow_the_deployment_target() {
        let p = TargetProfile::new(MacOS, 14, 0);
        assert_eq!(p.air_version, (2, 6, 0));
        assert_eq!(p.language_version, (3, 1, 0));
        let p = TargetProfile::new(IOS, 18, 0);
        assert_eq!(p.air_version, (2, 7, 0));
        assert_eq!(p.language_version, (3, 2, 0));
        // A future OS reads as the newest era rather than the oldest.
        let p = TargetProfile::new(MacOS, 99, 0);
        assert_eq!(p.air_version, (2, 8, 0));
        // Older than anything measured: floor, not panic.
        let p = TargetProfile::new(MacOS, 10, 15);
        assert_eq!(p.air_version, (2, 4, 0));
    }

    #[test]
    fn os_version_parsing() {
        assert_eq!(parse_os_version("26.2"), Some((26, 2)));
        assert_eq!(parse_os_version(" 18 "), Some((18, 0)));
        assert_eq!(parse_os_version("18.0.1"), Some((18, 0)));
        assert_eq!(parse_os_version("junk"), None);
        assert_eq!(parse_os_version(""), None);
    }

    /// The host profile is what every default path uses, so its triple must be the one
    /// the local toolchain produces.
    #[test]
    fn host_profile_targets_the_build_platform() {
        let p = TargetProfile::host();
        assert_eq!(Some(p.platform), Platform::host().or(Some(MacOS)));
        assert!(p.triple().starts_with("air64_v"), "{}", p.triple());
    }
}
