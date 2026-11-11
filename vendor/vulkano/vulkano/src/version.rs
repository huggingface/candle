// The `Version` object is reexported from the `instance` module.

use ash::vk;
use std::{
    fmt::{Debug, Display, Error as FmtError, Formatter},
    num::ParseIntError,
    str::FromStr,
};

include!(crate::autogen_output!("version.rs"));

/// Represents an API version of Vulkan.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
    /// Patch version number.
    pub patch: u32,
}

impl Version {
    pub const V1_0: Version = Version::major_minor(1, 0);
    pub const V1_1: Version = Version::major_minor(1, 1);
    pub const V1_2: Version = Version::major_minor(1, 2);
    pub const V1_3: Version = Version::major_minor(1, 3);
    pub const V1_4: Version = Version::major_minor(1, 4);
    pub const V1_5: Version = Version::major_minor(1, 5);
    pub const V1_6: Version = Version::major_minor(1, 6);

    /// Constructs a `Version` from the given major and minor version numbers.
    #[inline]
    pub const fn major_minor(major: u32, minor: u32) -> Version {
        Version {
            major,
            minor,
            patch: 0,
        }
    }
}

impl Default for Version {
    #[inline]
    fn default() -> Self {
        Self::V1_0
    }
}

impl From<u32> for Version {
    #[inline]
    fn from(val: u32) -> Self {
        Version {
            major: vk::api_version_major(val),
            minor: vk::api_version_minor(val),
            patch: vk::api_version_patch(val),
        }
    }
}

impl TryFrom<Version> for u32 {
    type Error = ();

    #[inline]
    fn try_from(val: Version) -> Result<Self, Self::Error> {
        if val.major <= 0x3ff && val.minor <= 0x3ff && val.patch <= 0xfff {
            Ok(vk::make_api_version(0, val.major, val.minor, val.patch))
        } else {
            Err(())
        }
    }
}

impl FromStr for Version {
    type Err = ParseIntError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.splitn(3, '.');
        let major: u32 = iter.next().unwrap().parse()?;
        let minor: u32 = iter.next().map_or(Ok(0), |n| n.parse())?;
        let patch: u32 = iter.next().map_or(Ok(0), |n| n.parse())?;

        Ok(Version {
            major,
            minor,
            patch,
        })
    }
}

impl Debug for Version {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Display for Version {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::Version;

    #[test]
    fn into_vk_version() {
        let version = Version {
            major: 1,
            minor: 0,
            patch: 0,
        };
        assert_eq!(u32::try_from(version).unwrap(), 0x400000);
    }

    #[test]
    fn greater_major() {
        let v1 = Version {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let v2 = Version {
            major: 2,
            minor: 0,
            patch: 0,
        };
        assert!(v2 > v1);
    }

    #[test]
    fn greater_minor() {
        let v1 = Version {
            major: 1,
            minor: 1,
            patch: 0,
        };
        let v2 = Version {
            major: 1,
            minor: 3,
            patch: 0,
        };
        assert!(v2 > v1);
    }

    #[test]
    fn greater_patch() {
        let v1 = Version {
            major: 1,
            minor: 0,
            patch: 4,
        };
        let v2 = Version {
            major: 1,
            minor: 0,
            patch: 5,
        };
        assert!(v2 > v1);
    }

    #[test]
    fn version_parse() {
        assert!(matches!(
            "1.1.1".parse::<Version>(),
            Ok(Version {
                major: 1,
                minor: 1,
                patch: 1,
            })
        ));
        assert!(matches!(
            "1.1".parse::<Version>(),
            Ok(Version {
                major: 1,
                minor: 1,
                patch: 0,
            })
        ));
        assert!(matches!(
            "1".parse::<Version>(),
            Ok(Version {
                major: 1,
                minor: 0,
                patch: 0,
            })
        ));

        assert!("".parse::<Version>().is_err());
        assert!("1.1.1.1".parse::<Version>().is_err());
        assert!("foobar".parse::<Version>().is_err());
        assert!("1.bar".parse::<Version>().is_err());
    }
}
