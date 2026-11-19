# Changelog

- Added support for `VK_OHOS_surface` (#1016)
- Keep loaded `Surface` extensions around via a new `SurfaceFactory` object on which `create_surface()` is called (#1019)

## [0.13.0] - 2024-03-31

- Bumped MSRV from 1.59 to 1.69 for `winit 0.28` and `raw-window-handle 0.5.1`, and `CStr::from_bytes_until_nul`. (#709, #716, #746)
- Bumped `raw-window-handle` to `0.6.0` (#799)
- Bumped `ash` version to [`0.38`](https://github.com/ash-rs/ash/releases/tag/0.38.0) (#897)

## [0.12.0] - 2022-09-23

### Changed

- Bumped `raw-window-handle` to `0.5.0`, now taking `RawDisplayHandle` and `RawWindowHandle` directly
  instead of requiring dynamic dispatch through the `HasRaw{Display,Window}Handle` traits (#645)

## [0.11.0] - 2022-07-29

### Changed

- Bumped `raw-window-handle` to `0.4.2` (#505)

## [0.10.0] - 2022-03-23

### Changed

- Bumped `ash` version to [`0.37`](https://github.com/ash-rs/ash/releases/tag/0.37.0) (#600)
- Make `enumerate_required_extensions()` return `&[*const c_char]` instead of `Vec<&CStr>` to match `ash::vk::InstanceCreateInfo` (#590)

## [0.9.1] - 2022-02-21

### Changed

- Convert `ash` version to a range, allowing [`0.34`](https://github.com/ash-rs/ash/releases/tag/0.34.0)-[`0.36`](https://github.com/ash-rs/ash/releases/tag/0.36.0) (#585)

## [0.9.0] - 2021-12-27

### Changed

- Bumped `ash` version to [`0.35`](https://github.com/ash-rs/ash/releases/tag/0.35.0)

## [0.8.0] - 2021-12-22

### Changed

- Bumped `ash` version to [`0.34`](https://github.com/ash-rs/ash/releases/tag/0.34.0)

## [0.7.0] - 2021-07-30

### Changed

- Bumped `ash` version to [`0.33`](https://github.com/ash-rs/ash/releases/tag/0.33.0)

## [0.6.0]

### Changed

- Bumped `ash` version to [`0.32`](https://github.com/ash-rs/ash/releases/tag/0.32.0)

## [0.5.0]

### Changed
- `impl HasRawWindowHandle` to `dyn HasRawWindowHandle`

## Version 0.4.1

### Changed
- Use `raw-window-metal` to automatically allocate a `CAMetalLayer` if there is none

## Version 0.4.0

### Changed
- Update `ash` version to 0.31

## Version 0.3.0

### Changed
- Update `ash` version to 0.30

## Version 0.2.0

### Changed
- `enumerate_required_extension` renamed to `enumerate_required_extensions`
- `enumerate_required_extensions` will return an error if the window handle is not supported instead of panic.
- `enumerate_required_extensions` may return multiple extension names. Includes all dependent extensions.
- `create_surface` will return an error if the window handle is not supported instead of panic.

## Version 0.1.0
Initial release for `raw-window-handle = "0.3"` with Windows, Linux, Android, MacOS/iOS support.

[Unreleased]: https://github.com/ash-rs/ash/compare/ash-window-0.13.0...HEAD
[0.12.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.13.0
[0.12.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.12.0
[0.11.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.11.0
[0.10.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.10.0
[0.9.1]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.9.1
[0.9.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.9.0
[0.8.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.8.0
[0.7.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.7.0
[0.6.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.6.0
[0.5.0]: https://github.com/ash-rs/ash/releases/tag/ash-window-0.5.0
