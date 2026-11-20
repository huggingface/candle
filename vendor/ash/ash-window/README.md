## Ash-window

Interoperability between [`ash`](https://github.com/ash-rs/ash) and [`raw-window-handle`](https://github.com/rust-windowing/raw-window-handle) for surface creation.

[![Crates.io Version](https://img.shields.io/crates/v/ash-window.svg)](https://crates.io/crates/ash-window)
[![Documentation](https://docs.rs/ash-window/badge.svg)](https://docs.rs/ash-window)
[![Build Status](https://github.com/ash-rs/ash/workflows/CI/badge.svg)](https://github.com/ash-rs/ash/actions?workflow=CI)
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![LICENSE](https://img.shields.io/badge/license-apache-blue.svg)](LICENSE-APACHE)
[![Join the chat at https://gitter.im/MaikKlein/ash](https://badges.gitter.im/MaikKlein/ash.svg)](https://gitter.im/MaikKlein/ash?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![MSRV](https://img.shields.io/badge/rustc-1.69.0+-ab6000.svg)](https://blog.rust-lang.org/2023/04/20/Rust-1.69.0.html)

## Usage

```toml
ash-window = "0.12.0"
```

The library exposes two functions:

- [`enumerate_required_extensions`] returns the required instance extensions needed for surface creation from a specific display handle.

- [`create_surface`] allows to create a surface from a type implementing [`RawDisplayHandle`] and [`RawWindowHandle`]:

  ```rust
  ash_window::create_surface(&entry, &instance, &window, None)?;
  ```

[`enumerate_required_extensions`]: https://docs.rs/ash-window/latest/ash_window/fn.enumerate_required_extensions.html
[`create_surface`]: https://docs.rs/ash-window/latest/ash_window/fn.create_surface.html
[`RawDisplayHandle`]: https://docs.rs/raw-window-handle/latest/raw_window_handle/enum.RawDisplayHandle.html
[`RawWindowHandle`]: https://docs.rs/raw-window-handle/latest/raw_window_handle/enum.RawWindowHandle.html

## Versions

```toml
ash = "0.37"
```

## Support

- [x] Windows (`VK_KHR_win32_surface`)
- [x] Unix (`VK_KHR_xlib_surface`/`VK_KHR_xcb_surface`/`VK_KHR_wayland_surface`)
- [x] MacOS/IOS (`VK_EXT_metal_surface`)
- [x] Android (`VK_KHR_android_surface`)

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any Contribution intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
