<img align="left" alt="" src="logo.png" height="150" />

# [Vulkano](https://vulkano.rs)

[![Build Status](https://github.com/vulkano-rs/vulkano/workflows/Rust/badge.svg)](https://github.com/vulkano-rs/vulkano/actions?query=workflow%3ARust)
[![Discord](https://img.shields.io/discord/937149253296476201?label=discord)](https://discord.gg/xjHzXQp8hw)
[![Website/guide](https://img.shields.io/badge/-website/guide-%23555.svg)](https://vulkano.rs/)
<br/>
[![vulkano crates.io](https://img.shields.io/crates/v/vulkano?label=vulkano)](https://crates.io/crates/vulkano)
[![vulkano-shaders crates.io](https://img.shields.io/crates/v/vulkano-shaders?label=shaders)](https://crates.io/crates/vulkano-shaders)
[![vulkano-taskgraph crates.io](https://img.shields.io/crates/v/vulkano-taskgraph?label=taskgraph)](https://crates.io/crates/vulkano-taskgraph)
[![vulkano-util crates.io](https://img.shields.io/crates/v/vulkano-util?label=util)](https://crates.io/crates/vulkano-util)
<br/>
[![vulkano docs](https://img.shields.io/docsrs/vulkano?label=vulkano%20docs)](https://docs.rs/vulkano)
[![vulkano-shaders docs](https://img.shields.io/docsrs/vulkano-shaders?label=shaders%20docs)](https://docs.rs/vulkano-shaders)
[![vulkano-taskgraph docs](https://img.shields.io/docsrs/vulkano-taskgraph?label=taskgraph%20docs)](https://docs.rs/vulkano-taskgraph)
[![vulkano-util docs](https://img.shields.io/docsrs/vulkano-util?label=util%20docs)](https://docs.rs/vulkano-util)

Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.khronos.org/vulkan/).
It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't
be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code
should always conform to valid API usage.

What does vulkano do?

- Provides a low-levelish API around Vulkan. It doesn't hide what it does but provides some
  comfort types.
- Plans to prevent all invalid API usages, even the most obscure ones. The purpose of Vulkano
  is not to simply let you draw a teapot, but to cover all possible usages of Vulkan and detect all
  the possible problems in order to write robust programs. Invalid API usage is prevented thanks to
  both compile-time checks and runtime checks.
- Can handle synchronization on the GPU side for you (unless you choose to do that yourself), as this
  aspect of Vulkan is both annoying to handle and error-prone. Dependencies between submissions are
  automatically detected, and semaphores are managed automatically. The behavior of the library can
  be customized thanks to unsafe trait implementations.
- Tries to be convenient to use. Nobody is going to use a library that requires you to browse
  the documentation for hours for every single operation.

### Comparison

Comparison to other well-known Graphics APIs in Rust ecosystem.

| Name | Open-sourced Since | API Level | Notable Features |
| ---- | ------------------ | --------- | ------------- |
| Vulkano | March, 2016 | High-level Rust API wrapping Vulkan APIs. | Type-safe compile-time shaders. Transparent interoperability with glsl and spir-v shader code types in Rust code. Automatically generated types for shader's Layout. |
| [Wgpu](https://github.com/gfx-rs/wgpu) | May, 2019 | High-level Rust API with multiple backends. | Supports multiple backends: Vulkan, Metal, DirectX, WebGPU, and other. Follows WebGPU specification. With async/await API. |
| [Miniquad](https://github.com/not-fl3/miniquad) | March, 2020 | High-level minimalistic Rust API with multiple backends. | Relatively minimalistic API well suited for small to medium graphics projects. Supports multiple backends, including browser target. |
| [Sierra](https://github.com/zakarumych/sierra) | March, 2021 | High-level Rust API for Vulkan/Metal APIs. | Layouts, Descriptors and shader Types construction in Rust code through the macro system. Built on top of [Erupt](https://gitlab.com/Friz64/erupt). Supports Ray Tracing Pipeline. |
| [Glium](https://github.com/glium/glium) | October, 2014 | High-level Rust API wrapping OpenGL | OpenGL only. |
| [Ash](https://github.com/MaikKlein/ash) | August, 2016 | Low-level API for Vulkan. | Unsafe Vulkan API bindings. |
| [Erupt](https://gitlab.com/Friz64/erupt) | April, 2020 | Low-level API for Vulkan. | Unsafe Vulkan API bindings. |

Please note that by the current date none of the known projects in the ecosystem(including Vulkano)
reached stable release versions and the final design goals, their APIs are changing from time
to time in a breakable way too, and there could be bugs and unfinished features too.

However, most of the projects mentioned above are already established definitive structures, all
feature breaking changes will likely be straightforward to fix in user code, and most of them
are maintained. As such we can recommend using any of them in the 3rd party code.
The choice depends on the end project's goals and requirements, and we recommend examining
their actual set of features and API capabilities beforehand.

## Documentation and Resources

To get started you are encouraged to use the following resources:

- The [examples](https://github.com/vulkano-rs/vulkano/tree/master/examples) folder in this repository.
 - [docs.rs](https://docs.rs/vulkano) - Full Vulkano API documentation
 - The guide on [vulkano.rs](https://vulkano.rs/01-introduction/01-introduction.html) - Starts with trivial compute
   examples (~50 lines of code) and then works up to rendering triangles and mandelbrots.
   The guide is currently outdated a little. We are planning to update it in the future, but it's
   a good place to start understanding the base building blocks of Vulkano API.
 - GitHub [Issues](https://github.com/vulkano-rs/vulkano/issues) - Raise a topic, ask a question
   or report a bug. The new topics there are watched regularly by maintainers and other
   community users.
 - Gitter [Chat](https://gitter.im/vulkano-rs/Lobby) - Another place to raise a question. However,
   the chat is not maintained regularly at this moment. Better use GitHub Issues for this purpose.

## Contributing

Contributions are very welcome! Feel free to submit pull requests, to open questions and topics
in the [Issues](https://github.com/vulkano-rs/vulkano/issues) section.

The project was initially developed by Pierre Krieger(Tomaka), who established Vulkano's base
design goals, and the code structure. In the meantime, development is driven by Vulkano
community members.

If your change adds, removes or modifies a trait or a function, please
specify changelog entries **in the Pull Request description**(not in the changelog file directly).
They will be transferred to the changelog right after the PR merge.

Every PR must pass tests in order to be merged to `master`.

Minor releases are usually happening between 1 to 3 months averagely depending on grow
of unreleased and breaking changes in `master`

### Repository Structure

This repository contains four libraries:

- `vulkano` is the main one.
- `vulkano-shaders` provides the `shader!` macro for compiling glsl shaders.
- `vulkano-taskgraph` allows building a dependency graph of tasks, which are automatically
  synchronized and are then executed on the Vulkan device.
- `vulkano-util` provides a variety of utility functions to streamline certain common operations such as device and swapchain creation.
  a window to render to.

In order to run tests, run `cargo test --all` at the root of the repository. Make sure your Vulkan
driver is up to date before doing so.

### Hall of Fame

We would love to mention some members, who put significant contributions to this project:
- Pierre Krieger. The initial developer. [Patreon page](https://www.patreon.com/tomaka).
- Lucas Kent. Maintainer. [Patreon page](https://www.patreon.com/rukai).
- Austin Johnson. Maintainer. [Patreon page](https://www.patreon.com/austinj235).
- Rua. An active developer, who put a lot of effort to improve Vulkano and constantly keeping it up to date.
- **You!** Thanks to your help, contributions, improvements, bug reports and user experience
  to make this project one of the major Rust graphics API library in Rust!

## Setup and Troubleshooting

Vulkano uses [`glslc`](https://github.com/google/shaderc/tree/main/glslc) for shader compilation.
`glslc` must be installed and available on your `PATH`. It is bundled with the
[Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) or can be installed separately via your package manager.

Note that in general vulkano does **not** require you to install the official Vulkan SDK. This is
not something specific to vulkano (you don't need the SDK to write programs that use Vulkan, even
without vulkano), but many people are unaware of that and install the SDK thinking that it is
required. However, macOS, iOS and tvOS platforms do require a little more Vulkan setup since it is not
natively supported. See below for more details.

These requirements can be either installed with your favourite package manager or with installers
from the projects' websites. Below are some examples of ways to set up.

### windows-msvc Specific Setup

1. `rustup default stable-x86_64-pc-windows-msvc`
2. Install [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017). If you have already been using this toolchain then its probably already installed.
3.  Install [msys2](https://www.msys2.org/), following ALL of the instructions.
4.  Then in the msys2 terminal run: `pacman --noconfirm -Syu mingw-w64-x86_64-cmake mingw-w64-x86_64-python2 mingw-w64-x86_64-ninja`
5.  Add the msys2 mingw64 binary path to the PATH environment variable.

### Windows-gnu Specific Setup

windows-gnu toolchain is not supported but you can instead cross-compile to windows-gnu from windows-msvc

Steps 1 and 2 are to workaround https://github.com/rust-lang/rust/issues/49078 by using the same mingw that rust uses.

1.  Download and extract https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/x86_64-6.3.0-release-posix-seh-rt_v5-rev2.7z
2.  Add the absolute path to mingw64\bin to your PATH environment variable. (This path needs to be before the msys2 path)
3.  Run the command: `rustup default stable-x86_64-pc-windows-msvc`
4.  Run the command: `rustup target install x86_64-pc-windows-gnu`
5.  Install [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017). If you have already been using this toolchain then it's probably already installed.
6.  Install [msys2](https://www.msys2.org/), following ALL of the instructions.
7.  Then in the msys2 terminal run: `pacman --noconfirm -Syu mingw64/mingw-w64-x86_64-pkg-config mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make mingw-w64-x86_64-python2 mingw-w64-x86_64-ninja`
8.  Add the msys2 mingw64 binary path to the PATH environment variable.
9.  Any cargo command that builds the project needs to include `--target x86_64-pc-windows-gnu` e.g. to run: `cargo run --target x86_64-pc-windows-gnu`

### Linux Specific Setup

Use your package manager to install the required dev-tools and Vulkan drivers

For example on ubuntu:
```
sudo apt-get install build-essential git python cmake libvulkan-dev vulkan-tools
```
On arch based system
```
sudo pacman -Sy base-devel git python cmake vulkan-devel --noconfirm
```

### macOS, iOS and tvOS Specific Setup

Vulkan is not natively supported by macOS, iOS and tvOS. However, there exists [MoltenVK](https://github.com/KhronosGroup/MoltenVK)
an open-source Vulkan implementation on top of Apple's Metal API. This allows vulkano to build and run on macOS, iOS and tvOS platforms.

The easiest way to get vulkano up and running with MoltenVK is to install the
[Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home). There are [installation instructions](https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html) on the LunarG website.

On iOS and tvOS, vulkano links directly to the MoltenVK framework. There is nothing else to do besides
installing it. Note that the Vulkan SDK for macOS also comes with the framework for iOS and tvOS.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
