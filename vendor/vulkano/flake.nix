{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        vulkanDependencies = with pkgs; [
          shader-slang
          shaderc
          vulkan-loader
        ];
        vulkanTools = with pkgs; [
          spirv-tools
          vulkan-tools
          vulkan-tools-lunarg
          vulkan-validation-layers
        ];
        # (needed for the gl-interop example)
        openglDependencies = with pkgs; [
          libGL
        ];
        winitDependencies = with pkgs; [
          libx11
          libxcursor
          libxi
          libxkbcommon
          libxrandr
          wayland
        ];
        workflowDependencies = with pkgs; [
          cargo-hack
          typos
        ];
      in {
        devShells.default = with pkgs; mkShell rec {
          buildInputs = [
            (rust-bin.stable.latest.minimal.override {
              extensions = [ "clippy" "rust-analyzer" "rust-docs" "rust-src" ];
            })
            # We use nightly rustfmt features.
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))
          ] ++ vulkanDependencies ++ vulkanTools ++ openglDependencies ++ winitDependencies;

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
        devShells.nightly = with pkgs; mkShell rec {
          buildInputs = [
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.minimal.override {
              extensions = [ "clippy" "miri" "rust-analyzer" "rustfmt" "rust-docs" "rust-src" ];
            }))
          ] ++ vulkanDependencies ++ vulkanTools ++ openglDependencies ++ winitDependencies;

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
        devShells.CI = with pkgs; mkShell rec {
          buildInputs = [
            (rust-bin.stable.latest.minimal.override {
              extensions = [ "clippy" ];
              # Windows CI unfortunately needs to cross-compile from within WSL because Nix doesn't
              # work on Windows.
              targets = [ "x86_64-pc-windows-msvc" ];
            })
            # We use nightly rustfmt features.
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))
          ] ++ vulkanDependencies ++ workflowDependencies;

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
        };
        devShells.CI-MSRV = let
          manifest = builtins.fromTOML (builtins.readFile ./Cargo.toml);
          msrv = manifest.workspace.package.rust-version;
        in with pkgs; mkShell rec {
          buildInputs = [
            (rust-bin.stable.${msrv}.minimal.override {
              # Windows CI unfortunately needs to cross-compile from within WSL because Nix doesn't
              # work on Windows.
              targets = [ "x86_64-pc-windows-msvc" ];
            })
          ] ++ vulkanDependencies;

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
        };
      }
    );
}
