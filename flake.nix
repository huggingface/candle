{
  description = "Development environment for Candle";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";

  outputs =
    {
      nixpkgs,
      rust-overlay,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "clippy"
              "rustfmt"
            ];
          };
          rustWasmToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "clippy"
              "rustfmt"
            ];
            targets = [ "wasm32-unknown-unknown" ];
          };
          darwinFrameworks = pkgs.lib.optionals pkgs.stdenv.isDarwin (
            with pkgs.darwin.apple_sdk.frameworks;
            [
              Accelerate
              Foundation
              Metal
              MetalPerformanceShaders
              Security
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustToolchain
                pkg-config
                openssl
                protobuf
                cmake
              ]
              ++ lib.optionals stdenv.isLinux [
                lld
              ]
              ++ lib.optionals stdenv.isDarwin [
                libiconv
              ]
              ++ darwinFrameworks;

            shellHook = ''
              echo "Candle development shell"
              echo "Common checks: cargo check --workspace, cargo test --workspace, cargo fmt --all -- --check"
            '';
          };

          pyo3 = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustToolchain
                pkg-config
                openssl
                protobuf
                python313
                maturin
                python313Packages.pytest
                python313Packages.black
              ]
              ++ lib.optionals stdenv.isDarwin [
                libiconv
              ];

            shellHook = ''
              echo "Candle PyO3 development shell"
              echo "Example: cd candle-pyo3 && python -m maturin develop -r --features onnx"
            '';
          };

          wasm = pkgs.mkShell {
            packages = with pkgs; [
              rustWasmToolchain
              trunk
              wasm-pack
              wasm-bindgen-cli
            ];

            shellHook = ''
              echo "Candle WASM development shell"
              echo "The Rust toolchain includes the wasm32-unknown-unknown target"
            '';
          };
        }
      );
    };
}
