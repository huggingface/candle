use std::process::Command;
use std::{env, str};
use std::path::PathBuf;

const METAL_SOURCES: [&str; 1] = ["reduce"];

fn main() -> Result<(), String> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=*.metal");
    println!("cargo:rerun-if-changed=*.m");

    let xcrun_output = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .expect("xcrun command failed to start");

    let sdk_path = str::from_utf8(&xcrun_output.stdout)
        .expect("Invalid UTF-8 from xcrun")
        .replace('\n', "");

    println!("cargo:rerun-if-changed={sdk_path}");
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_|"OUT_DIR not set")?);

    let sources = current_dir
        .join("src")
        .to_str()
        .unwrap()
        .to_string();

    // Compile metal to air
    let mut compile_air_cmd = Command::new("xcrun");
    compile_air_cmd
        .arg("metal")
        .arg(format!("-working-directory={}", out_dir.to_str().ok_or("")?))
        .arg("-c")
        .arg("-frecord-sources")
        .arg("-w");
    for metal_file in METAL_SOURCES {
        compile_air_cmd.arg(format!("{sources}/{metal_file}.metal"));
    }
    compile_air_cmd.spawn().expect("Failed to compile air");

    // Compile air to metallib
    let metallib = out_dir.join("candle.metallib");
    let mut compile_metallib_cmd = Command::new("xcrun");
    compile_metallib_cmd
        .arg("metal")
        .arg("-o")
        .arg(&metallib);

    for metal_file in METAL_SOURCES {
        compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
    }

    compile_metallib_cmd
        .spawn()
        .expect("Failed to compile metallib");

    Ok(())
}
