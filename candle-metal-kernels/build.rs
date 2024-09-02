use std::path::PathBuf;
use std::process::Command;
use std::{env, str};

const METAL_SOURCES: [&str; 1] = ["reduce"];

enum Platform {
    MacOS,
    IOS,
}

impl Platform {
    fn sdk(&self) -> &str {
        match self {
            Platform::MacOS => "macosx",
            Platform::IOS => "iphoneos",
        }
    }
}

fn compile(platform: Platform) -> Result<(), String> {
    println!("cargo::rerun-if-changed=reduce.metal");
    println!("cargo::rerun-if-changed=utils.metal");

    let current_dir = env::current_dir().expect("Failed to get current directory");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
    let working_directory = out_dir.to_string_lossy().to_string();
    let sources = current_dir.join("src");

    // Compile metal to air
    let mut compile_air_cmd = Command::new("xcrun");
    compile_air_cmd
        .arg("--sdk")
        .arg(platform.sdk())
        .arg("metal")
        .arg(format!("-working-directory={working_directory}"))
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-O3")
        .arg("-c")
        .arg("-w");
    for metal_file in METAL_SOURCES {
        compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
    }
    compile_air_cmd.arg(sources.join("utils.metal"));
    compile_air_cmd.spawn().expect("Failed to compile air");

    let mut child = compile_air_cmd.spawn().expect("Failed to compile air");

    match child.try_wait() {
        Ok(Some(status)) => {
            if !status.success() {
                panic!(
                    "Compiling metal -> air failed. Exit with status: {}",
                    status
                )
            }
        }
        Ok(None) => {
            let status = child
                .wait()
                .expect("Compiling metal -> air failed while waiting for result");
            if !status.success() {
                panic!(
                    "Compiling metal -> air failed. Exit with status: {}",
                    status
                )
            }
        }
        Err(e) => panic!("Compiling metal -> air failed: {:?}", e),
    }

    // Compile air to metallib
    let lib_name = match platform {
        Platform::MacOS => "candle.metallib",
        Platform::IOS => "candle_ios.metallib",
    };
    let metallib = out_dir.join(lib_name);
    let mut compile_metallib_cmd = Command::new("xcrun");
    compile_metallib_cmd.arg("metal").arg("-o").arg(&metallib);

    for metal_file in METAL_SOURCES {
        compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
    }
    compile_metallib_cmd.arg(out_dir.join("utils.air"));

    let mut child = compile_metallib_cmd
        .spawn()
        .expect("Failed to compile air -> metallib");

    match child.try_wait() {
        Ok(Some(status)) => {
            if !status.success() {
                panic!(
                    "Compiling air -> metallib failed. Exit with status: {}",
                    status
                )
            }
        }
        Ok(None) => {
            let status = child
                .wait()
                .expect("Compiling air -> metallib failed while waiting for result");
            if !status.success() {
                panic!(
                    "Compiling air -> metallib failed. Exit with status: {}",
                    status
                )
            }
        }
        Err(e) => panic!("Compiling air -> metallib failed: {:?}", e),
    }

    Ok(())
}

fn main() -> Result<(), String> {
    compile(Platform::MacOS)?;
    compile(Platform::IOS)?;

    Ok(())
}
