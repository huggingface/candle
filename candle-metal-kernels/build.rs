use std::process::Command;
use std::{env, str};

const COMPILED_KERNELS: [&str; 1] = ["reduce"];

enum Platform {
    MacOS,
    IOS,
}

impl Platform {
    fn as_str(&self) -> &'static str {
        match self {
            Platform::MacOS => "macosx",
            Platform::IOS => "iphoneos",
        }
    }
}

fn get_xcode_sdk_path(platform: Platform) -> Result<String, String> {
    let xcrun_output = Command::new("xcrun")
        .args(["--sdk", platform.as_str(), "--show-sdk-path"])
        .output()
        .expect("xcrun command failed to start");

    Ok(str::from_utf8(&xcrun_output.stdout)
        .expect("Invalid UTF-8 from xcrun")
        .replace('\n', ""))
}

fn compile_candle_metallib(sdk_path: String, bfloat_support: bool) -> Result<(), String> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let out_dir = current_dir.join("src/libraries");
    let air_dir = current_dir.join("src/air");
    let working_directory = air_dir.display();
    let sources = current_dir.join("src/kernels");

    // Compile metal to air
    let mut compile_air_cmd = Command::new("xcrun");
    compile_air_cmd
        .arg("metal")
        .arg(format!("-working-directory={working_directory}"))
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-O3")
        .arg("-c")
        .arg("-w");
    for metal_file in COMPILED_KERNELS {
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
    let metallib = out_dir.join("candle.metallib");

    let mut compile_metallib_cmd = Command::new("xcrun");
    compile_metallib_cmd.arg("metal").arg("-o").arg(&metallib);

    for metal_file in COMPILED_KERNELS {
        compile_metallib_cmd.arg(air_dir.join(format!("{metal_file}.air")));
    }
    compile_metallib_cmd.arg(air_dir.join("utils.air"));

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
    println!("cargo::rerun-if-changed=build.rs");

    let current_dir = env::current_dir().expect("Failed to get current directory");
    let sources = current_dir.join("src/kernels");

    for metal_file in COMPILED_KERNELS {
        println!("cargo::rerun-if-changed={}", sources.join(format!("{metal_file}.metal")).display());
        println!("cargo:warning=output {}", sources.join(format!("{metal_file}.metal")).display());
    }

    let macos_sdk = get_xcode_sdk_path(Platform::MacOS).expect("Failed to get MacOS SDK path");
    let iphoneos_sdk = get_xcode_sdk_path(Platform::IOS).expect("Failed to get IOS SDK path");

    compile_candle_metallib(macos_sdk, false)?;

    Ok(())
}
