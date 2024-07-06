use std::{fs, path::{Path, PathBuf}, time};
use glob::glob;
use anyhow::Context;

use log::error;
use pico_args::Arguments;
use xshell::Shell;
use notify::{Watcher, RecursiveMode};

use crate::util::{check_all_programs, Program};

fn copy_newest_matching_file(
    source_folder: &str,
    destination_folder: &str,
    filename: &str
) -> Result<(), std::io::Error> {
    // Construct a glob pattern that matches files starting with the specified filename_pattern and ending with .wasm
    let pattern = format!("{}/{}*.wasm", source_folder, filename);

    // Use glob crate to find files matching the constructed pattern and collect them into a vector
    let files: Vec<PathBuf> = glob(&pattern).unwrap()
        .filter_map(Result::ok)
        .collect();

    // Find the newest file in terms of modification time using max_by_key
    let newest_file = files.iter()
        .max_by_key(|&path| fs::metadata(path).unwrap().modified().unwrap());

    // If we found a newest file, copy it to the destination folder with the specified destination filename
    if let Some(file_path) = newest_file {
        let destination_path = Path::new(destination_folder).join(format!("{filename}.wasm"));

        fs::copy(&file_path, &destination_path)?;
        println!("Copied {} to {}", filename, destination_path.display());
    } else {
        println!("No matching files found for pattern '{}' in {}", filename, source_folder);
    }
    Ok(())
}

fn compile(shell: &Shell, mut args: Arguments, name : &str, is_bench : bool) -> Result<(), anyhow::Error> {
    let release = args.contains("--release");
    let release_flag: &[_] = if release { &["--release"] } else { &[] };
    let output_dir = if release { "release" } else { "debug" };

    

    log::info!("building, outdir:{output_dir}");

    let cargo_args = args.finish();

    xshell::cmd!(
        shell,
        "cargo build --target wasm32-unknown-unknown {release_flag...}"
    )
    .args(&cargo_args)
    .quiet()
    .run()
    .context("Failed to build tests examples for wasm")?;

    if is_bench{ //When running benchmark, we need to copy that file from deps folder:
        log::info!("copy bench");
        copy_newest_matching_file(
            &format!("target/wasm32-unknown-unknown/{output_dir}/deps"), 
            &format!("target/wasm32-unknown-unknown/{output_dir}"), 
         name)?;
    }

    log::info!("running wasm-bindgen");

    xshell::cmd!(
        shell,
        "wasm-bindgen ../../target/wasm32-unknown-unknown/{output_dir}/m.wasm  --target web --no-typescript --out-dir build --out-name m"
    )
    .quiet()
    .run().inspect_err(|f| println!("{:?}",f))
    .context("Failed to run wasm-bindgen")?;

    Ok(())
}

pub(crate) fn run_wasm(shell: Shell, mut args: Arguments) -> Result<(), anyhow::Error> {
    let no_serve = args.contains("--no-serve");
   
    let name1   = args.value_from_str::<&str, String>("--bin");//.unwrap_or(args.value_from_str("--bench").unwrap_or("test_1".to_owned()));
    let name2 = args.value_from_str::<&str, String>("--bench");

    let name : String;
    let mut is_bench = false;
    if let Ok(b) = name2{
        is_bench = true;
        name = b;
    }
    else if let Ok(b) = name1{
        name = b;
    }   
    else {
        name = "test_1".to_owned();
    }

    //let name : String = args.value_from_str("--bin").unwrap_or("test_1".to_owned());

    let programs_needed: &[_] = if no_serve {
        &[Program {
            crate_name: "wasm-bindgen-cli",
            binary_name: "wasm-bindgen",
        }]
    } else {
        &[
            Program {
                crate_name: "wasm-bindgen-cli",
                binary_name: "wasm-bindgen",
            },
            Program {
                crate_name: "simple-http-server",
                binary_name: "simple-http-server",
            },
        ]
    };

    check_all_programs(programs_needed)?;

    _ = compile(&shell, args.clone(), &name, is_bench).inspect_err(|err|  error!("couldnt compile: {}", err));

    let mut last_compile = time::Instant::now();
    let mut compiling = false;
    // Automatically select the best implementation for your platform.
    let mut watcher = notify::recommended_watcher(move|res : Result<notify::Event, notify::Error>| {
        match res {
           Ok(event) => {
                println!("event: {:?}", event);
                if event.paths.iter().any(|p| p.components().any(|c| c.as_os_str() == "src")) {
                    let now = time::Instant::now();
                    if now.duration_since(last_compile).as_secs_f32() > 0.5 && compiling == false {
                        let shell = xshell::Shell::new().context("Couldn't create xshell shell").expect("Couldn't create xshell shell");
                        shell.change_dir(String::from(env!("CARGO_MANIFEST_DIR")) + "/..");
                        _ = compile(&shell, args.clone(), &name, is_bench).inspect_err(|err|  error!("couldnt compile changes: {}", err));
                        last_compile = time::Instant::now();
                        compiling = false;
                    }
                }

                
            },
           Err(e) => println!("watch error: {:?}", e),
        }
    })?;

    // Add a path to be watched. All files and directories at that path and
    // below will be monitored for changes.
    watcher.watch(Path::new(&("./")), RecursiveMode::Recursive)?;

    if !no_serve {
        log::info!("serving on port 80");

        xshell::cmd!(
            shell,
            "simple-http-server -c wasm,html,js -i -p 80"
        )
        .quiet()
        .run()
        .context("Failed to simple-http-server")?;
    }

    Ok(())
}
