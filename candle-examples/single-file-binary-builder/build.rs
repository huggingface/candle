use std::{
    fs::{self, File},
    io::copy,
    path::Path,
};

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let base_url = core::option_env!("CANDLE_SINGLE_FILE_BINARY_BUILDER_URL");
    if base_url.is_none() {
        return Ok(());
    }
    let base_url = base_url.unwrap();

    println!("cargo::rerun-if-changed=build.rs");

    let example_name = "bert-single-file-binary-builder";
    let dest_path = Path::new("files");
    let files = ["config.json", "tokenizer.json", "model.safetensors"];

    let all_files_exist = files
        .iter()
        .all(|filename| dest_path.join(filename).exists());

    if all_files_exist {
        println!(
            "cargo::warning=All {} files already exist, skipping download",
            example_name
        );
        return Ok(());
    }

    println!("cargo::warning=Downloading {} files...", example_name);

    fs::create_dir_all(dest_path).context("Failed to create destination directory")?;

    for filename in &files {
        let dest_file = dest_path.join(filename);

        if dest_file.exists() {
            println!("cargo::warning=File already exists, skipping: {}", filename);
            continue;
        }

        let url = format!("{}/{}", base_url, filename);
        println!("cargo::warning=Downloading {} from {}...", filename, url);

        let response = ureq::get(&url)
            .call()
            .context(format!("Failed to download {}", url))?;

        if response.status() != 200 {
            anyhow::bail!(
                "Download failed for {} with status: {}",
                filename,
                response.status()
            );
        }

        let mut reader = response.into_reader();
        let mut file =
            File::create(&dest_file).context(format!("Failed to create file {:?}", dest_file))?;

        let bytes_written =
            copy(&mut reader, &mut file).context(format!("Failed to write {}", filename))?;

        println!(
            "cargo::warning=Downloaded {} ({} bytes)",
            filename, bytes_written
        );
    }

    println!(
        "cargo::warning=All {} files downloaded successfully",
        example_name
    );

    Ok(())
}
