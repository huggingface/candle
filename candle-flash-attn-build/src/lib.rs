//! Build utilities for fetching cutlass headers on-demand.
//!
//! This crate provides a function to fetch NVIDIA's cutlass library headers
//! during build time, avoiding the need for git submodules.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass.git";

/// Fetch cutlass headers if not already present at the specified commit.
///
/// The headers are cloned to `out_dir/cutlass` using sparse checkout to only
/// fetch the `include/` directory, minimizing download size.
///
/// # Arguments
/// * `out_dir` - The output directory (typically from `OUT_DIR` env var)
/// * `commit` - The git commit hash to checkout
///
/// # Returns
/// The path to the cutlass directory containing the `include/` subdirectory.
pub fn fetch_cutlass(out_dir: &PathBuf, commit: &str) -> Result<PathBuf> {
    let cutlass_dir = out_dir.join("cutlass");

    // Check if cutlass is already fetched and at the right commit
    if cutlass_dir.join("include").exists() {
        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&cutlass_dir)
            .output();

        if let Ok(output) = output {
            let current_commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if current_commit == commit {
                return Ok(cutlass_dir);
            }
        }
    }

    // Clone cutlass if the directory doesn't exist
    if !cutlass_dir.exists() {
        println!("cargo::warning=Cloning cutlass from {}", CUTLASS_REPO);
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                CUTLASS_REPO,
                cutlass_dir.to_str().unwrap(),
            ])
            .status()
            .context("Failed to clone cutlass repository")?;

        if !status.success() {
            anyhow::bail!("git clone failed with status: {}", status);
        }

        // Set up sparse checkout to only get the include directory
        let status = Command::new("git")
            .args(["sparse-checkout", "set", "include"])
            .current_dir(&cutlass_dir)
            .status()
            .context("Failed to set sparse checkout for cutlass")?;

        if !status.success() {
            anyhow::bail!("git sparse-checkout failed with status: {}", status);
        }
    }

    // Fetch and checkout the specific commit
    println!("cargo::warning=Checking out cutlass commit {}", commit);
    let status = Command::new("git")
        .args(["fetch", "origin", commit])
        .current_dir(&cutlass_dir)
        .status()
        .context("Failed to fetch cutlass commit")?;

    if !status.success() {
        anyhow::bail!("git fetch failed with status: {}", status);
    }

    let status = Command::new("git")
        .args(["checkout", commit])
        .current_dir(&cutlass_dir)
        .status()
        .context("Failed to checkout cutlass commit")?;

    if !status.success() {
        anyhow::bail!("git checkout failed with status: {}", status);
    }

    Ok(cutlass_dir)
}

/// Returns the include path argument for nvcc/compiler.
///
/// # Arguments
/// * `cutlass_dir` - Path returned from `fetch_cutlass`
pub fn cutlass_include_arg(cutlass_dir: &PathBuf) -> String {
    format!("-I{}/include", cutlass_dir.display())
}
