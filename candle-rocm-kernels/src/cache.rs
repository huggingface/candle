use crate::error::RocmKernelError;
use rocm_rs::hip::Device;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Manages Ahead-of-Time (AOT) compilation cache for ROCm kernels
pub struct CacheManager {
    /// Base cache directory (e.g., ~/.cache/candle-rocm/)
    cache_dir: PathBuf,
    /// GPU architecture (e.g., "gfx908")
    arch: String,
    /// ROCm version (e.g., "6.1")
    rocm_version: String,
    /// In-memory cache of loaded modules
    modules: Mutex<HashMap<String, Vec<u8>>>,
}

impl CacheManager {
    /// Create a new CacheManager for the given device
    pub fn new(device: &Device) -> Result<Self, RocmKernelError> {
        let arch = detect_gpu_arch(device)?;
        let rocm_version = detect_rocm_version()?;
        let cache_dir = get_cache_dir()?;

        // Create cache directory structure: ~/.cache/candle-rocm/{arch}-{rocm_version}/
        let arch_version = format!("{}-{}", arch, rocm_version);
        let kernel_dir = cache_dir.join(&arch_version);
        fs::create_dir_all(&kernel_dir).map_err(|e| {
            RocmKernelError::Internal(format!(
                "Failed to create cache directory {}: {}",
                kernel_dir.display(),
                e
            ))
        })?;

        Ok(Self {
            cache_dir: kernel_dir,
            arch,
            rocm_version,
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// Get or compile a kernel binary
    ///
    /// # Arguments
    /// - `name`: Unique name for the kernel (e.g., "binary")
    /// - `source`: The HIP source code to compile
    ///
    /// Returns the compiled binary as bytes, loading from cache or compiling if needed
    pub fn get_or_compile(&self, name: &str, source: &str) -> Result<Vec<u8>, RocmKernelError> {
        // Check in-memory cache first
        {
            let modules = self.modules.lock().map_err(|_| {
                RocmKernelError::Internal("Failed to lock modules cache".to_string())
            })?;
            if let Some(binary) = modules.get(name) {
                return Ok(binary.clone());
            }
        }

        // Compute hash of source to version the cache
        let source_hash = compute_source_hash(source);
        let cache_file = self.cache_dir.join(format!("{}_{}.cso", name, source_hash));

        // Try to load from disk cache
        if cache_file.exists() {
            let binary = fs::read(&cache_file).map_err(|e| {
                RocmKernelError::Internal(format!(
                    "Failed to read cached binary {}: {}",
                    cache_file.display(),
                    e
                ))
            })?;

            // Store in memory cache
            let mut modules = self.modules.lock().map_err(|_| {
                RocmKernelError::Internal("Failed to lock modules cache".to_string())
            })?;
            modules.insert(name.to_string(), binary.clone());

            return Ok(binary);
        }

        // Compile the kernel
        let binary = compile_kernel(name, source, &self.arch, &cache_file)?;

        // Store in memory cache
        let mut modules = self
            .modules
            .lock()
            .map_err(|_| RocmKernelError::Internal("Failed to lock modules cache".to_string()))?;
        modules.insert(name.to_string(), binary.clone());

        Ok(binary)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get GPU architecture
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Get ROCm version
    pub fn rocm_version(&self) -> &str {
        &self.rocm_version
    }
}

/// Detect the GPU architecture (e.g., "gfx908", "gfx90a", "gfx942")
fn detect_gpu_arch(_device: &Device) -> Result<String, RocmKernelError> {
    // rocm_rs doesn't expose device properties directly, so we use hipGetDeviceProperties
    // For now, we'll try to get it via hipcc or environment variable as fallback

    // First try to get from environment variable (useful for testing/build machines)
    if let Ok(arch) = std::env::var("CANDLE_ROCM_ARCH") {
        return Ok(arch);
    }

    // Try to use rocminfo to detect the architecture
    match Command::new("rocminfo").arg("-a").output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Look for "Name:" line with gfxXXXX
            for line in stdout.lines() {
                if line.contains("Name:") && line.contains("gfx") {
                    if let Some(start) = line.find("gfx") {
                        let arch = &line[start..];
                        // Extract just the gfxXXXX part
                        let end = arch
                            .find(|c: char| !c.is_alphanumeric())
                            .unwrap_or(arch.len());
                        return Ok(arch[..end].to_string());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to run rocminfo: {}", e);
        }
    }

    // Try hipcc to get default arch
    match Command::new("hipcc").args(&["--version"]).output() {
        Ok(_) => {
            // hipcc exists, try to get default arch
            // Default to a common architecture if we can't detect
            eprintln!("Warning: Could not detect GPU architecture, defaulting to gfx908");
            Ok("gfx908".to_string())
        }
        Err(e) => Err(RocmKernelError::Compilation(format!(
            "hipcc not found: {}. Please install ROCm or set CANDLE_ROCM_ARCH environment variable",
            e
        ))),
    }
}

/// Detect ROCm version
fn detect_rocm_version() -> Result<String, RocmKernelError> {
    // Try to get from environment variable first
    if let Ok(version) = std::env::var("CANDLE_ROCM_VERSION") {
        return Ok(version);
    }

    // Try to get from hipcc --version
    match Command::new("hipcc").args(&["--version"]).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse version from output like "HIP version: 6.1.0"
            for line in stdout.lines() {
                if line.contains("HIP version:") || line.contains("HIP_VERSION:") {
                    if let Some(v) = line.split(':').nth(1) {
                        let version = v.trim().split('.').take(2).collect::<Vec<_>>().join(".");
                        return Ok(version);
                    }
                }
            }
            // If we can't parse, return a default
            Ok("6.0".to_string())
        }
        Err(e) => {
            Err(RocmKernelError::Compilation(format!(
                "hipcc not found: {}. Please install ROCm or set CANDLE_ROCM_VERSION environment variable",
                e
            )))
        }
    }
}

/// Get the base cache directory
fn get_cache_dir() -> Result<PathBuf, RocmKernelError> {
    let home = dirs::cache_dir()
        .or_else(|| std::env::var("HOME").ok().map(PathBuf::from))
        .ok_or_else(|| {
            RocmKernelError::Internal("Could not determine cache directory".to_string())
        })?;

    Ok(home.join("candle-rocm"))
}

/// Compute a hash of the source code
fn compute_source_hash(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let result = hasher.finalize();
    // Use first 16 characters of hex as hash
    format!("{:x}", result)[..16].to_string()
}

/// Compile a kernel using hipcc
///
/// # Arguments
/// - `name`: Kernel name (for error messages)
/// - `source`: HIP source code
/// - `arch`: GPU architecture (e.g., "gfx908")
/// - `output_path`: Where to save the compiled binary
fn find_rocm_tool(tool_name: &str) -> Result<String, RocmKernelError> {
    let output = Command::new("hipcc")
        .args(&["--print-prog-name", tool_name])
        .output()
        .map_err(|e| RocmKernelError::Compilation(format!("Failed to run hipcc: {}", e)))?;
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() && PathBuf::from(&path).exists() {
            return Ok(path);
        }
    }
    Err(RocmKernelError::Compilation(format!(
        "{} not found via hipcc. Is ROCm installed?",
        tool_name
    )))
}

fn compile_kernel(
    name: &str,
    source: &str,
    arch: &str,
    output_path: &Path,
) -> Result<Vec<u8>, RocmKernelError> {
    let temp_dir = std::env::temp_dir();
    let source_hash = compute_source_hash(source);
    let source_file = temp_dir.join(format!("candle_{}_{}.hip", name, source_hash));
    let obj_file = temp_dir.join(format!("candle_{}_{}.o", name, source_hash));
    let fatbin_file = temp_dir.join(format!("candle_{}_{}.fatbin", name, source_hash));
    let hsaco_file = temp_dir.join(format!("candle_{}_{}.hsaco", name, source_hash));

    fs::write(&source_file, source).map_err(|e| {
        RocmKernelError::Compilation(format!(
            "Failed to write source file {}: {}",
            source_file.display(),
            e
        ))
    })?;

    let output = Command::new("hipcc")
        .args(&[
            &format!("--offload-arch={}", arch),
            "-O3",
            "-fPIC",
            "-c",
            "-o",
            obj_file.to_str().unwrap(),
            source_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| {
            RocmKernelError::Compilation(format!(
                "Failed to execute hipcc: {}. Is hipcc in PATH?",
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&source_file);
        return Err(RocmKernelError::Compilation(format!(
            "hipcc compilation failed for {}:\n{}",
            name, stderr
        )));
    }

    let extract_output = Command::new("objcopy")
        .args(&[
            "-O",
            "binary",
            "-j",
            ".hip_fatbin",
            obj_file.to_str().unwrap(),
            fatbin_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| {
            RocmKernelError::Compilation(format!(
                "Failed to execute objcopy: {}. Is binutils in PATH?",
                e
            ))
        })?;

    if !extract_output.status.success() {
        let stderr = String::from_utf8_lossy(&extract_output.stderr);
        let _ = fs::remove_file(&source_file);
        let _ = fs::remove_file(&obj_file);
        return Err(RocmKernelError::Compilation(format!(
            "objcopy extraction failed for {}:\n{}",
            name, stderr
        )));
    }

    let target = format!("hipv4-amdgcn-amd-amdhsa--{}", arch);
    let bundler_path = find_rocm_tool("clang-offload-bundler")?;
    let unbundle_output = Command::new(&bundler_path)
        .args(&[
            "--unbundle",
            "--type=o",
            "--input",
            fatbin_file.to_str().unwrap(),
            "--targets",
            &target,
            "--output",
            hsaco_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| {
            RocmKernelError::Compilation(format!(
                "Failed to execute clang-offload-bundler: {}. Is ROCm in PATH?",
                e
            ))
        })?;

    if !unbundle_output.status.success() {
        let stderr = String::from_utf8_lossy(&unbundle_output.stderr);
        let _ = fs::remove_file(&source_file);
        let _ = fs::remove_file(&obj_file);
        let _ = fs::remove_file(&fatbin_file);
        return Err(RocmKernelError::Compilation(format!(
            "clang-offload-bundler extraction failed for {}:\n{}",
            name, stderr
        )));
    }

    let binary = fs::read(&hsaco_file).map_err(|e| {
        RocmKernelError::Compilation(format!(
            "Failed to read code object {}: {}",
            hsaco_file.display(),
            e
        ))
    })?;

    fs::write(output_path, &binary).map_err(|e| {
        RocmKernelError::Internal(format!(
            "Failed to write cache file {}: {}",
            output_path.display(),
            e
        ))
    })?;

    let _ = fs::remove_file(&source_file);
    let _ = fs::remove_file(&obj_file);
    let _ = fs::remove_file(&fatbin_file);
    let _ = fs::remove_file(&hsaco_file);

    Ok(binary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_hash() {
        let source1 = "__global__ void test() {}";
        let source2 = "__global__ void test() {}";
        let source3 = "__global__ void test2() {}";

        let hash1 = compute_source_hash(source1);
        let hash2 = compute_source_hash(source2);
        let hash3 = compute_source_hash(source3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
