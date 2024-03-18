use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files: std::fs::ReadDir = std::fs::read_dir("src/").unwrap();
    for file in files {
        let file = file?;
        let path = file.path();
        if let Some(extension) = path.extension() {
            if extension == "metal" {
                build_kernel(&path)?;
            }
            println!("cargo:warning=output {:?}", path.file_stem());
        }
    }
    Ok(())
}

fn build_kernel(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let stem = path
        .file_stem()
        .expect("expect real filename")
        .to_str()
        .expect("expect real stem");
    Command::new("xcrun")
        .args([
            "metal",
            "-c",
            path.as_os_str().to_str().expect("Expect a real filename"),
            "-I",
            "src/",
            "-o",
            &format!("src/compiled/{stem}.air"),
        ])
        .output()?;
    Command::new("xcrun")
        .args([
            "metallib",
            &format!("src/compiled/{stem}.air"),
            "-o",
            &format!("src/compiled/{stem}.metallib"),
        ])
        .output()?;
    Ok(())
}
