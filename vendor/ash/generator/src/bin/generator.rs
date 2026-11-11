use generator::write_source_code;
use std::path::Path;

fn main() {
    let cwd = std::env::current_dir().unwrap();
    if cwd.ends_with("generator") {
        write_source_code(Path::new("Vulkan-Headers"), "../ash/src");
    } else {
        write_source_code(Path::new("generator/Vulkan-Headers"), "ash/src");
    }
}
