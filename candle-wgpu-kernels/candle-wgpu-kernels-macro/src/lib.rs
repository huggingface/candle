extern crate proc_macro;
use fs2::FileExt;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Ident};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::{AtomicU8, Ordering};
use std::env;
use std::io::{Seek, SeekFrom};

static COUNTER: AtomicU8 = AtomicU8::new(0);


/// Reads the current loader indices from the file, returning a Vec of (crate_name, loader_name, index)
fn read_loader_indices(file : &File) -> Vec<(String, String, u8)> {
    let reader = BufReader::new(file);
    let mut indices = Vec::new();
    for line in reader.lines() {
        match line {
            Ok(line) => {
                let parts: Vec<&str> = line.split(',').collect();
               
                if parts.len() == 3 {
                    
                    let crate_name = parts[0].to_string();
                    let loader_name = parts[1].to_string();
                    let index = parts[2].parse::<u8>().expect("Error parsing loader index.");
                    indices.push((crate_name, loader_name, index));
                }

            },
            Err(err) => {
                eprintln!("read line error: {err}");
                break;
            },
        }
    }

    indices
}

/// Writes the updated loader indices to the file
fn write_loader_indices(file : &mut File, indices: &Vec<(String, String, u8)>) {
    file.set_len(0).expect("could not set size to 0"); // Set length to 0 to truncate

    // Optionally seek back to the beginning if you want to write immediately
    file.seek(SeekFrom::Start(0)).expect("could not seek to the beginning"); // Go back to the start of the file
    let mut writer = BufWriter::new(file);
    for (crate_name, loader_name, index) in indices {
        writeln!(writer, "{},{},{}", crate_name, loader_name, index)
            .expect("Unable to write to loader indices file.");
    }
}

fn get_file() -> File{
    //let out_dir = std::env::var("OUT_DIR").expect("expected out_dir");
    //let dest_path = Path::new(&out_dir).join("wgpu_loader_indices.txt");
    let dest_path = "wgpu_loader_indices.txt";
    let file = OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(false)
    .open(dest_path)
    .expect("Unable to open or create loader indices file.");

    // Lock the file for writing
    file.lock_exclusive().expect("Unable to lock indices file.");
    file
}

/// Procedural macro to create a loader with a unique constant index
#[proc_macro]
pub fn create_loader(input: TokenStream) -> TokenStream {
    // Lock to ensure thread-safe access to the index file
    let mut file = get_file();

    // Parse the input to get the loader name
    let loader_name = parse_macro_input!(input as Ident);
    let loader_name_str = loader_name.to_string();

    // Get the crate name and the line number of the macro invocation
    let crate_name = env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "unknown_crate".to_string());

    // Read the current loader indices from the file
    let mut indices = read_loader_indices(&file);

   // Assign the next index using the atomic counter
   let mut new_index = COUNTER.fetch_add(1, Ordering::SeqCst);

    if new_index == 0{
        indices.retain(|(crate_name_existing, _, _)| *crate_name_existing != crate_name);
    }
    for i in 0..{
        if indices.iter().all(|c| c.2 != i) {
            new_index = i;
            break;
        }
    }

    if !indices.iter().any(|c| c.0 == crate_name && c.1 == loader_name_str && c.2 == new_index){
        indices.push((crate_name.clone(), loader_name_str.clone(), new_index));
    }
   
 
    write_loader_indices(&mut file, &indices);

    // Generate the loader code with a unique index
    let expanded = quote! {
        #[derive(Debug)]
        pub struct #loader_name;

        impl #loader_name {
            pub const LOADER_INDEX : candle_wgpu_kernels::LoaderIndex = candle_wgpu_kernels::LoaderIndex(#new_index);
        }
    };

    file.unlock().expect("Unable to unlock indices file.");

    TokenStream::from(expanded)
}


/// Procedural macro to create a loader with a unique constant index
#[proc_macro]
pub fn create_loader_internal(input: TokenStream) -> TokenStream {
    // Lock to ensure thread-safe access to the index file
    let mut file = get_file();

    // Parse the input to get the loader name
    let loader_name = parse_macro_input!(input as Ident);
    let loader_name_str = loader_name.to_string();

    // Get the crate name and the line number of the macro invocation
    let crate_name = env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "unknown_crate".to_string());

    // Read the current loader indices from the file
    let mut indices = read_loader_indices(&file);

   // Assign the next index using the atomic counter
   let mut new_index = COUNTER.fetch_add(1, Ordering::SeqCst);

    if new_index == 0{
        indices.retain(|(crate_name_existing, _, _)| *crate_name_existing != crate_name);
    }
    for i in 0..{
        if indices.iter().all(|c| c.2 != i) {
            new_index = i;
            break;
        }
    }

    if !indices.iter().any(|c| c.0 == crate_name && c.1 == loader_name_str && c.2 == new_index){
        indices.push((crate_name.clone(), loader_name_str.clone(), new_index));
    }
   
 
    write_loader_indices(&mut file, &indices);

    // Generate the loader code with a unique index
    let expanded = quote! {
        #[derive(Debug)]
        pub struct #loader_name;

        impl #loader_name {
            pub const LOADER_INDEX : crate::LoaderIndex = crate::LoaderIndex(#new_index);
        }
    };

    file.unlock().expect("Unable to unlock indices file.");

    TokenStream::from(expanded)
}