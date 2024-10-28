extern crate proc_macro;
use fs2::FileExt;
use proc_macro::TokenStream;
use quote::quote;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::io::{Seek, SeekFrom};
use syn::{parse_macro_input, Ident};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Loader {
    crate_name: String,
    loader_name: String,
    index: u8,
}

// Wrap loaders in a struct to match the TOML array of tables format.
#[derive(Debug, Deserialize)]
struct LoadersFile {
    loader: Vec<Loader>,
}

fn read_loader_indices(file: &File) -> Vec<Loader> {
    let mut reader = BufReader::new(file);
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .expect("Unable to read file");

    // Parse the content as TOML into LoadersFile
    let loaders_file: LoadersFile = toml::from_str(&content).expect("Error parsing TOML");
    loaders_file.loader
}

/// Writes the updated loader indices to the file
fn write_loader_indices(file: &mut File, indices: &Vec<Loader>) {
    file.set_len(0).expect("could not set size to 0"); // Set length to 0 to truncate
    file.seek(SeekFrom::Start(0))
        .expect("could not seek to the beginning"); // Go back to the start of the file
    let mut writer = BufWriter::new(file);

    for Loader {
        crate_name,
        loader_name,
        index,
    } in indices
    {
        writeln!(writer, "[[loader]]").expect("Unable to write to loader indices file.");
        writeln!(writer, "crate_name = \"{}\"", crate_name)
            .expect("Unable to write to loader indices file.");
        writeln!(writer, "loader_name = \"{}\"", loader_name)
            .expect("Unable to write to loader indices file.");
        writeln!(writer, "index = {}", index).expect("Unable to write to loader indices file.");
        writeln!(writer).expect("Unable to write to loader indices file."); // Blank line between entries
    }
}
fn get_file() -> File {
    //let out_dir = std::env::var("OUT_DIR").expect("expected out_dir");
    //let dest_path = Path::new(&out_dir).join("wgpu_loader_indices.txt");
    let dest_path = "wgpu_loader_indices.toml";
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

    let mut index = None;


    for e in indices.iter(){
        if e.crate_name == crate_name && e.loader_name == loader_name_str{
            index = Some(e.index);
        }
    }

    //element not found:
    if index.is_none(){
        //search new index:
        for i in 0.. {
            if indices.iter().all(|c| c.index != i) {
                index = Some(i);
                indices.push(Loader{crate_name, loader_name : loader_name_str, index : i});
                write_loader_indices(&mut file, &indices);
                break;
            }
        }
    }
        
    let new_index = index.unwrap();

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

    let mut index = None;


    for e in indices.iter(){
        if e.crate_name == crate_name && e.loader_name == loader_name_str{
            index = Some(e.index);
        }
    }

    //element not found:
    if index.is_none(){
        //search new index:
        for i in 0.. {
            if indices.iter().all(|c| c.index != i) {
                index = Some(i);
                indices.push(Loader{crate_name, loader_name : loader_name_str, index : i});
                write_loader_indices(&mut file, &indices);
                break;
            }
        }
    }
        
    let new_index = index.unwrap();

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
