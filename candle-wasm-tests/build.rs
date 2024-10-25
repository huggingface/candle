use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use syn::ItemFn;
use quote::quote;

fn copy_test_folders(source_dir : &str) -> io::Result<()> {
    let destination_dir = "./tests/";
    // Ensure the destination directory exists.
    fs::create_dir_all(destination_dir)?;

    // Iterate over all .rs files in the source directory.
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let path_str = path.to_str().expect("expected path to be convertible to str");
            if path_str.contains("pth_tests") || path_str.contains("serialization_tests")
            {
                continue;
            }
            println!("cargo::rerun-if-changed={}", path.to_str().expect(""));
            println!("processing: {:?}", path);
            // Read the source file content.
            let content = fs::read_to_string(&path)?;

            // Process the content.
            let new_content = process_content(&content);

            // Get the filename and create the destination path.
            if let Some(file_name) = path.file_name() {
                let destination_path = Path::new(destination_dir).join(file_name);
                
                if let Ok(old_content) = fs::read_to_string(destination_path.clone()) {
                    if old_content == new_content {
                        // No change, so skip rewriting
                        continue;
                    }
                }

                // Write the new content to the destination file.
                let mut file = File::create(destination_path)?;
                file.write_all(new_content.as_bytes())?;
            }
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    copy_test_folders("../candle-core/tests/")?;
    copy_test_folders("../candle-nn/tests/")?;
    copy_test_folders("../candle-transformers/tests/")?;

    Ok(())
}

use fancy_regex::Regex;

const BALANCED_PARENTHESES : &str = r"(\([^)(]*(?:\([^)(]*(?:\([^)(]*(?:\([^)(]*\)[^)(]*)*\)[^)(]*)*\)[^)(]*)*\))";

fn make_fn_async_base<'h>(content : &'h str, function_name : &str, new_name : &str) -> std::borrow::Cow<'h, str>{
    let re = Regex::new(&format!("{function_name}(::<[^>]+>)?{BALANCED_PARENTHESES}(?!\\s*[-{{])")).unwrap();

    re.replace_all(content, &format!("{new_name}$1$2.await"))
}


fn make_fn_async<'h>(content : &'h str, function_name : &str) -> std::borrow::Cow<'h, str>{
    make_fn_async_base(content, function_name, &format!("{function_name}_async"))
}

fn make_fn_async_same_name<'h>(content : &'h str, function_name : &str) -> std::borrow::Cow<'h, str>{
    let re = Regex::new(&format!("(?:[\\w_\\d:\\s]*){function_name}(<[^>]+>)?{BALANCED_PARENTHESES}(?!\\s*[-{{])")).unwrap();
    re.replace_all(content, &format!("{function_name}_async$1$2.await"))
}

fn make_fn_async_same_name_same_file<'h>(content : &'h str, function_name : &str) -> std::borrow::Cow<'h, str>{
    let re = Regex::new(&format!("(?<![\\.\\:\\w_]){function_name}(<[^>]+>)?{BALANCED_PARENTHESES}(?!\\s*[-{{])")).unwrap();
    re.replace_all(content, &format!("{function_name}$1$2.await"))
}

fn process_content(content: &str) -> String {
    let global_start = "#![allow(unused_imports, unexpected_cfgs)]".to_string();

    let header = "
// THIS FILE IS AUTO GENERATED, DO NOT EDIT
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch=\"wasm32\")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch=\"wasm32\"))]
use tokio::test as test;
use candle_wasm_tests::{to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async};

".to_string();
    
    let re_use_fix = Regex::new(r"test_device[\n\r\s]*,[\n\r\s]*test_utils::to_vec2_round\s*,").unwrap();

    let mut transformed_content =
        if content.contains("use "){
            (global_start + &content.replacen("use ", &(header + "\nuse ") , 1)).to_string()
        }
        else{
            (global_start + &header + "\n" + content).to_string()
        };

    transformed_content = make_fn_async_base(&transformed_content, "new_wgpu_sync", "new_wgpu").to_string();

    transformed_content =  make_fn_async(&transformed_content, "to_vec0").to_string();
    transformed_content =  make_fn_async(&transformed_content, "to_vec1").to_string();
    transformed_content =  make_fn_async(&transformed_content, "to_vec2").to_string();
    transformed_content =  make_fn_async(&transformed_content, "to_vec3").to_string();
    transformed_content =  make_fn_async(&transformed_content, "to_scalar").to_string();
    transformed_content =  make_fn_async(&transformed_content, "to_device").to_string();
    transformed_content =  make_fn_async(&transformed_content, "sample").to_string();
    transformed_content =  make_fn_async(&transformed_content, "one_hot").to_string();

    transformed_content = make_fn_async_same_name(&transformed_content, "to_vec0_round").to_string();
    transformed_content = make_fn_async_same_name(&transformed_content, "to_vec1_round").to_string();
    transformed_content = make_fn_async_same_name(&transformed_content, "to_vec2_round").to_string();
    transformed_content = make_fn_async_same_name(&transformed_content, "to_vec3_round").to_string();
    
    
    transformed_content = re_use_fix.replace_all(&transformed_content, "test_device,").to_string();

    transformed_content = transformed_content.replace("candle_core", "candle");
    transformed_content = transformed_content.replace("candle_nn::encoding::one_hot", "candle_nn::encoding::one_hot_async");
   
    transformed_content = transformed_content.replace("test_device!", "candle_wasm_tests::test_device!");

    transformed_content = transformed_content.replace("fn $fn_name", "async fn $fn_name");

    match syn::parse_file(&transformed_content){
        Ok(syntax_tree) => {

            let (mut new_output, converted_functions) = convert_to_async_if_await(syntax_tree, content);

            for function in converted_functions{
                new_output =  make_fn_async_same_name_same_file(&new_output, &function).to_string();
            }
            
            let re_fix_async_loop = Regex::new(r"let\s*v\s*=[^;]+?\.to_vec1_async[^;]+;").unwrap();
            new_output = re_fix_async_loop.replace_all(&new_output, "
            let mut v = Vec::new();
            for _ in 0..100 {
                let t = Tensor::randn(0f32, 1f32, N, device)?;
                let vec = t.to_vec1_async::<f32>().await?;
                v.push(vec);
            }
            ").to_string();
           
            new_output
        },
        Err(err) => {
            println!("{transformed_content}");
            panic!("{}", err)
        },
    }
}

fn convert_to_async_if_await(mut file: syn::File, code : &str) -> (String, Vec<String>) {
    let mut converted_functions = vec![];

    for mut item in file.items.iter_mut() {
        // We're only interested in function items
        if let syn::Item::Fn(func) = &mut item {
            // Check if the function body contains `.await`
            if contains_await(func, code) {
                // If the function contains `.await`, modify it to be `async`
                func.sig.asyncness = Some(syn::token::Async::default());
                let name = &func.sig.ident;
                converted_functions.push(quote!(#name).to_string())
            }
        } 
    }

    (prettyplease::unparse(&file), converted_functions)
}

/// Helper function to check if a function contains `.await`
fn contains_await(func: &ItemFn, code : &str) -> bool {

    if func.attrs.iter().any(|c| 
        {
            let name = c.path();
            quote!(#name).to_string() == "test"
        })
    {
        return true;
    }

    let name = &func.sig.ident;
    let name = quote!(#name).to_string();

    if Regex::new(&format!("test_device![\\n\\r\\s]*\\([\\n\\r\\s]*{name}[\\n\\r\\s]*,")).unwrap().is_match(code).unwrap(){
        return true;
    }

    // Traverse the function body and look for `.await`
    for stmt in &func.block.stmts {
        if quote!(#stmt).to_string().contains(". await") {
            return true;
        }
    }
    false
}