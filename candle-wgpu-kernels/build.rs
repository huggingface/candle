use std::borrow::Cow;
use std::collections::HashMap;
use std::{env, fs};
use std::path::{Path, PathBuf};

use wgpu_compute_layer_pwgsl::shader_loader::{DefineDefinition, DefinesDefinitions};
use wgpu_compute_layer_pwgsl::{ParseState, ShaderStore, shader_loader};

fn get_all_pwgsl_files<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    let mut files = Vec::new();
    visit_dirs(dir.as_ref(), &mut files).unwrap();
    files
}

fn visit_dirs(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else if path.extension().map(|ext| ext == "pwgsl").unwrap_or(false) {
                files.push(path);
            }
        }
    }
    Ok(())
}

fn to_upper_camel_case(snake_case: &str) -> String {
    snake_case.to_lowercase()
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first_char) => first_char.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

pub fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");

    let candle_core_path = PathBuf::from(&manifest_dir);
    let absolute_candle_core_path = fs::canonicalize(&candle_core_path).expect("Failed to get absolute outer folder path");

    let shader_dir = candle_core_path.join("src");
    if !shader_dir.exists(){
        panic!("could not find shader path {:?}", shader_dir);
    }
    println!("Searching Shaders in: {:?}", shader_dir);
    
    let mut modules : Vec<(String, String, String)> = Vec::new(); //name, path, content

    let mut parse_state = ParseState::new();

    //1. create store
    println!("Create Store:");
    let mut store = ShaderStore::new();
    for file in get_all_pwgsl_files(&shader_dir){
        println!("Found File: {:?}", file);
        let shader_code = fs::read_to_string(&file).expect("Failed to read PWGSL file");
        store.insert(file,  Cow::Owned(shader_code));
    }

    let mut files: Vec<&PathBuf> = store.keys().collect();
    files.sort();

    //2. generate .wgsl files for all files in the store
    println!("Create Shader Files for Store:");
    for file in files{
        println!("Found File: {:?}", file);
        parse_state.set_path(file.clone());
        let original_file_name = file.file_name().unwrap().to_str().unwrap();
        
        let absolute_file_path = fs::canonicalize(file).expect("Failed to get absolute file path");
        match absolute_file_path.strip_prefix(&absolute_candle_core_path) {
            Ok(relative_path) =>  println!("cargo::rerun-if-changed={}", relative_path.to_string_lossy()),
            Err(_) => println!("File path is not inside the outer folder"),
        }
        
        if original_file_name == "util.pwgsl" || original_file_name.contains("Helper"){
            continue;
        }
        
        println!("Found File: {:?}", file);
        let parent_name = file.file_stem().unwrap().to_str().unwrap().to_string();
        let parent_dir = file.parent().unwrap();
        let mut absolute_path = parent_dir.to_str().unwrap().replace("\\", "/");
        absolute_path.push('/');
        
        let relativ_path = parent_dir.strip_prefix(&shader_dir).unwrap();
        let mut relativ_path = relativ_path.to_str().unwrap().replace("\\", "/");
        relativ_path.push('/');

        let generated_dir = parent_dir.join("generated");
        fs::create_dir_all(&generated_dir).unwrap();
   
        let mut create_shader_file = |global_defines : DefinesDefinitions, file_name_ending : &str|{
            parse_state.set_defines(global_defines);
            let shader_content = shader_loader::load_shader(&mut parse_state, &store);
            let new_file_path = generated_dir.join(format!("{}{file_name_ending}", original_file_name));
            if !parse_state.info().global_functions.is_empty() && !shader_content.is_empty(){
                fs::write(new_file_path, shader_content).expect("Failed to write shader file");
                return true;
            }
            else if fs::exists(&new_file_path).expect("Failed to check shader file"){
                fs::remove_file(new_file_path).expect("Failed to remove shader file");
            }
            false
        };

        //create the File:
        let mut available_types = Vec::new();
        const TYPES : [&str;6] = ["f32", "u32", "i64", "f64", "f16", "u8"];
        for dtype in TYPES{
            if create_shader_file([(dtype.to_string(), DefineDefinition::new_empty())].into_iter().collect(), &format!("_generated_{dtype}.wgsl")){
                available_types.push(dtype);
            }
        }
        

        let global_functions = parse_state.reset_global_functions();
        if !global_functions.is_empty(){//no compute functions
            let functions : Vec<String> = global_functions.iter().map(|(key, _)| to_upper_camel_case(key)).collect();

            let functions_match : Vec<String> = global_functions.iter().map(|(key, value)| format!("Functions::{} => \"{}\"", to_upper_camel_case(key), value)).collect();
            
            let mut include_code = available_types.iter().map(|available_type|{
                format!("crate::DType::{} => include_str!(\"{absolute_path}/generated/{parent_name}.pwgsl_generated_{available_type}.wgsl\")", available_type.to_uppercase())
            }).collect::<Vec<_>>().join(",");
            if available_types.len() != TYPES.len(){
                include_code += ",\n\t\t_=> todo!(\"the type {typ:?} is not implemented for \")";
            }

            let content =  
            format!("pub mod {parent_name} {{
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[cfg_attr(feature=\"wgpu_debug_serialize\", derive(serde::Serialize, serde::Deserialize))]
        pub enum Functions{{{}}}
        impl crate::EntryPoint for Functions{{
            fn get_entry_point(&self) -> &'static str{{
                match self{{
                    {}
                }}
            }} 
        }}

        impl Functions{{
            pub fn get_index(&self, shader : crate::ShaderIndex) -> crate::PipelineIndex{{
                match self{{
                    {}
                }}
            }} 
            pub fn from_index(index : u8) -> Self{{
                match index{{
                    {}
                    _=> {{todo!()}}
                }}
            }} 
        }}

        pub fn load_shader(typ : crate::DType) -> &'static str {{
            match typ{{
                {include_code}
            }}
        }}
    }}
        ", functions.join(","), functions_match.join(","), 
        global_functions.iter().enumerate().map(|(index, (key, _))| format!("\t\tFunctions::{} => crate::PipelineIndex::new(shader, {index}),", to_upper_camel_case(key))).collect::<Vec<String>>().join("\n"),
        global_functions.iter().enumerate().map(|(index, (key, _))| format!("\t\t{index} => Functions::{},", to_upper_camel_case(key))).collect::<Vec<String>>().join("\n")
    );
            
            modules.push((parent_name.clone(),relativ_path.clone(), content));
        }
    }


    fn change_path(path : &str) -> String{
        path.replace("/", "::").to_string()
    }

    //create src/generated.rs
    let mut shader_content = "use crate::*; \n
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature=\"wgpu_debug_serialize\", derive(serde::Serialize, serde::Deserialize))]
pub enum Pipelines{\n".to_string();

    for (m, path, _) in modules.iter(){
        shader_content.push_str(&format!("\t{}(DType, {}{}::Functions),\n", to_upper_camel_case(m), change_path(path), m)); 
    }
    shader_content.push('}');

    let functions_match : Vec<String> = modules.iter().map(|(m, _, _)| format!("\t\t\tPipelines::{}(_, f) => f.get_entry_point()", to_upper_camel_case(m))).collect();



    shader_content.push_str(
        &format!("
impl crate::EntryPoint for Pipelines{{
    fn get_entry_point(&self) -> &'static str{{
        match self{{
{}
        }}
    }} 
}}
    ",functions_match.join(",\n")));

shader_content.push_str(
"#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shaders{\n");
for (m, _, _) in modules.iter(){
    shader_content.push_str(&format!("\t{}(DType),\n", to_upper_camel_case(m))); 
}
shader_content.push('}');

shader_content.push_str(
    &format!("
impl Pipelines {{
    pub fn get_shader(&self) -> Shaders{{
        match self{{
{}
        }}
    }}

    pub fn load_shader(&self) -> &'static str{{
        match self{{
{}        
        }}
    }}
}} 
",
modules.iter().map(|(m, _, _)| format!("\t\t\tPipelines::{}(typ, _) => Shaders::{}(typ.clone())", to_upper_camel_case(m), to_upper_camel_case(m))).collect::<Vec<String>>().join(",\n"),
modules.iter().map(|(m, path, _)| format!("\t\tPipelines::{}(typ, _) => {}{}::load_shader(typ.clone())", to_upper_camel_case(m),change_path(path), m)).collect::<Vec<String>>().join(",\n")
));


shader_content.push_str(
    &format!("


impl From<Shaders> for ShaderIndex
{{
    fn from(val : Shaders) -> Self{{
        match val{{
            {}
        }}
    }}
}}

    
impl From<ShaderIndex> for Shaders
{{
    fn from(val : ShaderIndex) -> Self{{
        let typ = val.get_index() % DTYPE_COUNT;
        let shader_index  = val.get_index() / DTYPE_COUNT;
        match shader_index{{
            {}
            _ => {{todo!()}}
        }}
    }}
}}



impl From<Pipelines> for PipelineIndex
{{
    fn from(val : Pipelines ) -> Self{{
        let shader = val.get_shader();
        match val{{
            {}
        }}
    }}
}}



impl From<PipelineIndex> for Pipelines
{{
    fn from(val : PipelineIndex) -> Self{{
        let shader = val.get_shader();
        let typ = shader.get_index() % DTYPE_COUNT;
        let shader_index  = shader.get_index() / DTYPE_COUNT;
        match shader_index{{
            {}
            _ => {{todo!()}}
        }}
    }}
}}
    ",
modules.iter().enumerate().map(|(i, (m, _, _))| 
    format!("\t\t\tShaders::{}(typ) => {{
        ShaderIndex::new(DefaultWgpuShader::LOADER_INDEX, {} typ.get_index())
    }}", to_upper_camel_case(m), {if i == 0 {"".to_owned()} else if i == 1{"DTYPE_COUNT + ".to_owned()} else {format!("{i}*DTYPE_COUNT +")}})).collect::<Vec<String>>().join(",\n"),

modules.iter().enumerate().map(|(i, (m, _, _))| format!("\t\t\t{i} => Shaders::{}(DType::from_index(typ)),", to_upper_camel_case(m))).collect::<Vec<String>>().join("\n"),


modules.iter().map(|(m, _, _)| format!("\t\t\tPipelines::{}(_, functions) => {{
        let shader_index : ShaderIndex = shader.into();
        functions.get_index(shader_index)
    }}", to_upper_camel_case(m))).collect::<Vec<String>>().join(",\n"),
modules.iter().enumerate().map(|(i, (m, path, _))| format!("\t\t\t{i} => Pipelines::{}(DType::from_index(typ), {}{}::Functions::from_index(val.get_index())),", to_upper_camel_case(m), change_path(path), m)).collect::<Vec<String>>().join("\n"),
));

shader_content.push_str(
    &format!("
impl Shaders {{
    pub fn get_shader(&self) -> Shaders{{
        match self{{
{}
        }}
    }}

    pub fn load_shader(&self) -> &'static str{{
        match self{{
{}        
        }}
    }}
}} 
",
modules.iter().map(|(m, _, _)| format!("\t\t\tShaders::{}(typ) => Shaders::{}(typ.clone())", to_upper_camel_case(m), to_upper_camel_case(m))).collect::<Vec<String>>().join(",\n"),
modules.iter().map(|(m, path, _)| format!("\t\tShaders::{}(typ) => {}{}::load_shader(typ.clone())", to_upper_camel_case(m),change_path(path), m)).collect::<Vec<String>>().join(",\n")
));


shader_content.push_str(
    &format!("
#[derive(Debug, Clone, PartialEq, Eq, Hash, std::marker::Copy,Default)]
pub enum Constants {{
    #[default]
    None,
    {}
}}

impl crate::EntryPoint for Constants{{
    fn get_entry_point(&self) -> &'static str{{
        match self{{
{},
            Constants::None => panic!(\"not expected\")
        }}
    }} 
}}
",
parse_state.info().global_overrides.keys().map(|k| format!("\t{}", to_upper_camel_case(k))).collect::<Vec<String>>().join(",\n"),
parse_state.info().global_overrides.iter().map(|(k, v)| format!("\t\t\tConstants::{} => \"{v}\"", to_upper_camel_case(k))).collect::<Vec<String>>().join(",\n"),
));

    shader_content.push_str(&organize_modules(modules.iter().map(|(_m, p, code)| (p.clone(), code.clone()))));

    //add runtime store:
    shader_content.push_str("pub fn embedded_shader_store() -> ShaderStore {\n");
    shader_content.push_str("    let mut store = ShaderStore::new();\n\n");

    for file in store.keys(){
        let relativ_path = file.strip_prefix(&shader_dir).unwrap();
        let relativ_path = relativ_path.to_str().unwrap().replace("\\", "/");

        shader_content.push_str(&format!(
            "    store.insert(std::path::Path::new({:?}).into(), std::borrow::Cow::Borrowed(include_str!({:?})));\n",
            relativ_path,
            file.display(),
        ));
    }
    shader_content.push_str("\n    store\n}\n");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated.rs");
    println!("Write generated.rs to {:?}", dest_path);
    fs::write(dest_path, shader_content).expect("Failed to write shader map file");


}

fn organize_modules(snippets: impl Iterator<Item=(String, String)>) -> String {
    let mut module_map: HashMap<String, Vec<String>> = HashMap::new();

    // Step 1: Group code snippets by their relative pathsq
    for (path, code) in snippets {
        module_map.entry(path).or_default().push(code);
    }
    // Step 2: Recursively create module code
    fn create_module_code(prefix: &str, module_map: &HashMap<String, Vec<String>>) -> String {
        let mut code = String::new();

        // Collect all immediate modules and their codes
        let mut immediate_modules: HashMap<&str, Vec<&String>> = HashMap::new();
        let mut immediate_code = Vec::new();
        for (path, codes) in module_map {
            if let Some(remainder) = path.strip_prefix(prefix) {
                if remainder.is_empty() {
                    immediate_code.extend(codes.iter());
                } else {
                    let mut parts = remainder.splitn(2, '/');
                    let next_module = parts.next().unwrap();
                    let _rest = parts.next().unwrap_or("");
                    immediate_modules.entry(next_module).or_default().extend(codes.iter());
                }
            }
        }

        // Add immediate code
        for code_snippet in immediate_code {
            code.push_str(&format!("    {}\n", code_snippet));
        }

        // Add nested modules
        for module_name in immediate_modules.keys() {
            code.push_str(&format!("    pub mod {} {{\n", module_name));
            let nested_prefix = format!("{}{}/", prefix, module_name);
            let nested_code = create_module_code(&nested_prefix, module_map);
            code.push_str(&nested_code);
            code.push_str("    }\n");
        }
        code
    }

    let mut final_code = String::new();
    final_code.push_str("pub mod kernels {\n");
    let root_code = create_module_code("kernels/", &module_map);
    final_code.push_str(&root_code);
    final_code.push_str("}\n");
    final_code
  
}