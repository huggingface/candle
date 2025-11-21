use std::collections::HashMap;
use std::{env, fs};
use std::path::{Path, PathBuf};

const SHORTEN_NORMAL_VARIABLES : bool = false;
const SHORTEN_NORMAL_FUNCTIONS : bool = false;
const SHORTEN_GLOBAL_FUNCTIONS : bool = false;
const SHORTEN_OVERRIDES : bool = false;
const REMOVE_UNUSED : bool = true;
const REMOVE_SPACES : bool = false;
const REMOVE_NEW_LINES : bool = false;

fn get_all_wgsl_files<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
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

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");

    let candle_core_path = PathBuf::from(&manifest_dir);
    let absolute_candle_core_path = fs::canonicalize(&candle_core_path).expect("Failed to get absolute outer folder path");

    let shader_dir = candle_core_path.join("src");
    if !shader_dir.exists(){
        panic!("could not find shader path {:?}", shader_dir);
    }
    println!("Searching Shaders in: {:?}", shader_dir);
    
    let mut modules : Vec<(String, String, String)> = Vec::new(); //name, path, content

    let mut shader_info: shader_loader::shader_shortener::ShaderInfo = shader_loader::shader_shortener::ShaderInfo::new();

    for file in get_all_wgsl_files(shader_dir.clone()){
        let original_file_name = file.file_name().unwrap().to_str().unwrap();
        
        let absolute_file_path = fs::canonicalize(&file).expect("Failed to get absolute file path");
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
   
        let mut create_shader_file = |global_defines : &Vec<&'static str>, file_name_ending : &str|{
            let shader_content = shader_loader::load_shader(&file,global_defines, &mut shader_info);
            let new_file_path = generated_dir.join(format!("{}{file_name_ending}", original_file_name));
            if !shader_info.global_functions.is_empty() && !shader_content.is_empty(){
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
            if create_shader_file(&vec![dtype], &format!("_generated_{dtype}.wgsl")){
                available_types.push(dtype);
            }
        }
        

        let global_functions = shader_info.global_functions;
        shader_info.global_functions = Vec::new();
        shader_info.global_function_counter = 1;

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
shader_info.global_overrides.keys().map(|k| format!("\t{}", to_upper_camel_case(k))).collect::<Vec<String>>().join(",\n"),
shader_info.global_overrides.iter().map(|(k, v)| format!("\t\t\tConstants::{} => \"{v}\"", to_upper_camel_case(k))).collect::<Vec<String>>().join(",\n"),
));

    shader_content.push_str(&organize_modules(modules.iter().map(|(_m, p, code)| (p.clone(), code.clone()))));

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated.rs");
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


mod shader_loader{
    use std::{collections::HashMap, path::PathBuf};
    use fancy_regex::Regex;
    use super::*;

    pub fn load_shader(path: &PathBuf, global_defines : &Vec<&'static str>, global_functions : &mut shader_shortener::ShaderInfo) -> String {
        let mut result = shader_defines::load_shader(path, &mut HashMap::new(), global_defines); 
        
        if REMOVE_SPACES{
            result = Regex::new(r"((\s+)(?![\w\s])|(?<!\w)(\s+))", ).unwrap().replace_all(&result, "").to_string(); //replaces newline and not used spaces
        }
        result = global_functions.shorten_variable_names(&result);
        if REMOVE_UNUSED{
            loop{
                let new_result = global_functions.remove_unused(&result);
                if new_result == result{
                    break;
                }
                result = new_result;
            }
           
        }
        if REMOVE_NEW_LINES{
            result = result.replace("\n", "");
        }
        
        result
    }

    pub mod shader_tokeniser{
        use std::{iter::Peekable, str::Chars};

        #[derive(Debug, PartialEq, Eq, Clone)]
        pub enum Token<'a> {
            Word(&'a str),
            Symbol(char)
        }

        pub struct Tokenizer<'a> {
            chars: Chars<'a>,
            input: &'a str,
            pos: usize,
        }

        impl<'a> Tokenizer<'a> {
            pub fn new(input: &'a str) -> Self {
                Tokenizer {
                    chars: input.chars(),
                    input,
                    pos: 0,
                }
            }
        }

        impl<'a> Iterator for Tokenizer<'a> {
            type Item = Token<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(c) = self.chars.next() {
                    self.pos += c.len_utf8();
                    if c.is_alphabetic() || c == '_' {
                        let start = self.pos - c.len_utf8();
                        while let Some(&next_c) = self.chars.clone().peekable().peek() {
                            if next_c.is_alphanumeric() || next_c == '_' {
                                self.chars.next();
                                self.pos += next_c.len_utf8();
                            } else {
                                break;
                            }
                        }
                        return Some(Token::Word(&self.input[start..self.pos]));
                    } else {
                        return Some(Token::Symbol(c));
                    }
                }
                None
            }
        }

        //called after var, const, override or let. Matches until the name of the variable, returns the name
        pub fn match_variable<'a>(tokens : &mut impl Iterator<Item=Token<'a>>, result : &mut String) -> String{
            let mut generic_counter = 0;
            for token in tokens.by_ref() {
                match token{
                    Token::Word(var_name) => {
                        if generic_counter == 0{
                            return var_name.to_string();
                        }
                        else{
                            result.push_str(var_name);
                        }
                    },
                    Token::Symbol(c) => {
                        if c=='<'{
                            generic_counter+=1;
                        }
                        else if c=='>'{
                            generic_counter-=1;
                        }
                        result.push(c);
                    }
                }
            }
            "".to_string()
        }

        pub fn match_function_block<'a>(tokens : &mut impl Iterator<Item=Token<'a>>, result : &mut String){
            let mut generic_counter = 0;
            for token in tokens.by_ref() {
                match token{
                    Token::Word(var_name) => {
                        result.push_str(var_name);
                    },
                    Token::Symbol(c) => {
                        result.push(c);
                        if c=='{'{
                            generic_counter+=1;
                        }
                        else if c=='}'{
                            generic_counter-=1;
                            if generic_counter == 0{
                                return;
                            }
                        }
                    }
                }
            }
        }

        pub fn match_string<'a>(tokens : &mut impl Iterator<Item=Token<'a>>) -> String{
            let mut result = String::new();
            let mut has_string_started = false;
            for token in tokens.by_ref() {
                match token{
                    Token::Word(var_name) => {
                        result.push_str(var_name);
                    },
                    Token::Symbol(c) => {
                        if c=='"'{
                            
                            if has_string_started{
                                break;
                            }
                            else{
                                has_string_started = true;
                            }
                        }
                        else {
                            result.push(c);
                        }
                    }
                }
            }
            result
        }

        pub fn match_until_char<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>, char : char, result : &mut String){
            while let Some(token) = tokens.peek().cloned() {
                match token{
                    Token::Word(var_name) => {
                        tokens.next();
                        result.push_str(var_name);
                    },
                    Token::Symbol(c) => {
                        if c==char{
                            return;
                        }
                        tokens.next();
                        result.push(c);
                    }
                }
            }
        }

        pub fn match_whitespace<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>) -> bool{
            let mut contains_whitspace = false;
            while let Some(token) = tokens.peek() {
                match token{
                    Token::Word(_) => {
                        return contains_whitspace;
                    },
                    Token::Symbol(c) => {
                        if c.is_whitespace(){
                            tokens.next();
                            contains_whitspace = true;
                        }
                        else{
                            return contains_whitspace;
                        }
                    }
                }
            }
            contains_whitspace
        }

        pub fn expect_word<'a>(tokens : &mut impl Iterator<Item=Token<'a>>) -> Option<&'a str>{
            if let Some(token) = tokens.next() {
                match token{
                    Token::Word(name) => {
                        return Some(name);
                    },
                    Token::Symbol(_) => {
                        return None
                    }
                }
            }
            None
        }

        pub fn peek_char<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>) -> Option<char>{
            if let Some(Token::Symbol(c)) = tokens.peek() {
                return Some(*c)
            }
            None
        }

        pub fn peek_word<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>) -> Option<&'a str>{
            if let Some(Token::Word(name)) = tokens.peek() {
                return Some(*name);
            }
            None
        }

        pub fn skip_until_endifdef<'a>(tokens :  &mut Peekable<impl Iterator<Item=Token<'a>>>){
            let mut if_counter = 0;
            while let Some(token) = tokens.next() {
                match token{
                    Token::Word(_) => {
                    },
                    Token::Symbol(c) => {
                        if c == '\n'{
                            match_whitespace(tokens);
                            if let Some(token) = tokens.next(){
                                match token{
                                    Token::Symbol(c2) => 
                                    {
                                        if c2 == '#'{ //this is a preprocessor command:
                                            if let Some(w) = peek_word(tokens){

                                                if w == "if" || w == "ifdef" || w == "ifndef"{ //new if counter
                                                    if_counter+=1;
                                                }

                                                if w == "endif" || w == "else" || w == "elif" || w == "elifdef" || w == "elifndef"{ 
                                                    if if_counter == 0{
                                                        return;
                                                    }
                                                    else if w =="endif"{
                                                        if_counter -= 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Token::Word(_) => {},
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    pub type DefinesDefinitions = HashMap<String, DefineDefinition>;
    pub struct DefineDefinition{
        params : Vec<String>,
        content : String
    }
    
    impl DefineDefinition{
        pub fn new_func(params: Vec<String>, content: String) -> Self {
            Self { params, content }
        }
        pub fn new(content: String) -> Self {
            Self { params : Vec::new(), content }
        }
    }

    pub mod shader_defines{
        use std::{fs, iter::Peekable, path::PathBuf};
        use evalexpr::*;
        use fancy_regex::Regex;

        use crate::shader_loader::{shader_tokeniser::{expect_word, match_string, match_until_char, peek_char}, DefineDefinition};

        use super::{shader_tokeniser::{match_whitespace, Token, Tokenizer}, DefinesDefinitions};

       
        pub fn load_shader(path: &PathBuf, defines :  &mut DefinesDefinitions, global_defines : &Vec<&'static str>) -> String {
            let shader_code = fs::read_to_string(path).expect("Failed to read WGSL file");
            
            let shader_code = Regex::new(r"//.*\n", ).unwrap().replace_all(&shader_code, "\n"); //remove coments

            let tokenizer = Tokenizer::new(&shader_code);
            let mut tokens = tokenizer.peekable();
            let mut result = String::new();

            let mut if_blocks : Vec<bool> = vec![];
            
            fn apply_defines_word<'a>(tokens: &mut Peekable<impl Iterator<Item=Token<'a>>>, word : &str, result :&mut String, defines: &mut DefinesDefinitions){
                if let Some(define) = defines.get(word) {
                    if !define.params.is_empty() {
                        // Function-like macro detected, ensure '(' follows
                        if let Some(Token::Symbol('(')) = tokens.peek() {
                            tokens.next(); // Consume '('

                            let mut args = Vec::new();
                            let mut current_arg = String::new();
                            let mut paren_level = 1; // Track nested parentheses

                            for t in tokens.by_ref() {
                                match t {
                                    Token::Symbol(')') => {
                                        paren_level -= 1;
                                        if paren_level == 0 {
                                            args.push(current_arg.trim().to_string());
                                            break;
                                        } else {
                                            current_arg.push(')');
                                        }
                                    }
                                    Token::Symbol('(') => {
                                        paren_level += 1;
                                        current_arg.push('(');
                                    }
                                    Token::Symbol(',') if paren_level == 1 => {
                                        args.push(current_arg.trim().to_string());
                                        current_arg.clear();
                                    }
                                    Token::Symbol(c) => {
                                        current_arg.push(c);
                                    }
                                    Token::Word(w) => {
                                        current_arg.push_str(w);
                                    }
                                }
                            }

                            // Substitute parameters in macro content
                            let mut expanded = define.content.clone();
                            for (param, arg) in define.params.iter().zip(args.iter()) {
                                expanded = expanded.replace(param, arg);
                            }

                            // Recursively process the expanded content
                            let expanded_result = apply_defines(&expanded, defines);
                            result.push_str(&expanded_result);
                        } else {
                            // If not followed by '(', treat it as a normal word
                            result.push_str(word);
                        }
                    } else {
                        // Normal macro replacement
                        result.push_str(&define.content);
                    }
                } else {
                    result.push_str(word);
                }
            }
            
            fn apply_defines_tokens<'a>(tokens: &mut Peekable<impl Iterator<Item=Token<'a>>>, defines: &mut DefinesDefinitions) -> String {
                let mut result = String::new();
            
                while let Some(token) = tokens.next() {
                    match token {
                        Token::Word(word) => {
                            apply_defines_word(tokens, word, &mut result, defines);
                        }
                        Token::Symbol(c) => result.push(c),
                    }
                }
                result
            }

            fn apply_defines(code: &str, defines: &mut DefinesDefinitions) -> String {
                let tokenizer = Tokenizer::new(code);
                let mut tokens = tokenizer.peekable();
                apply_defines_tokens(&mut tokens, defines)
            }
            

            fn match_preprocessor<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>, result : &mut String, defines :  &mut DefinesDefinitions, path: &PathBuf, if_blocks : &mut Vec<bool>, global_defines : &Vec<&'static str>){
                if match_whitespace(tokens){
                    result.push(' ');
                }

                if let Some(w) = expect_word(tokens){
                    if w == "include"{ //add content of file:
                        match_whitespace( tokens);
                        
                        let included_file = match_string(tokens);
                        let included_path = path.parent().unwrap().join(included_file);

                        let included_shader_content = load_shader(&included_path, defines, global_defines);
                        result.push_str(&included_shader_content);
                    }
                    else if w == "define"{
                      
                        match_whitespace(tokens);     
                        if let Some(key) = expect_word(tokens){
                            
                            let mut params = Vec::new();
                             // Check if this is a function-like macro (next token is '(')
                            if let Some(Token::Symbol('(')) = tokens.peek(){
                                tokens.next(); // Consume '('
                                
                                // Read parameter names until ')'
                                while let Some(Token::Word(param)) = tokens.peek() {
                                    params.push(param.to_string());
                                    tokens.next(); // Consume parameter
                    
                                    if let Some(Token::Symbol(',')) = tokens.peek() {
                                        tokens.next(); // Consume ','
                                    } else {
                                        break;
                                    }
                                }

                                // Consume closing ')'
                                if let Some(Token::Symbol(')')) = tokens.next() {
                                    // Successfully parsed parameter list
                                } else {
                                    panic!("Expected closing ')' in macro definition");
                                }
                            }

                            let mut value = String::new();
                            match_until_char(tokens, '\n', &mut value);
                            let condition_parsed = apply_defines(&value, defines);
                            // let tokenizer = Tokenizer::new(&value);
                            // let tokens_condition = tokenizer.peekable();
                            // let mut condition_parsed = String::new();

                            // for token in tokens_condition {
                            //     match token {
                            //         Token::Word(word) => {
                            //             if let Some(value) = defines.get(word){
                            //                 condition_parsed.push_str(value);
                            //             }    
                            //             else{
                            //                 condition_parsed.push_str(word);
                            //             }
                            //         },
                            //         Token::Symbol(c) => condition_parsed.push(c),
                            //     }
                            // }
                            defines.insert(key.to_string(), DefineDefinition::new_func(params, condition_parsed.trim().to_string()));
                        }
                    }
                    else if w == "definec"{
                      
                        match_whitespace(tokens);     
                        if let Some(key) = expect_word(tokens){
                            let mut value = String::new();
                            match_until_char(tokens, '\n', &mut value);
                           
                            let condition_parsed = apply_defines(&value, defines);
                            let condition_parsed = condition_parsed.trim().replace("u", "");

                            let result_exp = eval(&condition_parsed).unwrap();

                            let exp = match result_exp{
                                Value::Int(val) => format!("{val}u"),
                                Value::Boolean(bool) =>format!("{bool}"),
                                _ => panic!("could not match expression")
                            };
                            defines.insert(key.to_string(), DefineDefinition::new(exp));
                            //defines.insert(key.to_string(), condition_parsed.trim().to_string());
                        }
                    }
                    else if w == "ifdef" || w == "elifdef" || w == "ifndef" || w == "elifndef"{
                        let is_new_block = w == "ifdef" || w == "ifndef";
                        if !is_new_block{ //"elifdef" or "elifndef"
                            if let Some(last) = if_blocks.last(){
                                if *last{ //the last if was already true, so skip this:
                                    crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                    match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                                    return;
                                }
                            }
                        }

                        match_whitespace(tokens);     
                        if let Some(word) = expect_word(tokens){
                            if (defines.contains_key(word) || global_defines.contains(&word)) ^ w.contains("n"){ //n for ndef
                                if is_new_block{
                                    if_blocks.push(true);
                                }
                                else{
                                    *if_blocks.last_mut().unwrap() = true;
                                }
                            } else {
                                if is_new_block{
                                    if_blocks.push(false);
                                }
                                crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                            }
                        }
                    }
                    else if w == "if" || w == "elif"{
                        let is_new_block = w == "if";
                        //check if prev condition was true:
                        if w == "elif"{
                            if let Some(last) = if_blocks.last(){
                                if *last{ //the last if was already true, so skip this:
                                    crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                    match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                                    return;
                                }
                            }
                        }

                        match_whitespace(tokens);
                        let mut condition = String::new();
                        match_until_char(tokens, '\n', &mut condition);
                        let condition_parsed = apply_defines(&condition, defines);
                        let condition_parsed = condition_parsed.replace("u", "");

                        //validate if condition is true:
                        let result_exp = eval(&condition_parsed).unwrap();
                        if let Value::Boolean(result_exp) = result_exp{
                            if result_exp{
                                if is_new_block{
                                    if_blocks.push(true);
                                }
                                else{
                                    *if_blocks.last_mut().unwrap() = true;
                                }
                            }
                            else{
                                if is_new_block{
                                    if_blocks.push(false);
                                }
                                crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                            }
                        }
                        else {
                            if is_new_block{
                                if_blocks.push(false);
                            }
                            crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                            match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                        }

                    }
                    else if w == "assert"{
                        match_whitespace(tokens);
                        let mut condition = String::new();
                        match_until_char(tokens, '\n', &mut condition);
                        let condition_parsed = apply_defines(&condition, defines);
                        let condition_parsed = condition_parsed.replace("u", "");
                        //validate if condition is true:
                        
                        let result_exp = eval(&condition_parsed).unwrap(); //remove u, e.g. "32u > 42u"        ->  "32 > 42"  
                        if let Value::Boolean(result_exp) = result_exp{
                            if !result_exp{
                                panic!("Shader Assert in: {:?}: {condition_parsed}/({condition})", path);
                            }
                        }
                        else {
                            panic!("Shader Assert in: {:?}: {condition_parsed} was not a Boolean Expression", path);
                        }

                    }
                    else if w == "else"{
                        if let Some(last) = if_blocks.last(){
                            if *last{ //skip if tast if was true
                                crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                            }
                            else{
                                *if_blocks.last_mut().unwrap() = true;
                            }
                        }
                    }
                    else if w == "endif"{
                        if_blocks.pop();
                    }
                }
            }

            fn analyse_preprocessor<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>, result : &mut String, defines :  &mut DefinesDefinitions, path: &PathBuf, if_blocks : &mut Vec<bool>, global_defines : &Vec<&'static str>){
                if match_whitespace(tokens){
                    result.push(' ');
                }
                
                if let Some(token) = tokens.next(){
                    match token{
                        Token::Word(w) => {
                            apply_defines_word(tokens, w, result, defines);
                        },
                        Token::Symbol(c2) => 
                        {
                            if c2 == '#'{ //this is a preprocessor command:
                                match_preprocessor(tokens, result, defines, path, if_blocks, global_defines);
                            }
                            else{
                                let mut is_comment = false;
                                if c2 == '/'{
                                    if let Some(c) = peek_char(tokens){
                                        if c == '/'{ //this is a comment
                                            match_until_char(tokens, '\n', &mut String::new());
                                            is_comment = true;
                                        }
                                    }
                                }
                                if !is_comment{
                                    result.push(c2);
                                }
                            }
                        },
                    }
                }
            }

            analyse_preprocessor(&mut tokens, &mut result, defines, path, &mut if_blocks, global_defines); //first line could be a preprocessor command
            
            

            while let Some(token) = tokens.next() {
                match token {
                    Token::Word(word) => {
                        apply_defines_word(&mut tokens, word, &mut result, defines);
                    },
                    Token::Symbol(c) => {
                        if c == '\n'{
                            result.push(c);
                            analyse_preprocessor(&mut tokens, &mut result, defines, path, &mut if_blocks, global_defines);
                        }
                        else{
                            let mut is_comment = false;
                            if c == '/'{
                                if let Some(c) = peek_char(&mut tokens){
                                    if c == '/'{ //this is a comment
                                        match_until_char(&mut tokens, '\n', &mut String::new());
                                        is_comment = true;
                                    }
                                }
                            }
                            if !is_comment{
                                result.push(c);
                            }
                        }
                    }
                }
            }
            result
        }

    }

    pub mod shader_shortener{
        use std::collections::{HashMap, HashSet};

        use super::{shader_tokeniser::{match_function_block, match_until_char, match_variable, match_whitespace, Token, Tokenizer}, SHORTEN_GLOBAL_FUNCTIONS, SHORTEN_NORMAL_FUNCTIONS, SHORTEN_NORMAL_VARIABLES, SHORTEN_OVERRIDES};
        pub struct ShaderInfo{
            pub global_functions : Vec<(String, String)>,
            pub global_overrides : HashMap<String, String>,
            pub global_function_counter : usize,
            pub global_overrides_counter : usize,
        }

        impl ShaderInfo{
            pub fn contains_function(&self, name: &str) -> bool{
                for (k, _v) in &self.global_functions{
                    if k == name{
                        return true;
                    }
                }
                false
            }
        }

        impl ShaderInfo{
            pub fn new() -> ShaderInfo{
                ShaderInfo{ global_functions: Vec::new(), global_overrides: HashMap::new(), global_function_counter: 1, global_overrides_counter: 1 }
            }

            pub fn shorten_variable_names(&mut self, shader_code: &str) -> String {
                let tokenizer = Tokenizer::new(shader_code);
                let mut result = String::new();
                let mut variables = HashMap::new();
                let mut functions = HashMap::new();
                let mut var_counter = 1;
                let mut is_compute_fn = false;
                let mut tokens = tokenizer.peekable();
                let mut prev_token = None;

                while let Some(token) = tokens.next() {
                    match token {
                        Token::Word(word) if word == "let" || word=="var" || word=="const" => {
                            result.push_str(word);
                            let var_name = match_variable(&mut tokens, &mut result);
                            if !var_name.is_empty() {
                                let short_name = variables.entry(var_name.clone()).or_insert_with(||
                                    {
                                        if SHORTEN_NORMAL_VARIABLES{
                                            let (name, new_counter ) = generate_short_name(var_counter);
                                            var_counter = new_counter;
                                            name
                                        }
                                        else{
                                            var_name.to_string()
                                        }
                                });
                                result.push_str(short_name);
                            }
                        }
                        Token::Word(word) if word == "fn" => {
                            result.push_str(word);

                            let var_name = match_variable(&mut tokens, &mut result);
                            if !var_name.is_empty(){
                                if is_compute_fn{

                                    let short_name;
                                    if self.contains_function(&var_name){
                                        short_name = self.global_functions.iter().find(|(k, _)| k == &var_name).unwrap().1.to_string();
                                    }
                                    else{
                                        if SHORTEN_GLOBAL_FUNCTIONS{
                                            let (n, new_counter ) = generate_short_name(self.global_function_counter);
                                            self.global_function_counter = new_counter;
                                            short_name = format!("z{n}");
                                          
                                        }
                                        else{
                                            short_name = var_name.clone();
                                        }
                                        self.global_functions.push((var_name.to_string(), short_name.to_string()));          
                                    }
                                    result.push_str(&short_name);
                                    is_compute_fn = false;
                                }
                                else{
                                    let short_name = functions.entry(var_name.to_string()).or_insert_with(||
                                        {
                                            if SHORTEN_NORMAL_FUNCTIONS{
                                                let (name, new_counter ) = generate_short_name(var_counter);
                                                var_counter = new_counter;
                                                name
                                            }
                                            else{
                                                var_name.to_string()
                                            }
                                        });
                                        
                                    result.push_str(short_name);
                                }
                            }
                        }
                        Token::Word(word) if word=="struct" => {
                            result.push_str(word);
                            let var_name = match_variable(&mut tokens, &mut result);
                            result.push_str(&var_name);
                            match_function_block(&mut tokens, &mut result);
                        }
                        Token::Word(word) if word == "default" => {
                            result.push_str(word);
                        }
                        Token::Word(word) if word == "override" => {
                            result.push_str(word);
                            let var_name = match_variable(&mut tokens, &mut result);
                            if !var_name.is_empty(){
                                

                                let short_name: String;
                                if self.global_overrides.contains_key(&var_name){
                                    short_name = self.global_overrides.get(&var_name).unwrap().to_string();
                                }
                                else{
                                    if SHORTEN_OVERRIDES{
                                        let (n, new_counter ) = generate_short_name(self.global_overrides_counter);
                                        self.global_overrides_counter = new_counter;
                                        short_name = format!("y{n}");
                                      
                                    }
                                    else{
                                        short_name = var_name.clone();
                                    }
                                    self.global_overrides.insert(var_name.to_string(), short_name.clone());
                                }
                                result.push_str(&short_name);
                               
                            }
                        }
                        Token::Word(word) => {
                            let mut valid = true;
                            if let Some(Token::Symbol(c)) = prev_token{
                                if c == '.' || c.is_alphanumeric(){
                                    result.push_str(word);
                                    valid = false;
                                }
                            }
                            if valid{
                                let removec_whitespace = match_whitespace(&mut tokens);
                            
                                if let Some(Token::Symbol(c)) = tokens.peek(){ //this may be a function call
                                    if *c == '('{ //this is a function call
                                        if let Some(short_name) = functions.get(word) {
                                            result.push_str(short_name); 
                                        } else {
                                            result.push_str(word);
                                        } 
                                        valid = false;
                                    }
                                    else if *c == ':'{ //this is a variable definition, e.g. v1 : u32, v2 : u32
                                        let short_name = 
                                            variables.entry(word.to_string()).or_insert_with(||
                                                {
                                                    if SHORTEN_NORMAL_VARIABLES{
                                                        let (name, new_counter ) = generate_short_name(var_counter);
                                                        var_counter = new_counter;
                                                        name
                                                    }
                                                    else{
                                                        word.to_string()
                                                    }
                                                });

                                        result.push_str(short_name);
                                        valid=false;
                                    }
                                }
                                if valid{
                                    if let Some(short_name) = self.global_overrides.get(word){
                                        result.push_str(short_name);
                                    }
                                    else if let Some(short_name) = variables.get(word) {
                                        result.push_str(short_name); 
                                    } else {
                                        result.push_str(word);
                                    } 
                                }
                                if removec_whitespace{
                                    result.push(' ');
                                }
                            }
                        }
                        Token::Symbol(c) => {
                            if c == '@'{
                                if let Some(Token::Word(word)) = tokens.peek(){
                                    if *word == "compute"{
                                        is_compute_fn = true;
                                    }
                                }
                            }
                            result.push(c);}
                    }
                    prev_token = Some(token);
                }

                result
            }



            pub fn remove_unused(&self, shader_code : &str) -> String{
                let tokenizer = Tokenizer::new(shader_code);
                let tokens = tokenizer.peekable();

                let mut defined_variables = HashSet::new();

                let mut used_variables = HashSet::new();
                let mut last_token = None;

                for token in tokens{
                    match token{
                        Token::Word(w) => {                
                            if defined_variables.contains(w){
                                //Check for 42e+42 
                                //here e is not a variable that is used. no number can be placed before a variable.
                                if let Some(Token::Symbol(s)) = &last_token{
                                    if s.is_numeric(){
                                        continue;
                                    }
                                }

                                used_variables.insert(w.to_string());
                            }
                            defined_variables.insert(w.to_string());
                        },
                        Token::Symbol(_) => {},
                    }
                    last_token = Some(token);
                }

                let tokenizer = Tokenizer::new(shader_code);
                let mut tokens = tokenizer.peekable();
            
                let mut result = String::new();    
                let mut is_compute_fn = false;
                let mut current_item = String::new();
                while let Some(token) = tokens.next(){
                    match token{
                        Token::Word(w) => 
                        {
                            if w == "const" || w == "var" || w == "override"{//we are a global variable
                                current_item.push_str(w);
                                let var_name = match_variable(&mut tokens, &mut current_item);
                                current_item.push_str(&var_name);
                                match_until_char(&mut tokens, ';', &mut current_item);
                                tokens.next(); //remove ';' from the tokens list as well
                                current_item.push(';');
                                if used_variables.contains(&var_name){
                                    result.push_str(&current_item);
                                }

                                current_item = String::new();
                            }
                            else if w == "fn"{//we are a function
                                current_item.push_str(w);
                                let var_name = match_variable(&mut tokens, &mut current_item);
                                current_item.push_str(&var_name);
                                match_function_block(&mut tokens, &mut current_item);
                                if is_compute_fn  || used_variables.contains(&var_name){
                                    result.push_str(&current_item);
                                }
                                is_compute_fn = false;
                                current_item = String::new();
                            }
                            else if w == "struct"{ //we are a struct
                                current_item.push_str(w);
                                let var_name = match_variable(&mut tokens, &mut current_item);
                                current_item.push_str(&var_name);
                                match_function_block(&mut tokens, &mut current_item);
                                if used_variables.contains(&var_name){
                                    result.push_str(&current_item);
                                }
                                current_item = String::new();
                            }
                            else{
                                current_item.push_str(w);
                            }
                        },
                        Token::Symbol(c) => {
                            if c == '@'{
                                if let Some(Token::Word(word)) = tokens.peek(){
                                    if *word == "compute"{
                                        is_compute_fn = true;
                                    }
                                }
                            }
                            current_item.push(c);
                        }                    
                    }
                }
                result
            }

        }

        fn generate_short_name(counter: usize) -> (String,usize) {
            let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let mut name = String::new();
            let mut count = counter;
            let reserved_words = ["as", "if", "do", "fn", "of"];

            while count > 0 {
                name.push(alphabet.chars().nth((count - 1) % alphabet.len()).unwrap());
                count = (count - 1) / alphabet.len();
            }
            
            let name = name.chars().rev().collect::<String>();

            if reserved_words.contains(&name.as_str()) {
                generate_short_name(counter + 1)
            } else {
                (name, counter + 1)
            }

        }
    }
}


