use std::collections::HashMap;
use std::{env, fs};
use std::path::{Path, PathBuf};

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

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");

    let candle_core_path = PathBuf::from(&manifest_dir);
    let absolute_candle_core_path = fs::canonicalize(&candle_core_path).expect("Failed to get absolute outer folder path");

    let shader_dir = candle_core_path.join("src/wgpu_backend/wgpu_functions");
    if !shader_dir.exists(){
        panic!("could not find shader path {:?}", shader_dir);
    }
    println!("Searching Shaders in: {:?}", shader_dir);
    let output_file = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shader_map.rs");

    let mut shader_map = HashMap::new();
    let mut virtual_id_counter = 1;

 

    for file in get_all_wgsl_files(shader_dir){
        println!("Found File: {:?}", file);
        let shader_content = preprocess_shader(&file, &mut shader_map, &mut virtual_id_counter);
        
        shader_map.insert(virtual_id_counter, shader_content);
        

        let parent_dir = file.parent().unwrap();
        let generated_dir = parent_dir.join("generated");
        fs::create_dir_all(&generated_dir).unwrap();
        let original_file_name = file.file_name().unwrap().to_str().unwrap();
       

        //create the File:
        let debug_s = shader_loader::load_shader(virtual_id_counter, &shader_map, &vec!["f32"]);
        let new_file_name = format!("{}_generated_f32.wgsl", original_file_name);
        let new_file_path = generated_dir.join(new_file_name);
        fs::write(new_file_path, debug_s).expect("Failed to write shader file");
     

        let debug_s = shader_loader::load_shader(virtual_id_counter, &shader_map, &vec!["u32"]);
        let new_file_name = format!("{}_generated_u32.wgsl", original_file_name);
        let new_file_path = generated_dir.join(new_file_name);
        fs::write(new_file_path, debug_s).expect("Failed to write shader file");

        let debug_s = shader_loader::load_shader(virtual_id_counter, &shader_map, &vec!["u8"]);
        let new_file_name = format!("{}_generated_u8.wgsl", original_file_name);
        let new_file_path = generated_dir.join(new_file_name);
        fs::write(new_file_path, debug_s).expect("Failed to write shader file");
        
        let absolute_file_path = fs::canonicalize(&file).expect("Failed to get absolute file path");
        match absolute_file_path.strip_prefix(&absolute_candle_core_path) {
            Ok(relative_path) =>  println!("cargo::rerun-if-changed={}", relative_path.to_string_lossy()),
            Err(_) => println!("File path is not inside the outer folder"),
        }
        virtual_id_counter += 1;
    }

    // Generate the shader map file
    let mut shader_map_content = String::new();
    shader_map_content.push_str("use std::collections::HashMap;\n\n");
    shader_map_content.push_str("pub fn get_shader_map() -> HashMap<u32, &'static str> {\n");
    shader_map_content.push_str("    let mut map = HashMap::new();\n");

    for (id, content) in &shader_map {
        shader_map_content.push_str(&format!("    map.insert({}, r#\"{}\"#);\n", id, content));
    }

    shader_map_content.push_str("    map\n");
    shader_map_content.push_str("}\n");
    
    fs::write(output_file, shader_map_content).expect("Failed to write shader map file");
}

fn preprocess_shader(
    path: &PathBuf,
    shader_map: &mut HashMap<u32, String>,
    virtual_id_counter: &mut u32,
) -> String {
    println!("Process File: {:?}", path);
    let contents = fs::read_to_string(path).expect("Failed to read WGSL file");
    let mut result = String::new();
    let mut lines = contents.lines();

    while let Some(line) = lines.next() {
        if line.starts_with("#include") {
          
            let included_file = line.split_whitespace().nth(1).expect("No file specified in #include");
            let included_file = included_file.trim_matches('"');
            let included_path = path.parent().unwrap().join(included_file);
            println!("Found Include: {:?}", included_path);
            let included_shader_content = preprocess_shader(&included_path, shader_map, virtual_id_counter);
            result.push_str(&format!("#include_virtual {}\n", virtual_id_counter));
            shader_map.insert(*virtual_id_counter, included_shader_content);



            *virtual_id_counter += 1;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}



mod shader_loader{
    use fancy_regex::Regex;
    use std::collections::HashMap;

    pub fn load_shader<T : AsRef<str>>(virtual_id: u32, shader_map: &std::collections::HashMap<u32, T>, defines: &Vec<&str>) -> String {
        println!("Load Shader {virtual_id}");
        let mut shader_code : String= (*shader_map.get(&virtual_id).expect("Shader ID not found")).as_ref().to_string();
        while let Some(include_pos) = shader_code.find("#include_virtual") {
            let start_pos = include_pos + "#include_virtual".len();
            let end_pos = shader_code[start_pos..].find('\n').unwrap() + start_pos;
            let virtual_id_str = &shader_code[start_pos..end_pos].trim();
            let include_id: u32 = virtual_id_str.parse().expect("Invalid virtual ID");
            let include_content = (*shader_map.get(&include_id).expect("Included shader ID not found")).as_ref();
            shader_code.replace_range(include_pos..end_pos, include_content);
        }
        
        let result = apply_defines(&shader_code, defines);
        
        let result = Regex::new(r"//.*\n", ).unwrap().replace_all(&result, "");
        let result = Regex::new(r"((\s+)(?![\w\s])|(?<!\w)(\s+))", ).unwrap().replace_all(&result, "");
        
        let result = shader_shortener::shorten_variable_names(&result);
        
        //let result = result.replace("\n", "");

        result
    }
    
    fn apply_defines(shader_code: &str, global_defines: &Vec<&str>) -> String {
        let mut result = String::new();
        let mut lines = shader_code.lines().peekable();
        let mut inside_ifdef = false;
        let mut defines : HashMap<String, String> = HashMap::new();
        while let Some(line) = lines.next() {
            if line.starts_with("#ifdef") {
                let key = line.split_whitespace().nth(1).expect("No key specified in #ifdef");
                if defines.contains_key(key) || global_defines.contains(&key) {
                    inside_ifdef = true;
                } else {
                    skip_until_else_or_endif(&mut lines);
                }
            } else if line.starts_with("#else") {
                if inside_ifdef {
                    inside_ifdef = false;
                    skip_until_endif(&mut lines);
                }
            } else if line.starts_with("#endif") {
                inside_ifdef = false;
            } else if line.starts_with("#define") {
                let mut parts = line.split_whitespace();
                parts.next(); // Skip the directive
                let key = parts.next().expect("No key specified in #define").to_string();
                let value = parts.next().unwrap_or("").to_string();
                defines.insert(key, value);
            } else {
                let processed_line = replace_defines(line, &defines);
                result.push_str(&processed_line);
                result.push('\n');
            }
        }
    
        result
    }
    
    fn replace_defines(line: &str, defines: &HashMap<String, String>) -> String {
        let mut processed_line = String::from(line);
        for (key, value) in defines {
            processed_line = processed_line.replace(key, value);
        }
        for (key, value) in defines {
            processed_line = processed_line.replace(key, value);
        }
        processed_line
    }
    
    fn skip_until_else_or_endif<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>) {
        while let Some(line) = lines.next() {
            if line.starts_with("#else") || line.starts_with("#endif") {
                break;
            }
        }
    }
    
    fn skip_until_endif<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>) {
        while let Some(line) = lines.next() {
            if line.starts_with("#endif") {
                break;
            }
        }
    }

    mod shader_shortener{
        use std::collections::HashMap;
        use std::str::Chars;

        #[derive(Debug, PartialEq, Eq)]
        enum Token<'a> {
            Word(&'a str),
            Symbol(char)
        }

        struct Tokenizer<'a> {
            chars: Chars<'a>,
            input: &'a str,
            pos: usize,
        }

        impl<'a> Tokenizer<'a> {
            fn new(input: &'a str) -> Self {
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
                while let Some(c) = self.chars.next() {
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

        pub fn shorten_variable_names(shader_code: &str) -> String {
            let tokenizer = Tokenizer::new(shader_code);
            let mut result = String::new();
            let mut variables = HashMap::new();
            let mut var_counter = 1;

            let mut tokens = tokenizer.peekable();
            let mut prev_token = None;
            while let Some(token) = tokens.next() {
                match token {
                    Token::Word(word) if word == "let" || word=="var" || word=="const" => {
                        result.push_str(word);
                        let mut generic_counter = 0;
                        while let Some(token) = tokens.next() {
                            match token{
                                Token::Word(var_name) => {
                                    if(generic_counter == 0){
                                        let short_name = variables.entry(var_name).or_insert_with(||
                                             {
                                            let (name, new_counter ) = generate_short_name(var_counter);
                                            var_counter = new_counter;
                                            name
                                        });
                                       
                                        result.push_str(short_name);
                                        break;
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
                    }
                    Token::Word(word) => {
                        let mut valid = true;
                        if let Some(p_token) = prev_token{
                            if let Token::Symbol(c) = p_token{
                                if c == '.'{
                                    result.push_str(word);
                                    valid = false;
                                }
                            }
                        }
                        if valid{
                            if let Some(short_name) = variables.get(word) {
                                result.push_str(short_name); 
                            } else {
                                result.push_str(word);
                            } 
                        }
                    }
                    Token::Symbol(c) => result.push(c)
                }
                prev_token = Some(token);
            }

            result
        }

    }
}


