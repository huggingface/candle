use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::shader_loader::{DefinesDefinitions, shader_shortener};

const SHORTEN_NORMAL_VARIABLES : bool = false;
const SHORTEN_NORMAL_FUNCTIONS : bool = false;
const SHORTEN_GLOBAL_FUNCTIONS : bool = false;
const SHORTEN_OVERRIDES : bool = false;
const REMOVE_COMMENT : bool = false;
const REMOVE_UNUSED : bool = true;
const REMOVE_SPACES : bool = false;
const REMOVE_NEW_LINES : bool = false;

pub type ShaderStore = HashMap<PathBuf, Cow<'static, str>>;

#[derive(Clone)]
pub struct ParseState{
    defines : DefinesDefinitions, 
    path : PathBuf,
    shader_info : shader_shortener::ShaderInfo
}

impl ParseState{
    pub fn new() -> Self{
        ParseState{
                defines: HashMap::new(),
                path: Path::new("").to_path_buf(),
                shader_info : shader_loader::shader_shortener::ShaderInfo::new(),
            }
    }

    pub fn set_path(&mut self, new_path : PathBuf){
        self.path = new_path;
    }

    pub fn set_defines(&mut self, global_defines : DefinesDefinitions){
        self.defines.clear();
        self.defines = global_defines;
    }

    pub fn info(&self) -> &shader_shortener::ShaderInfo{
        &self.shader_info
    }

    pub fn reset_global_functions(&mut self) -> Vec<(String, String)>{
        self.shader_info.global_function_counter = 1;
        std::mem::take(&mut self.shader_info.global_functions)
    }
}

impl Default for ParseState {
    fn default() -> Self {
        Self::new()
    }
}

pub mod shader_loader{
    use std::collections::HashMap;
    use fancy_regex::Regex;
    use super::*;

    pub fn load_shader(state : &mut ParseState, store : &ShaderStore) -> String {
        let mut result = shader_defines::load_shader(state, store); 
        
        if REMOVE_SPACES{
            result = Regex::new(r"((\s+)(?![\w\s])|(?<!\w)(\s+))", ).unwrap().replace_all(&result, "").to_string(); //replaces newline and not used spaces
        }
        result = state.shader_info.shorten_variable_names(&result);
        if REMOVE_UNUSED{
            loop{
                let new_result = state.shader_info.remove_unused(&result);
                if new_result == result{
                    break;
                }
                result = new_result;
            }
           
        }
        if REMOVE_NEW_LINES{
            result = result.replace("\r", "").replace("\n", "");
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

        #[derive(Debug, Clone)]
        pub struct Tokenizer<'a> {
            chars: Peekable<Chars<'a>>,
            input: &'a str,
            pos: usize,
        }

        impl<'a> Tokenizer<'a> {
            pub fn new(input: &'a str) -> Self {
                Tokenizer {
                    chars: input.chars().peekable(),
                    input,
                    pos: 0,
                }
            }
        }

        impl<'a> Iterator for Tokenizer<'a> {
            type Item = Token<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                let c = self.chars.next()?;
                let c_len = c.len_utf8();
                let start_pos = self.pos;
                self.pos += c_len;

                if c.is_alphabetic() || c == '_' {
                    while let Some(&next_c) = self.chars.peek() {
                        if next_c.is_alphanumeric() || next_c == '_' {
                            let nc = self.chars.next().unwrap();
                            self.pos += nc.len_utf8();
                        } else {
                            break;
                        }
                    }
                    Some(Token::Word(&self.input[start_pos..self.pos]))
                } else {
                    Some(Token::Symbol(c))
                }
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

        pub fn match_until_newline_no_comment<'a>(
            tokens: &mut Peekable<impl Iterator<Item = Token<'a>>>,
            result: &mut String,
        ) {
            while let Some(token) = tokens.peek().cloned() {
                match token {
                    Token::Word(word) => {
                        tokens.next();
                        result.push_str(word);
                    }
                    
                    Token::Symbol('\r') => {
                        //Stop at newline
                        return;
                    }
                    Token::Symbol('\n') => {
                        //Stop at newline
                        return;
                    }

                    Token::Symbol('/') => {
                        //Possible start of a comment
                        tokens.next(); // consume first '/'

                        match tokens.peek().cloned() {
                            Some(Token::Symbol('/')) => {
                                //this is a comment

                                tokens.next(); // consume second '/'

                                //Skip everything until newline (DO NOT push)
                                while let Some(token) = tokens.peek().cloned() {
                                    match token {
                                        Token::Symbol('\n') => return,
                                        _ => {
                                            tokens.next();
                                        }
                                    }
                                }

                                return;
                            }
                            //Not a comment → this was just a '/'
                            _ => {
                                result.push('/');
                            }
                        }
                    }

                    Token::Symbol(c) => {
                        tokens.next();
                        result.push(c);
                    }
                }
            }
        }



        pub fn match_until_pp_end<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>, result : &mut String){
            while let Some(token) = tokens.next() {
                match token{
                    Token::Word(var_name) => {
                        result.push_str(var_name);
                    },
                    Token::Symbol(c) => {
                        result.push(c);
                        if c == '\n'{
                            match_whitespace_to_result(tokens, result);
                            if let Some(token) = tokens.next(){
                                match token{
                                    Token::Symbol(c2) => 
                                    {
                                        if c2 == '#'{ //this is a preprocessor command:
                                            if let Some(w) = peek_word(tokens){
                                                if w == "pp_end"{
                                                    tokens.next();
                                                    return;
                                                }
                                                else{
                                                    result.push(c2);
                                                }
                                            }
                                        }
                                        else{
                                            result.push(c2);
                                        }
                                    }
                                    Token::Word(word) => { result.push_str(word); },
                                }
                            }
                        }
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

        pub fn match_whitespace_to_result<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>, result : &mut String) -> bool{
            let mut contains_whitspace = false;
            while let Some(token) = tokens.peek() {
                match token{
                    Token::Word(_) => {
                        return contains_whitspace;
                    },
                    Token::Symbol(c) => {
                        if c.is_whitespace(){
                            result.push(*c);
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

        pub fn match_whitespace_no_newline<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>>>) -> bool{
            let mut contains_whitspace = false;
            while let Some(token) = tokens.peek() {
                match token{
                    Token::Word(_) => {
                        return contains_whitspace;
                    },
                    Token::Symbol(c) => {
                        if c.is_whitespace() && *c != '\n' && *c != '\r'{
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

        pub fn expect_char<'a>(tokens : &mut impl Iterator<Item=Token<'a>>, ch : char){
            if let Some(token) = tokens.next() {
                match token{
                    Token::Word(name) => {
                        panic!("expected char: '{}', but got: '{}'", ch, name);
                    },
                    Token::Symbol(c) => {
                        if c == ch{
                            return ;
                        }
                        let lookahead: Vec<_> = tokens.take(10).collect();
                        panic!(
                            "expected char '{}', but got {:?}\nnext tokens: {:?}",
                            ch, token, lookahead
                        );
                    }
                }
            }
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


    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    pub struct DefineDefinition{
        params : Vec<String>,
        content : Option<String>, 
        load_shader_when_used : bool,
    }
    
    impl DefineDefinition{
        pub fn new_func(params: Vec<String>, content: String, load_shader_when_used: bool) -> Self {
            Self { params, content: Some(content), load_shader_when_used }
        }
        pub fn new(content: String) -> Self {
            Self { params : Vec::new(), content: Some(content), load_shader_when_used : false }
        }

        pub fn new_empty() -> Self {
            Self { params : Vec::new(), content: None, load_shader_when_used : false }
        }

        pub fn value<T: ToString>(value: T) -> Self {
            Self {
                params: Vec::new(),
                content: Some(value.to_string()),
                load_shader_when_used: false,
            }
        }
    }

    pub mod shader_defines{
        use crate::{ParseState, ShaderStore};
        use std::{iter::Peekable, path::{Component, Path, PathBuf}};
        use evalexpr::*;
        use fancy_regex::Regex;

        use crate::{REMOVE_COMMENT, REMOVE_SPACES, shader_loader::{DefineDefinition, shader_tokeniser::{expect_char, expect_word, match_string, match_until_char, match_until_newline_no_comment, match_until_pp_end, match_whitespace_no_newline, match_whitespace_to_result, peek_char}}};

        use super::{shader_tokeniser::{match_whitespace, Token, Tokenizer}};

        pub fn load_shader_content(shader_code : &str, state : &mut ParseState, store : &ShaderStore) -> String{
            let tokenizer = Tokenizer::new(shader_code);
            let mut tokens = tokenizer.peekable();
            let mut result = String::new();

            let mut if_blocks : Vec<bool> = vec![];
            
            fn apply_defines_word<'a>(tokens: &mut Peekable<impl Iterator<Item=Token<'a>>>, word : &str, result :&mut String, state : &mut ParseState, store : &ShaderStore){
                if let Some(define) = state.defines.get(word).cloned() {
                    if !define.params.is_empty() {
                        
                        // Function-like macro detected, ensure '(' follows
                        if let (Some(Token::Symbol('(')), Some(expanded)) = (tokens.peek(), &define.content) {
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
                            let mut new_state = state.clone();
                            new_state.defines.clear();
                            for (param, arg) in define.params.iter().zip(args.iter()) {
                                new_state.defines.insert(param.clone(), DefineDefinition::new(arg.clone()));
                            }

                            let mut expanded = apply_defines(expanded, &mut new_state, store);

                            if define.load_shader_when_used{
                                expanded = load_shader_content(&expanded, state, store)
                            }
                            
                            // Recursively process the expanded content
                            let expanded_result = apply_defines(&expanded, state, store);
                            result.push_str(&expanded_result);
                            
                        } else {
                            // If not followed by '(', treat it as a normal word
                            result.push_str(word);
                        }
                    } else if define.load_shader_when_used{
                        if let Some(content) = &define.content{
                            result.push_str(&load_shader_content(content, state, store));
                        }
                        else{
                            result.push_str(word);
                        }
                    }
                    else{
                        // Normal macro replacement
                        if let Some(content) = &define.content{
                            result.push_str(content);
                        }
                        else{
                            result.push_str(word);
                        }
                    }
                } else {
                    result.push_str(word);
                }
            }
            
            fn apply_defines_tokens<'a>(tokens: &mut Peekable<impl Iterator<Item=Token<'a>>>, state : &mut ParseState, store : &ShaderStore) -> String {
                let mut result = String::new();
            
                while let Some(token) = tokens.next() {
                    match token {
                        Token::Word(word) => {
                            apply_defines_word(tokens, word, &mut result, state, store);
                        }
                        Token::Symbol(c) => result.push(c),
                    }
                }
                result
            }

            fn apply_defines(code: &str, state : &mut ParseState, store : &ShaderStore) -> String {
                let tokenizer = Tokenizer::new(code);
                let mut tokens = tokenizer.peekable();
                apply_defines_tokens(&mut tokens, state, store)
            }

            fn match_preprocessor<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>> + std::clone::Clone>, result : &mut String, state : &mut ParseState, if_blocks : &mut Vec<bool>, store : &ShaderStore)
            {
                if REMOVE_SPACES{
                    if match_whitespace(tokens){
                        result.push(' ');
                    }
                }
                else{
                    match_whitespace_to_result(tokens, result);
                }

                if let Some(w) = expect_word(tokens){
                    if w == "include"{ //add content of file:
                        match_whitespace(tokens);
                        
                        let included_file = match_string(tokens);
                        let included_path = state.path.parent().unwrap().join(included_file);
                        
                        fn normalize_path(path: &Path) -> PathBuf {
                            let mut normalized = PathBuf::new();

                            for component in path.components() {
                                match component {
                                    Component::Prefix(prefix) => {
                                        // Windows drive letter or UNC prefix
                                        normalized.push(prefix.as_os_str());
                                    }
                                    Component::RootDir => {
                                        normalized.push(Component::RootDir.as_os_str());
                                    }
                                    Component::Normal(c) => {
                                        normalized.push(c);
                                    }
                                    Component::ParentDir => {
                                        normalized.pop();
                                    }
                                    Component::CurDir => {}
                                }
                            }

                            normalized
                        }
                        //println!("Including file: {:?}", &included_path);
                        let included_path = normalize_path(&included_path);

                        //println!("Including file: {:?}", &included_path);
                        let original_path = state.path.clone();
                        state.path = included_path;
                        let included_shader_content = load_shader(state, store);
                        state.path = original_path;
                        result.push_str(&included_shader_content);
                    }
                    else if w == "define"{
                        match_whitespace_no_newline(tokens);     
                        if let Some(key) = expect_word(tokens){
                            //println!("parsing define: '{key}'");
                            match_whitespace_no_newline(tokens);     
                            let mut params = Vec::new();
                            // Check if this is a function-like macro (next token is '(')
                            if let Some(Token::Symbol('(')) = tokens.peek(){
                                tokens.next(); // Consume '('
                                
                                // Read parameter names until ')'
                                while let Some(Token::Word(param)) = tokens.peek() {
                                    params.push(param.to_string());
                                    tokens.next(); // Consume parameter

                                    match_whitespace_no_newline(tokens);
                                    if let Some(Token::Symbol(',')) = tokens.peek() 
                                    {
                                        tokens.next(); // Consume
                                    } else {
                                        break;
                                    }
                                    match_whitespace_no_newline(tokens);
                                }
                                
                                expect_char(tokens, ')');
                            }

                            let mut value = String::new();
                            match_until_newline_no_comment(tokens, &mut value);
                            //println!("parsing define: '{}' = '{}'", key, value);
                            if value.is_empty(){
                                state.defines.insert(key.to_string(), DefineDefinition::new_empty());
                            }
                            else{
                                let condition_parsed = apply_defines(&value, state, store);
                                state.defines.insert(key.to_string(), DefineDefinition::new_func(params, condition_parsed.trim().to_string(), false));
                            }
                        }
                    }
                    else if w == "definec"{
                      
                        match_whitespace_no_newline(tokens);     
                        if let Some(key) = expect_word(tokens){
                            let mut value = String::new();
                            match_until_newline_no_comment(tokens, &mut value);
                           
                            let condition_parsed = apply_defines(&value, state, store);
                            let condition_parsed = condition_parsed.trim().replace("u", "");
                            let result_exp = eval(&condition_parsed);
                            let result_exp = match result_exp{
                                Ok(val) => val,
                                Err(e) => 
                                panic!(r#"definec failed!: 
                                File: '{}', 
                                Condition: '{}', 
                                Expanded:'{}'
                                Error: '{:?}')"#, state.path.display(), value.replace("\n", "").replace("\r", ""), condition_parsed.replace("\n", "").replace("\r", ""), e)
                            };
                            //println!("parsing definec: '{key}' = '{result_exp}'");
                            let exp = match result_exp{
                                Value::Int(val) => format!("{val}u"),
                                Value::Boolean(bool) =>format!("{bool}"),
                                _ => panic!("could not match expression")
                            };
                            state.defines.insert(key.to_string(), DefineDefinition::new(exp));
                        }
                    }
                    else if w == "pp_begin"{
                        match_whitespace_no_newline(tokens);     
                        if let Some(key) = expect_word(tokens){
                            match_whitespace_no_newline(tokens);     
                            let mut params = Vec::new();
                             // Check if this is a function-like macro (next token is '(')
                            if let Some(Token::Symbol('(')) = tokens.peek(){
                                tokens.next(); // Consume '('
                                
                                // Read parameter names until ')'
                                while let Some(Token::Word(param)) = tokens.peek() {
                                    params.push(param.to_string());
                                    tokens.next(); // Consume parameter
                    
                                    match_whitespace_no_newline(tokens);
                                    if let Some(Token::Symbol(',')) = tokens.peek() 
                                    {
                                        tokens.next(); // Consume
                                    } else {
                                        break;
                                    }
                                    match_whitespace_no_newline(tokens);
                                }

                                // Consume closing ')'
                                if let Some(Token::Symbol(')')) = tokens.next() {
                                    // Successfully parsed parameter list
                                } else {
                                    panic!("Expected closing ')' in macro definition");
                                }
                            }
                            //println!("match until pp_end for key: {}, params: {:?}", key, params);
                            let mut value: String = String::new();
                            match_until_pp_end(tokens, &mut value);
                            state.defines.insert(key.to_string(), DefineDefinition::new_func(params, value.trim().to_string(), true));
                        }
                    }
                    else if w == "ifdef" || w == "elifdef" || w == "ifndef" || w == "elifndef"{
                        let is_new_block = w == "ifdef" || w == "ifndef";
                        if !is_new_block{ //"elifdef" or "elifndef"
                            if let Some(last) = if_blocks.last(){
                                if *last{ //the last if was already true, so skip this:
                                    crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                    match_preprocessor(tokens, result, state, if_blocks, store);
                                    return;
                                }
                            }
                        }
                        
                        match_whitespace_no_newline(tokens);     
                        if let Some(word) = expect_word(tokens){
                            if (state.defines.contains_key(word)) ^ w.contains("n"){ //n for ndef
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
                                match_preprocessor(tokens, result, state, if_blocks, store);
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
                                    match_preprocessor(tokens, result, state, if_blocks, store);
                                    return;
                                }
                            }
                        }

                        match_whitespace(tokens);
                        let mut condition = String::new();
                        match_until_newline_no_comment(tokens, &mut condition);
                        let condition_parsed = apply_defines(&condition, state, store);
                        let condition_parsed = condition_parsed.replace("u", "");

                        //validate if condition is true:
                         
                        let result_exp =
                        match eval(&condition_parsed){
                            Ok(c) => c,
                            Err(err) => panic!("Error while checking #{w}: '{condition}': {err}")
                        }; //remove u, e.g. "32u > 42u"        ->  "32 > 42"  
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
                                match_preprocessor(tokens, result, state, if_blocks, store);
                            }
                        }
                        else {
                            if is_new_block{
                                if_blocks.push(false);
                            }
                            crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                            match_preprocessor(tokens, result, state, if_blocks, store);
                        }

                    }
                    else if w == "assert"{
                        match_whitespace(tokens);
                        let mut condition = String::new();
                        match_until_newline_no_comment(tokens, &mut condition);
                        let condition_parsed = apply_defines(&condition, state, store);
                        let condition_parsed = condition_parsed.replace("u", "").replace("\n", "");
                        //validate if condition is true:
                        
                        let result_exp =
                        match eval(&condition_parsed){
                            Ok(c) => c,
                            Err(err) => panic!("Error while checking assert: '{condition}': {err}")
                        }; //remove u, e.g. "32u > 42u"        ->  "32 > 42"  
                        if let Value::Boolean(result_exp) = result_exp{
                            if !result_exp{
                                panic!(r#"Shader assertion failed!: 
                                File: '{}', 
                                Condition: '{}', 
                                Expanded:'{}')"#, state.path.display(), condition.replace("\n", "").replace("\r", ""), condition_parsed.replace("\n", "").replace("\r", ""));
                            }
                        }
                        else {
                            panic!(r#"Shader assertion was not a Boolean Expression!: 
                            File: '{}', 
                            Condition: '{}', 
                            Expanded:'{}')"#, state.path.display(), condition.replace("\n", "").replace("\r", ""), condition_parsed.replace("\n", "").replace("\r", ""));
                        }

                    }
                    else if w == "else"{
                        if let Some(last) = if_blocks.last(){
                            if *last{ //skip if tast if was true
                                crate::shader_loader::shader_tokeniser::skip_until_endifdef(tokens);
                                match_preprocessor(tokens, result, state, if_blocks, store);
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

            fn analyse_preprocessor<'a>(tokens : &mut Peekable<impl Iterator<Item=Token<'a>> + std::clone::Clone>, result : &mut String, state : &mut ParseState, if_blocks : &mut Vec<bool>, store : &ShaderStore) -> bool{
                let mut whitespace = String::new();
                match_whitespace_to_result(tokens, &mut whitespace);
                
                if let Some(token) = tokens.next(){
                    match token{
                        Token::Word(w) => {
                            if REMOVE_SPACES{
                                if !whitespace.is_empty(){
                                    result.push(' ');
                                }
                            }
                            else{
                                result.push_str(&whitespace);
                            }
                            apply_defines_word(tokens, w, result, state, store);
                        },
                        Token::Symbol(c2) => 
                        {
                            if c2 == '#'{ //this is a preprocessor command, we will skip the whitespace at the beginning
                                match_preprocessor(tokens, result, state, if_blocks, store);
                                return true;
                            }
                            else{
                                if REMOVE_SPACES{
                                    if !whitespace.is_empty(){
                                        result.push(' ');
                                    }
                                }
                                else{
                                    result.push_str(&whitespace);
                                }                           
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
                false
            }

            analyse_preprocessor(&mut tokens, &mut result,state, &mut if_blocks, store); //first line could be a preprocessor command
            let mut was_last_preprocessor = false;
            while let Some(token) = tokens.next() {
                match token {
                    Token::Word(word) => {
                        apply_defines_word(&mut tokens, word, &mut result, state, store);
                    },
                    Token::Symbol(c) => {
                        if c == '\n' || c == '\r'{
                            if !was_last_preprocessor{
                                if c == '\r'{
                                    if let Some(Token::Symbol('\n')) = tokens.peek(){
                                        tokens.next();
                                        result.push_str("\r\n");
                                    }
                                }else{
                                    result.push(c);
                                }
                            }
                            else if c == '\r'{
                                if let Some(Token::Symbol('\n')) = tokens.peek(){
                                    tokens.next();
                                }
                            }
                            was_last_preprocessor = analyse_preprocessor(&mut tokens, &mut result, state, &mut if_blocks, store);
                        }
                        else{
                            result.push(c);
                        }
                    }
                }
            }

            result
        }
       
        pub fn load_shader(state : &mut ParseState, store : &ShaderStore) -> String {
            //println!("load_shader: {:?}", &state.path);
            
            let shader_code = store
                .get(&state.path)
                .unwrap_or_else(|| {
                    let available: Vec<_> = store.keys().collect();
                    panic!(
                        "WGSL shader not found in ShaderStore.\n\
                        Requested path: {:?}\n\
                        Available shaders:\n  {}",
                        state.path,
                        available
                            .iter()
                            .map(|p| format!("{:?}", p))
                            .collect::<Vec<_>>()
                            .join("\n  ")
                    )
                });

            if REMOVE_COMMENT{
                let shader_code = Regex::new(r"//.*\n", ).unwrap().replace_all(shader_code, "\n"); //remove coments
                load_shader_content(&shader_code, state, store)
            }
            else{
                load_shader_content(shader_code, state, store)
            }
        }

    }

    pub mod shader_shortener{
        use std::collections::{HashMap, HashSet};

        use crate::{REMOVE_SPACES, shader_loader::shader_tokeniser::match_whitespace_to_result};

        use super::{shader_tokeniser::{match_function_block, match_until_char, match_variable, Token, Tokenizer}, SHORTEN_GLOBAL_FUNCTIONS, SHORTEN_NORMAL_FUNCTIONS, SHORTEN_NORMAL_VARIABLES, SHORTEN_OVERRIDES};
        
        #[derive(Clone)]
        pub struct ShaderInfo{
            pub global_functions : Vec<(String, String)>, //original Function Name to New Function Name
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
                                let mut whitespaces = String::new();
                                match_whitespace_to_result(&mut tokens, &mut whitespaces);
                            
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
                                if !whitespaces.is_empty(){
                                    if REMOVE_SPACES{
                                            result.push(' ');
                                    }
                                    else{
                                        result.push_str(&whitespaces);
                                    }
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

impl Default for ShaderInfo {
    fn default() -> Self {
        Self::new()
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


