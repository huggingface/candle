use std::path::Path;

use anyhow::Result;
use crate::generic_error::GenericResult;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemRemoveOptions};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen (js_name = FileSystemDirectoryHandle, extends=::web_sys::FileSystemDirectoryHandle, typescript_type = "FileSystemDirectoryHandle")]
    #[derive(Debug, Clone, PartialEq)]
    #[doc = "The `FileSystemDirectoryHandle` class."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemDirectoryHandle)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `FileSystemDirectoryHandle`*"]
    pub type FileSystemDirectoryHandleCustom;
    # [wasm_bindgen (method , structural , js_class = "FileSystemDirectoryHandle" , js_name = entries)]
    pub fn entries(this: &FileSystemDirectoryHandleCustom) -> ::js_sys::AsyncIterator;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen (js_name = ReadableStream , extends=::web_sys::ReadableStream, typescript_type = "ReadableStream")]
    #[derive(Debug, Clone, PartialEq)]
    #[doc = "The `ReadableStream` class."]
    #[doc = ""]
    pub type ReadableStreamCustom;
    # [wasm_bindgen (method , structural , js_class = "ReadableStream" , js_name = values)]
    pub fn values(this: &ReadableStreamCustom) -> ::js_sys::AsyncIterator;

}

//opfs API:
pub enum FileSystemDirectoryEntries{
    Directory(web_sys::FileSystemDirectoryHandle),
    File(web_sys::FileSystemFileHandle),
}

pub async fn get_root() -> GenericResult<web_sys::FileSystemDirectoryHandle> {
    let storage = js_sys::Reflect::get(
        &web_sys::window().ok_or("no global `window` exists")?.navigator(),
        &JsValue::from_str("storage"),
    )?;
    let get_directory = js_sys::Reflect::get(&storage, &JsValue::from_str("getDirectory"))?
        .dyn_into::<js_sys::Function>()?;
    let promise = get_directory.call0(&storage)?.dyn_into::<js_sys::Promise>()?;
    let result = JsFuture::from(promise).await?;
    result.dyn_into::<web_sys::FileSystemDirectoryHandle>().map_err(|_| "Failed to convert result".into())
}

pub async fn get_dir_entries(dir : FileSystemDirectoryHandleCustom) -> GenericResult<Vec<(String, FileSystemDirectoryEntries)>> {
    let iter : js_sys::AsyncIterator = dir.entries();
    
    let mut result : Vec<(String, FileSystemDirectoryEntries)> = vec![];
    loop {
        let next : js_sys::IteratorNext = JsFuture::from(iter.next()?).await?.into();
        if next.done(){
            break;
        }
        let value = next.value();

        let value : js_sys::Array = value.into();
        let name : js_sys::JsString = value.get(0).into();

        let value = value.get(1);

        if value.is_instance_of::<web_sys::FileSystemDirectoryHandle>(){
            let directory : web_sys::FileSystemDirectoryHandle = value.into();
            result.push((name.into(), FileSystemDirectoryEntries::Directory(directory)));
        }
        else if value.is_instance_of::<web_sys::FileSystemFileHandle>(){
            let file : web_sys::FileSystemFileHandle = value.into();
            result.push((name.into(), FileSystemDirectoryEntries::File(file)));
        }
    }   
    Ok(result)
}



pub async fn clear_directory(directory : web_sys::FileSystemDirectoryHandle, recursive : bool) -> GenericResult<()>
{
    let dir : JsValue = directory.clone().into();
    log::info!("clear directory");

    let entries = get_dir_entries(dir.into()).await?;
    for (name, _) in entries{
        log::info!("remove entry: {name}");
        let fsro = FileSystemRemoveOptions::new();
        fsro.set_recursive(recursive);
        JsFuture::from(directory.remove_entry_with_options(&name, &fsro)).await?;
    }
    Ok(())
}

pub async fn clear_all(recursive : bool) -> GenericResult<()>{
    log::info!("clear all");
    clear_directory(get_root().await?, recursive).await
}


pub async fn exist_file<P>(file_name : P)  -> bool
where
    P: AsRef<Path>
{
   open_file(file_name).await.is_ok()
}


pub async fn create_file<P>(file_name : P)  -> GenericResult<web_sys::FileSystemFileHandle>
where
    P: AsRef<Path>
{
    log::info!("create file: {:?}", file_name.as_ref());
    let mut root = get_root().await?;
    
    let path = file_name.as_ref();
    let components : Vec<_> = path.components().collect();
    for (index, p) in components.iter().enumerate(){
        if let std::path::Component::Normal(p) = p {
            let name = p.to_str().unwrap();
            let is_file = index == components.len() - 1;
            if !is_file{
                let fsgdo = FileSystemGetDirectoryOptions::new();
                fsgdo.set_create(true);
                root = JsFuture::from(root.get_directory_handle_with_options(name, &fsgdo)).await?.into();
            }
            else{
                let fsgfo = FileSystemGetFileOptions::new();
                fsgfo.set_create(true);
                let file_handle : web_sys::FileSystemFileHandle = JsFuture::from(root.get_file_handle_with_options(name, &fsgfo)).await?.into();
                return Ok(file_handle);
            }
        }
      
    }

    Err("File Creating File".into())
}


pub async fn open_file<P>(file_name : P)  -> GenericResult<web_sys::FileSystemFileHandle>
where
    P: AsRef<Path>
{
    let mut root = get_root().await?;
    let path = file_name.as_ref();
    let components : Vec<_> = path.components().collect();
    for (index, p) in components.iter().enumerate(){
        if let std::path::Component::Normal(p) = p {
            let name = p.to_str().unwrap();
            let is_file = index == components.len() - 1;
            if !is_file{
                root = JsFuture::from(root.get_directory_handle(name)).await?.into();
            }
            else{
                let file_handle : web_sys::FileSystemFileHandle = JsFuture::from(root.get_file_handle(name)).await?.into();
                return Ok(file_handle);
            }
        }
    }
    Err("File not Found".into())
}


pub async fn open_dir<P>(file_name : P)  -> GenericResult<web_sys::FileSystemDirectoryHandle>
where
    P: AsRef<Path>
{
    let mut root = get_root().await?;
    let path = file_name.as_ref();
    let components : Vec<_> = path.components().collect();
    for p in components.iter(){
        if let std::path::Component::Normal(p) = p {
            let name = p.to_str().unwrap();
            root = JsFuture::from(root.get_directory_handle(name)).await?.into();
        }
    }
    Ok(root)
}

pub async fn get_file(file_handle : web_sys::FileSystemFileHandle) -> GenericResult<web_sys::File> {
    let file : web_sys::File =  JsFuture::from(file_handle.get_file()).await?.into();
    Ok(file)
}

pub async fn read_file<P>(file_name : P) -> GenericResult<Vec<u8>>
where
    P: AsRef<Path>
{
    let mut result = vec![];

    match open_file(&file_name).await{
        Ok(file_handle) =>{
            let file : web_sys::File =  JsFuture::from(file_handle.get_file()).await?.into();
            let stream : JsValue = file.stream().into();
            let stream : ReadableStreamCustom = stream.into();
            let iter : js_sys::AsyncIterator = stream.values();
            loop {
                let next : js_sys::IteratorNext = JsFuture::from(iter.next()?).await?.into();
                if next.done(){
                    break;
                }
                let value = next.value();
                let value : js_sys::Uint8Array = value.into();
                let mut chunk = value.to_vec();
                result.append(&mut chunk);
            }
            Ok(result)
        },
        Err(e) =>  Err(e),
    }
}


pub struct ReadableRustStream{
    total_length : u64,
    data : js_sys::AsyncIterator,
    chunk : Vec<u8>, //current chunk,
    chunk_index : usize
}

impl ReadableRustStream {
    pub fn new(stream:  ReadableStreamCustom,total_length : u64) -> Self {
        let iter : js_sys::AsyncIterator = stream.values();
        Self {data : iter, chunk : vec![], chunk_index : 0, total_length}
    }

    pub fn len(&self) -> usize{
        self.total_length as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub async fn read_bytes(&mut self, size : usize) -> GenericResult<Vec<u8>>{
        let mut result = vec![];


        while result.len() < size{
            let chunk_copy = (size - result.len()).min(self.chunk.len() - self.chunk_index);
            result.extend_from_slice(&self.chunk[self.chunk_index..(self.chunk_index+chunk_copy)]);
            self.chunk_index += chunk_copy;

            if self.chunk_index >= self.chunk.len(){
                let next : js_sys::IteratorNext = JsFuture::from(self.data.next()?).await?.into();
                if next.done(){
                    break;
                }
                let value = next.value();
                let value : js_sys::Uint8Array = value.into();
                let chunk: Vec<u8> = value.to_vec();
                self.chunk = chunk;
                self.chunk_index = 0;
            }
        }
        
        Ok(result)
    }
}


#[derive(Debug)]
pub struct Blob{
    blob : web_sys::Blob
}

impl Blob {
    pub fn new<T : Into<web_sys::Blob>>(blob: T) -> Self {
        Self { blob : blob.into()}
    }

    pub fn len(&self) -> usize{
        self.blob.size() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub async fn get_bytes(&self, start : usize, length : usize) -> GenericResult<Vec<u8>>{
        let slice = self.blob.slice_with_f64_and_f64(start as f64, (start + length) as f64)?;
        let data: JsValue = JsFuture::from(slice.array_buffer()).await?;
        let uint8_array = js_sys::Uint8Array::new(&data);
        let data = uint8_array.to_vec();
        if data.len() != length {
            panic!("Get Bytes could not load {length} bytes, only got: {}", data.len());
        }

        Ok(data)
    }

    pub fn get_stream(&self)-> GenericResult<ReadableRustStream>
    {   
        let stream : JsValue = self.blob.stream().into();
        let stream : ReadableStreamCustom = stream.into();
        Ok(ReadableRustStream::new(stream, self.len() as u64))
    }
}


pub async fn get_rust_blob<P>(file_name : P)-> GenericResult<Blob>
where
    P: AsRef<Path>{
        match open_file(&file_name).await{
            Ok(file_handle) =>{
                let file : web_sys::File =  JsFuture::from(file_handle.get_file()).await?.into();
                Ok(Blob::new(file))
            },
            Err(e) =>  Err(e),
        }
}



pub async fn write_file(file_handle : web_sys::FileSystemFileHandle, data : &[u8]) -> Result<(), JsValue>{
    let writable : web_sys::FileSystemWritableFileStream = JsFuture::from(file_handle.create_writable()).await?.into();
    JsFuture::from(writable.write_with_u8_array(data)?).await?;
    JsFuture::from(writable.close()).await?;
    Ok(())
}


pub async fn write_file_blob(file_handle : web_sys::FileSystemFileHandle, data : web_sys::Blob) -> Result<(), JsValue>{
    let writable : web_sys::FileSystemWritableFileStream = JsFuture::from(file_handle.create_writable()).await?.into();
    JsFuture::from(writable.write_with_blob(&data)?).await?;
    JsFuture::from(writable.close()).await?;
    Ok(())
}

