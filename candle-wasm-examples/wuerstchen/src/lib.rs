// use std::path::Path;

// use anyhow::Result;
// use generic_error::GenericResult;
// use wasm_bindgen::{convert::IntoWasmAbi, prelude::*};
// use wasm_bindgen_futures::JsFuture;
// use web_sys::{FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemRemoveOptions, Request, RequestInit, RequestMode, Response};

// pub mod generic_error;
// pub mod hfhub_helper;
// pub mod hfhub_helper_api;

// #[wasm_bindgen]
// extern "C" {
//     #[wasm_bindgen(js_namespace = ["navigator", "storage"])]
//     async fn getDirectory() -> JsValue;
// }



// #[wasm_bindgen]
// extern "C" {
//     #[wasm_bindgen (js_name = FileSystemDirectoryHandle , typescript_type = "FileSystemDirectoryHandle")]
//     #[derive(Debug, Clone, PartialEq)]
//     #[doc = "The `FileSystemDirectoryHandle` class."]
//     #[doc = ""]
//     #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemDirectoryHandle)"]
//     #[doc = ""]
//     #[doc = "*This API requires the following crate features to be activated: `FileSystemDirectoryHandle`*"]
//     pub type FileSystemDirectoryHandleCustom;
//     # [wasm_bindgen (method , structural , js_class = "FileSystemDirectoryHandle" , js_name = entries)]
//     pub fn entries(this: &FileSystemDirectoryHandleCustom) -> ::js_sys::AsyncIterator;

// }

// //opfs API:

// pub enum FileSystemDirectoryEntries{
//     Directory(web_sys::FileSystemDirectoryHandle),
//     File(web_sys::FileSystemFileHandle),
// }

// pub async fn get_root() -> web_sys::FileSystemDirectoryHandle{
//     return getDirectory().await.into();
// }

// pub async fn get_dir_entries() -> GenericResult<Vec<(String, FileSystemDirectoryEntries)>> {
//     let root : FileSystemDirectoryHandleCustom = getDirectory().await.into();
//     let iter : js_sys::AsyncIterator = root.entries();
    
//     let mut result : Vec<(String, FileSystemDirectoryEntries)> = vec![];
//     loop {
//         let next : js_sys::IteratorNext = JsFuture::from(iter.next()?).await?.into();
//         if next.done(){
//             break;
//         }
//         let value = next.value();

//         let value : js_sys::Array = value.into();
//         let name : js_sys::JsString = value.get(0).into();

//         let value = value.get(1);

//         if value.is_instance_of::<web_sys::FileSystemDirectoryHandle>(){
//             let directory : web_sys::FileSystemDirectoryHandle = value.into();
//             result.push((name.into(), FileSystemDirectoryEntries::Directory(directory)));
//         }
//         else if value.is_instance_of::<web_sys::FileSystemFileHandle>(){
//             let file : web_sys::FileSystemFileHandle = value.into();
//             result.push((name.into(), FileSystemDirectoryEntries::File(file)));
//         }
//     }   
//     return Ok(result);
// }



// pub async fn clear_directory(directory : web_sys::FileSystemDirectoryHandle, recursive : bool) -> GenericResult<()>
// {
//     log::info!("clear directory");
//     let entries = get_dir_entries().await?;
//     for (name, _) in entries{
//         JsFuture::from(directory.remove_entry_with_options(&name, FileSystemRemoveOptions::new().recursive(recursive))).await?;
//     }
//     Ok(())
// }

// pub async fn clear_all(recursive : bool) -> GenericResult<()>{
//     log::info!("clear all");
//     clear_directory(get_root().await, recursive).await
// }


// pub async fn exist_file<P>(file_name : P)  -> bool
// where
//     P: AsRef<Path>
// {
//    open_file(file_name).await.is_ok()
// }


// pub async fn create_file<P>(file_name : P)  -> GenericResult<web_sys::FileSystemFileHandle>
// where
//     P: AsRef<Path>
// {
//     log::info!("create file: {:?}", file_name.as_ref());
//     let mut root = get_root().await;
    
//     let path = file_name.as_ref();
//     let components : Vec<_> = path.components().collect();
//     for (index, p) in components.iter().enumerate(){
//         match p{
//             std::path::Component::Normal(p) => {
//                 let name = p.to_str().unwrap();
//                 let is_file = index == components.len() - 1;
//                 if !is_file{
//                     log::info!("create dir: {:?}", name);
//                     root = JsFuture::from(root.get_directory_handle_with_options(name, FileSystemGetDirectoryOptions::new().create(true))).await?.into();
//                 }
//                 else{
//                     log::info!("create file: {:?}", name);
//                     let file_handle : web_sys::FileSystemFileHandle = JsFuture::from(root.get_file_handle_with_options(name, FileSystemGetFileOptions::new().create(true))).await?.into();
//                     log::info!("file created: {:?}", name);
//                     return Ok(file_handle);
//                 }
//             },
//             _ => {},
//         }
      
//     }

//     return Err("File Creating File".into());
// }


// pub async fn open_file<P>(file_name : P)  -> GenericResult<web_sys::FileSystemFileHandle>
// where
//     P: AsRef<Path>
// {
//     log::info!("open file: {:?}", file_name.as_ref());
//     let mut root = get_root().await;
//     let path = file_name.as_ref();
//     let components : Vec<_> = path.components().collect();
//     for (index, p) in components.iter().enumerate(){
//         match p{
//             std::path::Component::Normal(p) => {
//                 let name = p.to_str().unwrap();
//                 let is_file = index == components.len() - 1;
//                 if !is_file{
//                     log::info!("open dir: {:?}", name);
//                     root = JsFuture::from(root.get_directory_handle(name)).await?.into();
//                 }
//                 else{
//                     log::info!("open file: {:?}", name);
//                     let file_handle : web_sys::FileSystemFileHandle = JsFuture::from(root.get_file_handle(name)).await?.into();
//                     return Ok(file_handle);
//                 }
//             },
//             _ => {},
//         }
//     }
//     return Err("File not Found".into());
// }


// pub async fn open_dir<P>(file_name : P)  -> GenericResult<web_sys::FileSystemDirectoryHandle>
// where
//     P: AsRef<Path>
// {
//     log::info!("open file: {:?}", file_name.as_ref());
//     let mut root = get_root().await;
//     let path = file_name.as_ref();
//     let components : Vec<_> = path.components().collect();
//     for p in components.iter(){
//         match p{
//             std::path::Component::Normal(p) => {
//                 let name = p.to_str().unwrap();
                
//                 log::info!("open dir: {:?}", name);
//                 root = JsFuture::from(root.get_directory_handle(name)).await?.into();
//             },
//             _ => {},
//         }
//     }
//     return Ok(root);
// }


// pub async fn read_file<P>(file_name : P) -> GenericResult<Vec<u8>>
// where
//     P: AsRef<Path>
// {
//     log::info!("Read File \"{:?}\".", file_name.as_ref());
//     match open_file(&file_name).await{
//         Ok(file_handle) =>{
            
//             log::info!("File \"{:?}\" Found", file_name.as_ref());
//             let file : web_sys::File =  JsFuture::from(file_handle.get_file()).await?.into();
//             log::info!("File \"{:?}\" Get File", file_name.as_ref());
//             let buffer = JsFuture::from(file.array_buffer()).await?;  
//             log::info!("File \"{:?}\" Get Buffer", file_name.as_ref());
//             let uint8_array = js_sys::Uint8Array::new(&buffer);
//             log::info!("File \"{:?}\" Get Uint8Array", file_name.as_ref());
//             return Ok(uint8_array.to_vec())
//         },
//         Err(e) =>  Err(e),
//     }
// }

// pub async fn write_file(file_handle : web_sys::FileSystemFileHandle, data : &[u8]) -> Result<(), JsValue>{
//     log::info!("Write File, data:{:?}", data);
//     let writable : web_sys::FileSystemWritableFileStream = JsFuture::from(file_handle.create_writable()).await?.into();
//     JsFuture::from(writable.write_with_u8_array(data)?).await?;
//     log::info!("wrote, closing:");
//     JsFuture::from(writable.close()).await?;
//     Ok(())
// }


// pub async fn read_file_to_string<P>(file_name : P) -> GenericResult<String>
// where
//     P: AsRef<Path>
// {
//     return read_file(file_name).await.map(|f| return String::from_utf8(f).unwrap());
// }





// //use proxy server becasue of cors
// pub async fn download_file(url : &str) -> GenericResult<Vec<u8>> {
    
//     log::info!("download file: {url}");
    
//     let mut opts = RequestInit::new();
//     opts.method("GET");
//     opts.mode(RequestMode::Cors);
//     opts.credentials(web_sys::RequestCredentials::Omit);


//     log::info!("Method: {opts:?}");

//     let request = Request::new_with_str_and_init(url, &opts)?;

//     log::info!("request: {request:?}");

//     request.headers().set("Accept", "*/*")?;
//     //request.headers().set("Referrer-Policy", "unsafe-url")?;

//     let window = web_sys::window().unwrap();
//     let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

//     log::info!("resp_value: {resp_value:?}");

//     // `resp_value` is a `Response` object.
//     assert!(resp_value.is_instance_of::<Response>());
//     let resp: Response = resp_value.dyn_into().unwrap();



//     log::info!("resp: {resp:?}");
    
//     let status = resp.status();

//     log::info!("status: {status:?}");

//     let status_text = resp.status_text();


//     log::info!("status_text: {status_text:?}");
    
//     // Convert this other `Promise` into a rust `Future`.
//     let buffer = JsFuture::from(resp.array_buffer()?).await?;

//     log::info!("buffer: {buffer:?}");

    
//     let uint8_array = js_sys::Uint8Array::new(&buffer);

//     log::info!("uint8_array: {uint8_array:?}");

//     return Ok(uint8_array.to_vec())
// }

// pub async fn load_file(url : &str) -> GenericResult<Vec<u8>> {
//     log::info!("load file: {url}");
//     let root :  web_sys::FileSystemDirectoryHandle = getDirectory().await.into();
//     let file_name = url.split('/').last().unwrap();
//     log::info!("search filename: {file_name}");
//     match  JsFuture::from(root.get_file_handle(file_name)).await{
//         Ok(file_handle) => 
//         {
//             log::info!("found filename, trying to load from cache");
//             let file_handle : web_sys::FileSystemFileHandle = file_handle.into();
//             let file : web_sys::File =  JsFuture::from(file_handle.get_file()).await?.into();
//             let buffer = JsFuture::from(file.array_buffer()).await?;
//             let uint8_array = js_sys::Uint8Array::new(&buffer);
//             return Ok(uint8_array.to_vec());
//         }
//         ,
//         Err(_) =>   
//         {   
//             log::info!("download file{url}");
//             let data = download_file(url).await?;
//             log::info!("downloaded file{url} contet:{data:?}, trying to store at {file_name}!");
//             let file_handle : web_sys::FileSystemFileHandle = JsFuture::from(root.get_file_handle_with_options(file_name, FileSystemGetFileOptions::new().create(true))).await?.into();
//             let writable : web_sys::FileSystemWritableFileStream = JsFuture::from(file_handle.create_writable()).await?.into();
//             JsFuture::from(writable.write_with_u8_array(&data)?).await?;
//             JsFuture::from(writable.close()).await?;
//             log::info!("stored file{url} at {file_name}");
//             return Ok(data);
//         },
//     }
// }