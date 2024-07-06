use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

use wasm_bindgen::JsCast;
use crate::generic_error::GenericResult;
use urlencoding::encode;


pub async fn download_file(url : &str) -> GenericResult<web_sys::Blob> {
    let encoded = encode(url);

    let new_url = format!("http://localhost:3000/fetch-resource/?url={}", encoded);

    log::info!("download file: {new_url}");
    
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);

    log::info!("Method: {opts:?}");

    let request = Request::new_with_str_and_init(&new_url, &opts)?;

    log::info!("request: {request:?}");

    //request .headers().set("Accept", "")?;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    log::info!("resp_value: {resp_value:?}");

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();



    log::info!("resp: {resp:?}");
    
    let status = resp.status();

    log::info!("status: {status:?}");

    let status_text = resp.status_text();


    log::info!("status_text: {status_text:?}");
    log::info!("trying to create blob");
    
    let blob : web_sys::Blob = JsFuture::from(resp.blob()?).await?.into();

    log::info!("blob created");

    return Ok(blob);

    // log::info!("trying to create array buffer");

    // // Convert this other `Promise` into a rust `Future`.
    // let buffer = JsFuture::from(resp.array_buffer()?).await?;
    // log::info!("buffer created");
   
    // let array : js_sys::ArrayBuffer = buffer.into(); 
    
    // let uint8_array = js_sys::Uint8Array::new(&array);

    // //log::info!("buffer: {buffer:?}");

    // log::info!("uint8_array: {uint8_array:?}");

    // return Ok(uint8_array.to_vec())
}
