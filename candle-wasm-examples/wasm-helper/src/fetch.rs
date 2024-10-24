use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

use wasm_bindgen::JsCast;
use crate::generic_error::GenericResult;

pub async fn download_file(url : &str) -> GenericResult<web_sys::Blob> {
    log::info!("download file: {url}");
    
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    log::info!("Method: {opts:?}");

    let request = Request::new_with_str_and_init(url, &opts)?;

    log::info!("request: {request:?}");

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

    Ok(blob)
}
