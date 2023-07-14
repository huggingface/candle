use candle::{DType, Device::Cpu, Result, Tensor};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

fn test_fn_impl() -> Result<String> {
    let t1 = Tensor::randn((3, 4), DType::F32, &Cpu, 0., 1.)?;
    let t2 = Tensor::randn((4, 2), DType::F32, &Cpu, 0., 1.)?;
    let t = t1.matmul(&t2)?;
    console_log!("matmul result: {t}");
    let res = format!("Hello Candle!\n\nt1:\n{t1}\n\nt2:\n{t2}\n\nt1@t2:\n{t}\n");
    Ok(res)
}

#[wasm_bindgen]
pub fn test_fn() -> std::result::Result<(), JsValue> {
    let result = match test_fn_impl() {
        Ok(v) => v,
        Err(err) => format!("error: {err:?}"),
    };
    let window = web_sys::window().expect("no global `window` exists");
    let document = window.document().expect("should have a document on window");
    let p_element = document.create_element("p")?;
    p_element.set_text_content(Some(&result));
    let body = document.body().expect("document should have a body");
    body.append_child(&p_element)?;
    Ok(())
}
