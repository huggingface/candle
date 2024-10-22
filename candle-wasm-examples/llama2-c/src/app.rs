use crate::console_log;
use crate::worker::{ModelData, Worker, WorkerInput, WorkerOutput};
use std::str::FromStr;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use yew::{html, Component, Context, Html};
use yew_agent::{Bridge, Bridged};

async fn fetch_url(url: &str) -> Result<Vec<u8>, JsValue> {
    use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};
    let window = web_sys::window().ok_or("window")?;
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    opts.set_cache(RequestCache::NoCache);
    let request = Request::new_with_str_and_init(url, &opts)?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into()?;
    let data = JsFuture::from(resp.blob()?).await?;
    let blob = web_sys::Blob::from(data);
    let array_buffer = JsFuture::from(blob.array_buffer()).await?;
    let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
    Ok(data)
}

pub enum Msg {
    Refresh,
    Run,
    UpdateStatus(String),
    SetModel(ModelData),
    WorkerIn(WorkerInput),
    WorkerOut(Result<WorkerOutput, String>),
}

pub struct CurrentDecode {
    start_time: Option<f64>,
}

pub struct App {
    status: String,
    loaded: bool,
    temperature: std::rc::Rc<std::cell::RefCell<f64>>,
    top_p: std::rc::Rc<std::cell::RefCell<f64>>,
    prompt: std::rc::Rc<std::cell::RefCell<String>>,
    generated: String,
    n_tokens: usize,
    current_decode: Option<CurrentDecode>,
    worker: Box<dyn Bridge<Worker>>,
}

async fn model_data_load() -> Result<ModelData, JsValue> {
    let tokenizer = fetch_url("tokenizer.json").await?;
    let model = fetch_url("model.bin").await?;
    console_log!("{}", model.len());
    Ok(ModelData { tokenizer, model })
}

fn performance_now() -> Option<f64> {
    let window = web_sys::window()?;
    let performance = window.performance()?;
    Some(performance.now() / 1000.)
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let status = "loading weights".to_string();
        let cb = {
            let link = ctx.link().clone();
            move |e| link.send_message(Self::Message::WorkerOut(e))
        };
        let worker = Worker::bridge(std::rc::Rc::new(cb));
        Self {
            status,
            n_tokens: 0,
            temperature: std::rc::Rc::new(std::cell::RefCell::new(0.)),
            top_p: std::rc::Rc::new(std::cell::RefCell::new(1.0)),
            prompt: std::rc::Rc::new(std::cell::RefCell::new("".to_string())),
            generated: String::new(),
            current_decode: None,
            worker,
            loaded: false,
        }
    }

    fn rendered(&mut self, ctx: &Context<Self>, first_render: bool) {
        if first_render {
            ctx.link().send_future(async {
                match model_data_load().await {
                    Err(err) => {
                        let status = format!("{err:?}");
                        Msg::UpdateStatus(status)
                    }
                    Ok(model_data) => Msg::SetModel(model_data),
                }
            });
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetModel(md) => {
                self.status = "weights loaded successfully!".to_string();
                self.loaded = true;
                console_log!("loaded weights");
                self.worker.send(WorkerInput::ModelData(md));
                true
            }
            Msg::Run => {
                if self.current_decode.is_some() {
                    self.status = "already generating some sample at the moment".to_string()
                } else {
                    let start_time = performance_now();
                    self.current_decode = Some(CurrentDecode { start_time });
                    self.status = "generating...".to_string();
                    self.n_tokens = 0;
                    self.generated.clear();
                    let temp = *self.temperature.borrow();
                    let top_p = *self.top_p.borrow();
                    let prompt = self.prompt.borrow().clone();
                    console_log!("temp: {}, top_p: {}, prompt: {}", temp, top_p, prompt);
                    ctx.link()
                        .send_message(Msg::WorkerIn(WorkerInput::Run(temp, top_p, prompt)))
                }
                true
            }
            Msg::WorkerOut(output) => {
                match output {
                    Ok(WorkerOutput::WeightsLoaded) => self.status = "weights loaded!".to_string(),
                    Ok(WorkerOutput::GenerationDone(Err(err))) => {
                        self.status = format!("error in worker process: {err}");
                        self.current_decode = None
                    }
                    Ok(WorkerOutput::GenerationDone(Ok(()))) => {
                        let dt = self.current_decode.as_ref().and_then(|current_decode| {
                            current_decode.start_time.and_then(|start_time| {
                                performance_now().map(|stop_time| stop_time - start_time)
                            })
                        });
                        self.status = match dt {
                            None => "generation succeeded!".to_string(),
                            Some(dt) => format!(
                                "generation succeeded in {:.2}s ({:.1} ms/token)",
                                dt,
                                dt * 1000.0 / (self.n_tokens as f64)
                            ),
                        };
                        self.current_decode = None
                    }
                    Ok(WorkerOutput::Generated(token)) => {
                        self.n_tokens += 1;
                        self.generated.push_str(&token)
                    }
                    Err(err) => {
                        self.status = format!("error in worker {err:?}");
                    }
                }
                true
            }
            Msg::WorkerIn(inp) => {
                self.worker.send(inp);
                true
            }
            Msg::UpdateStatus(status) => {
                self.status = status;
                true
            }
            Msg::Refresh => true,
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        use yew::TargetCast;
        let temperature = self.temperature.clone();
        let oninput_temperature = ctx.link().callback(move |e: yew::InputEvent| {
            let input: web_sys::HtmlInputElement = e.target_unchecked_into();
            if let Ok(temp) = f64::from_str(&input.value()) {
                *temperature.borrow_mut() = temp
            }
            Msg::Refresh
        });
        let top_p = self.top_p.clone();
        let oninput_top_p = ctx.link().callback(move |e: yew::InputEvent| {
            let input: web_sys::HtmlInputElement = e.target_unchecked_into();
            if let Ok(top_p_input) = f64::from_str(&input.value()) {
                *top_p.borrow_mut() = top_p_input
            }
            Msg::Refresh
        });
        let prompt = self.prompt.clone();
        let oninput_prompt = ctx.link().callback(move |e: yew::InputEvent| {
            let input: web_sys::HtmlInputElement = e.target_unchecked_into();
            *prompt.borrow_mut() = input.value();
            Msg::Refresh
        });
        html! {
            <div style="margin: 2%;">
                <div><p>{"Running "}
                <a href="https://github.com/karpathy/llama2.c" target="_blank">{"llama2.c"}</a>
                {" in the browser using rust/wasm with "}
                <a href="https://github.com/huggingface/candle" target="_blank">{"candle!"}</a>
                </p>
                <p>{"Once the weights have loaded, click on the run button to start generating content."}
                </p>
                </div>
                {"temperature  \u{00a0} "}
                <input type="range" min="0." max="1.2" step="0.1" value={self.temperature.borrow().to_string()} oninput={oninput_temperature} id="temp"/>
                {format!(" \u{00a0} {}", self.temperature.borrow())}
                <br/ >
                {"top_p  \u{00a0} "}
                <input type="range" min="0." max="1.0" step="0.05" value={self.top_p.borrow().to_string()} oninput={oninput_top_p} id="top_p"/>
                {format!(" \u{00a0} {}", self.top_p.borrow())}
                <br/ >
                {"prompt: "}<input type="text" value={self.prompt.borrow().to_string()} oninput={oninput_prompt} id="prompt"/>
                <br/ >
                {
                    if self.loaded{
                        html!(<button class="button" onclick={ctx.link().callback(move |_| Msg::Run)}> { "run" }</button>)
                    }else{
                        html! { <progress id="progress-bar" aria-label="Loading weights..."></progress> }
                    }
                }
                <br/ >
                <h3>
                  {&self.status}
                </h3>
                {
                    if self.current_decode.is_some() {
                        html! { <progress id="progress-bar" aria-label="generating…"></progress> }
                    } else {
                        html! {}
                    }
                }
                <blockquote>
                <p> { self.generated.chars().map(|c|
                    if c == '\r' || c == '\n' {
                        html! { <br/> }
                    } else {
                        html! { {c} }
                    }).collect::<Html>()
                } </p>
                </blockquote>
            </div>
        }
    }
}
