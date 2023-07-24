use crate::console_log;
use crate::worker::{ModelData, Worker, WorkerInput, WorkerOutput};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use yew::{html, Component, Context, Html};
use yew_agent::{Bridge, Bridged};

async fn fetch_url(url: &str) -> Result<Vec<u8>, JsValue> {
    use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};
    let window = web_sys::window().ok_or("window")?;
    let mut opts = RequestInit::new();
    let opts = opts
        .method("GET")
        .mode(RequestMode::Cors)
        .cache(RequestCache::NoCache);

    let request = Request::new_with_str_and_init(url, opts)?;

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
    Run,
    UpdateStatus(String),
    SetModel(ModelData),
    WorkerInMsg(WorkerInput),
    WorkerOutMsg(Result<WorkerOutput, String>),
}

pub struct CurrentDecode {
    start_time: Option<f64>,
}

pub struct App {
    status: String,
    generated: String,
    current_decode: Option<CurrentDecode>,
    worker: Box<dyn Bridge<Worker>>,
}

async fn model_data_load() -> Result<ModelData, JsValue> {
    let tokenizer = fetch_url("tokenizer.bin").await?;
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
            move |e| link.send_message(Self::Message::WorkerOutMsg(e))
        };
        let worker = Worker::bridge(std::rc::Rc::new(cb));
        Self {
            status,
            generated: String::new(),
            current_decode: None,
            worker,
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
                self.status = "weights loaded succesfully!".to_string();
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
                    self.generated.clear();
                    ctx.link().send_message(Msg::WorkerInMsg(WorkerInput::Run))
                }
                true
            }
            Msg::WorkerOutMsg(output) => {
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
                            Some(dt) => format!("generation succeeded in {:.2}s", dt),
                        };
                        self.current_decode = None
                    }
                    Ok(WorkerOutput::Generated(token)) => self.generated.push_str(&token),
                    Err(err) => {
                        self.status = format!("error in worker {err:?}");
                    }
                }
                true
            }
            Msg::WorkerInMsg(inp) => {
                self.worker.send(inp);
                true
            }
            Msg::UpdateStatus(status) => {
                self.status = status;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div>
                <div>{"Running "}
                <a href="https://github.com/karpathy/llama2.c">{"llama2.c"}</a>
                {" in the browser using rust/wasm with "}
                <a href="https://github.com/LaurentMazare/candle">{"candle!"}</a>
                </div>
                <button class="button" onclick={ctx.link().callback(move |_| Msg::Run)}> { "run" }</button>
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
