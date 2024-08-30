use crate::console_log;
use crate::worker::{ModelData, Segment, Worker, WorkerInput, WorkerOutput};
use js_sys::Date;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use yew::{html, Component, Context, Html};
use yew_agent::{Bridge, Bridged};

const SAMPLE_NAMES: [&str; 6] = [
    "audios/samples_jfk.wav",
    "audios/samples_a13.wav",
    "audios/samples_gb0.wav",
    "audios/samples_gb1.wav",
    "audios/samples_hp0.wav",
    "audios/samples_mm0.wav",
];

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
    Run(usize),
    UpdateStatus(String),
    SetDecoder(ModelData),
    WorkerIn(WorkerInput),
    WorkerOut(Result<WorkerOutput, String>),
}

pub struct CurrentDecode {
    start_time: Option<f64>,
}

pub struct App {
    status: String,
    loaded: bool,
    segments: Vec<Segment>,
    current_decode: Option<CurrentDecode>,
    worker: Box<dyn Bridge<Worker>>,
}

async fn model_data_load() -> Result<ModelData, JsValue> {
    let quantized = false;
    let is_multilingual = false;

    let (tokenizer, mel_filters, weights, config) = if quantized {
        console_log!("loading quantized weights");
        let tokenizer = fetch_url("quantized/tokenizer-tiny-en.json").await?;
        let mel_filters = fetch_url("mel_filters.safetensors").await?;
        let weights = fetch_url("quantized/model-tiny-en-q80.gguf").await?;
        let config = fetch_url("quantized/config-tiny-en.json").await?;
        (tokenizer, mel_filters, weights, config)
    } else {
        console_log!("loading float weights");
        if is_multilingual {
            let mel_filters = fetch_url("mel_filters.safetensors").await?;
            let tokenizer = fetch_url("whisper-tiny/tokenizer.json").await?;
            let weights = fetch_url("whisper-tiny/model.safetensors").await?;
            let config = fetch_url("whisper-tiny/config.json").await?;
            (tokenizer, mel_filters, weights, config)
        } else {
            let mel_filters = fetch_url("mel_filters.safetensors").await?;
            let tokenizer = fetch_url("whisper-tiny.en/tokenizer.json").await?;
            let weights = fetch_url("whisper-tiny.en/model.safetensors").await?;
            let config = fetch_url("whisper-tiny.en/config.json").await?;
            (tokenizer, mel_filters, weights, config)
        }
    };

    let timestamps = true;
    let _task = Some("transcribe".to_string());
    console_log!("{}", weights.len());
    Ok(ModelData {
        tokenizer,
        mel_filters,
        weights,
        config,
        quantized,
        timestamps,
        task: None,
        is_multilingual,
        language: None,
    })
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
            segments: vec![],
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
                    Ok(model_data) => Msg::SetDecoder(model_data),
                }
            });
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetDecoder(md) => {
                self.status = "weights loaded successfully!".to_string();
                self.loaded = true;
                console_log!("loaded weights");
                self.worker.send(WorkerInput::ModelData(md));
                true
            }
            Msg::Run(sample_index) => {
                let sample = SAMPLE_NAMES[sample_index];
                if self.current_decode.is_some() {
                    self.status = "already decoding some sample at the moment".to_string()
                } else {
                    let start_time = performance_now();
                    self.current_decode = Some(CurrentDecode { start_time });
                    self.status = format!("decoding {sample}");
                    self.segments.clear();
                    ctx.link().send_future(async move {
                        match fetch_url(sample).await {
                            Err(err) => {
                                let output = Err(format!("decoding error: {err:?}"));
                                // Mimic a worker output to so as to release current_decode
                                Msg::WorkerOut(output)
                            }
                            Ok(wav_bytes) => Msg::WorkerIn(WorkerInput::DecodeTask { wav_bytes }),
                        }
                    })
                }
                //
                true
            }
            Msg::WorkerOut(output) => {
                let dt = self.current_decode.as_ref().and_then(|current_decode| {
                    current_decode.start_time.and_then(|start_time| {
                        performance_now().map(|stop_time| stop_time - start_time)
                    })
                });
                self.current_decode = None;
                match output {
                    Ok(WorkerOutput::WeightsLoaded) => self.status = "weights loaded!".to_string(),
                    Ok(WorkerOutput::Decoded(segments)) => {
                        self.status = match dt {
                            None => "decoding succeeded!".to_string(),
                            Some(dt) => format!("decoding succeeded in {:.2}s", dt),
                        };
                        self.segments = segments;
                    }
                    Err(err) => {
                        self.status = format!("decoding error {err:?}");
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
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div>
                <table>
                <thead>
                <tr>
                  <th>{"Sample"}</th>
                  <th></th>
                  <th></th>
                </tr>
                </thead>
                <tbody>
                {
                    SAMPLE_NAMES.iter().enumerate().map(|(i, name)| { html! {
                <tr>
                  <th>{name}</th>
                  <th><audio controls=true src={format!("./{name}")}></audio></th>
                  { if self.loaded {
                      html!(<th><button class="button" onclick={ctx.link().callback(move |_| Msg::Run(i))}> { "run" }</button></th>)
                       }else{html!()}
                  }
                </tr>
                    }
                    }).collect::<Html>()
                }
                </tbody>
                </table>
                <h2>
                  {&self.status}
                </h2>
                {
                    if !self.loaded{
                        html! { <progress id="progress-bar" aria-label="loading weights…"></progress> }
                    } else if self.current_decode.is_some() {
                        html! { <progress id="progress-bar" aria-label="decoding…"></progress> }
                    } else { html!{
                <blockquote>
                <p>
                  {
                      self.segments.iter().map(|segment| { html! {
                          <>
                          <i>
                          {
                              format!("{:.2}s-{:.2}s: (avg-logprob: {:.4}, no-speech-prob: {:.4})",
                                  segment.start,
                                  segment.start + segment.duration,
                                  segment.dr.avg_logprob,
                                  segment.dr.no_speech_prob,
                              )
                          }
                          </i>
                          <br/ >
                          {&segment.dr.text}
                          <br/ >
                          </>
                      } }).collect::<Html>()
                  }
                </p>
                </blockquote>
                }
                }
                }

                // Display the current date and time the page was rendered
                <p class="footer">
                    { "Rendered: " }
                    { String::from(Date::new_0().to_string()) }
                </p>
            </div>
        }
    }
}
