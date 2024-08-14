use crate::console_log;
use crate::worker::{ModelData, RunData, Worker, WorkerInput, WorkerOutput};
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
    generated: String,
    current_decode: Option<CurrentDecode>,
    worker: Box<dyn Bridge<Worker>>,
}

async fn model_data_load() -> Result<ModelData, JsValue> {
    let weights = fetch_url("yolov8s.safetensors").await?;
    let model_size = "s".to_string();
    console_log!("loaded weights {}", weights.len());
    Ok(ModelData {
        weights,
        model_size,
    })
}

fn performance_now() -> Option<f64> {
    let window = web_sys::window()?;
    let performance = window.performance()?;
    Some(performance.now() / 1000.)
}

fn draw_bboxes(bboxes: Vec<Vec<crate::model::Bbox>>) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = match document.get_element_by_id("canvas") {
        Some(canvas) => canvas,
        None => return Err("no canvas".into()),
    };
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let context = canvas
        .get_context("2d")?
        .ok_or("no 2d")?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

    let image_html_element = document.get_element_by_id("bike-img");
    let image_html_element = match image_html_element {
        Some(data) => data,
        None => return Err("no bike-img".into()),
    };
    let image_html_element = image_html_element.dyn_into::<web_sys::HtmlImageElement>()?;
    canvas.set_width(image_html_element.natural_width());
    canvas.set_height(image_html_element.natural_height());
    context.draw_image_with_html_image_element(&image_html_element, 0., 0.)?;
    context.set_stroke_style(&JsValue::from("#0dff9a"));
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            let name = crate::coco_classes::NAMES[class_index];
            context.stroke_rect(
                b.xmin as f64,
                b.ymin as f64,
                (b.xmax - b.xmin) as f64,
                (b.ymax - b.ymin) as f64,
            );
            if let Ok(metrics) = context.measure_text(name) {
                let width = metrics.width();
                context.set_fill_style(&"#3c8566".into());
                context.fill_rect(b.xmin as f64 - 2., b.ymin as f64 - 12., width + 4., 14.);
                context.set_fill_style(&"#e3fff3".into());
                context.fill_text(name, b.xmin as f64, b.ymin as f64 - 2.)?
            }
        }
    }
    Ok(())
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
                    self.status = "already processing some image at the moment".to_string()
                } else {
                    let start_time = performance_now();
                    self.current_decode = Some(CurrentDecode { start_time });
                    self.status = "processing...".to_string();
                    self.generated.clear();
                    ctx.link().send_future(async {
                        match fetch_url("bike.jpeg").await {
                            Err(err) => {
                                let status = format!("{err:?}");
                                Msg::UpdateStatus(status)
                            }
                            Ok(image_data) => Msg::WorkerIn(WorkerInput::RunData(RunData {
                                image_data,
                                conf_threshold: 0.5,
                                iou_threshold: 0.5,
                            })),
                        }
                    });
                }
                true
            }
            Msg::WorkerOut(output) => {
                match output {
                    Ok(WorkerOutput::WeightsLoaded) => self.status = "weights loaded!".to_string(),
                    Ok(WorkerOutput::ProcessingDone(Err(err))) => {
                        self.status = format!("error in worker process: {err}");
                        self.current_decode = None
                    }
                    Ok(WorkerOutput::ProcessingDone(Ok(bboxes))) => {
                        let mut content = Vec::new();
                        for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
                            for b in bboxes_for_class.iter() {
                                content.push(format!(
                                    "bbox {}: xs {:.0}-{:.0}  ys {:.0}-{:.0}",
                                    crate::coco_classes::NAMES[class_index],
                                    b.xmin,
                                    b.xmax,
                                    b.ymin,
                                    b.ymax
                                ))
                            }
                        }
                        self.generated = content.join("\n");
                        let dt = self.current_decode.as_ref().and_then(|current_decode| {
                            current_decode.start_time.and_then(|start_time| {
                                performance_now().map(|stop_time| stop_time - start_time)
                            })
                        });
                        self.status = match dt {
                            None => "processing succeeded!".to_string(),
                            Some(dt) => format!("processing succeeded in {:.2}s", dt,),
                        };
                        self.current_decode = None;
                        if let Err(err) = draw_bboxes(bboxes) {
                            self.status = format!("{err:?}")
                        }
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
        html! {
            <div style="margin: 2%;">
                <div><p>{"Running an object detection model in the browser using rust/wasm with "}
                <a href="https://github.com/huggingface/candle" target="_blank">{"candle!"}</a>
                </p>
                <p>{"Once the weights have loaded, click on the run button to process an image."}</p>
                <p><img id="bike-img" src="bike.jpeg"/></p>
                <p>{"Source: "}<a href="https://commons.wikimedia.org/wiki/File:V%C3%A9lo_parade_-_V%C3%A9lorution_-_bike_critical_mass.JPG">{"wikimedia"}</a></p>
                </div>
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
                <div>
                <canvas id="canvas" height="150" width="150"></canvas>
                </div>
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
