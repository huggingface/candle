use anyhow::Result;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::idefics3::model::{ColIdefics3Model, Idefics3Config};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use image::DynamicImage;
use pdf2image::{RenderOptionsBuilder, PDF};

mod processing;
use processing::Idefics3Processor;

struct PageRetriever {
    model: ColIdefics3Model,
    pdf: PDF,
    device: Device,
    processor: Idefics3Processor,
    range: pdf2image::Pages,
    batch_size: usize,
    top_k: usize,
}

impl PageRetriever {
    fn new(
        model: ColIdefics3Model,
        pdf: PDF,
        processor: Idefics3Processor,
        device: &Device,
        range: Option<pdf2image::Pages>,
        batch_size: usize,
        top_k: usize,
    ) -> Self {
        let page_count = pdf.page_count();
        Self {
            model,
            pdf,
            device: device.clone(),
            processor,
            range: range.unwrap_or_else(|| pdf2image::Pages::Range(1..=page_count)),
            batch_size,
            top_k,
        }
    }

    fn get_images_from_pdf(&self) -> Result<Vec<DynamicImage>> {
        let pages = self
            .pdf
            .render(self.range.clone(), RenderOptionsBuilder::default().build()?)?;
        Ok(pages)
    }

    fn retrieve(&mut self, prompt: &str) -> Result<Vec<usize>> {
        let pages = self.get_images_from_pdf()?;
        let mut all_scores = Vec::new();
        for batch in pages.chunks(self.batch_size) {
            let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
                self.processor.preprocess(batch, &self.device)?;

            let image_embeddings = self.model.forward(
                &input_ids,
                &attention_mask,
                &Some(pixel_values),
                &pixel_attention_mask,
            )?;

            // println!("Image embeddings: {}", image_embeddings);

            let (input, attention_mask) =
                self.processor.tokenize_batch(vec![prompt], &self.device)?;

            // println!("Input: {}", input);

            let text_embeddings = self.model.forward(&input, &attention_mask, &None, &None)?;

            // println!("Text embeddings: {}", text_embeddings);
            let scores = text_embeddings
                .unsqueeze(1)?
                .broadcast_matmul(&image_embeddings.unsqueeze(0)?.transpose(3, 2)?)?
                .max(3)?
                .sum(2)?;

            let batch_scores: Vec<f32> = scores
                .to_dtype(DType::F32)?
                .to_vec2()?
                .into_iter()
                .flatten()
                .collect();
            all_scores.extend(batch_scores);
        }

        let mut indices: Vec<usize> = (0..all_scores.len()).collect();
        indices.sort_by(|a, b| all_scores[*b].partial_cmp(&all_scores[*a]).unwrap());

        let top_k_indices = indices[0..self.top_k].to_vec();

        Ok(top_k_indices)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// number of top pages to show.
    #[arg(long, default_value_t = 3)]
    top_k: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    pdf: String,

    #[arg(long)]
    start: Option<u32>,

    #[arg(long)]
    end: Option<u32>,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let api = Api::new().unwrap();
    let repo = api.repo(Repo::new(
        "akshayballal/colSmol-256M-merged".to_string(),
        RepoType::Model,
    ));
    let config_file = repo.get("config.json").unwrap();
    let model_file = repo.get("model.safetensors").unwrap();
    let config: Idefics3Config =
        serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();

    let processor = Idefics3Processor::from_pretrained("akshayballal/colSmol-256M-merged").unwrap();

    let device = candle_examples::device(false).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file], candle::DType::BF16, &device).unwrap()
    };
    let model = ColIdefics3Model::load(&config, false, vb).unwrap();

    let pdf = PDF::from_file(args.pdf)?;

    // check if start and end given in arg
    let range = if let (Some(start), Some(end)) = (args.start, args.end) {
        pdf2image::Pages::Range(start..=end)
    } else {
        pdf2image::Pages::Range(1..=pdf.page_count()) // can use pdf2image::Pages::All but there is a bug in the library which causes the first page to rendered twice.
    };

    let mut retriever = PageRetriever::new(model, pdf, processor, &device, Some(range), 1, 3);

    let start_time = std::time::Instant::now();
    let top_k_indices = retriever.retrieve(&args.prompt)?;
    let end_time = std::time::Instant::now();
    println!("Time taken: {:?}", end_time.duration_since(start_time));

    println!("Prompt: {}", args.prompt);
    println!(
        "top {} page numbers that contain similarity to the prompt",
        retriever.top_k
    );
    println!("-----------------------------------");
    for index in top_k_indices {
        println!("Page: {:?}", index + 1);
    }
    println!("-----------------------------------");
    Ok(())
}
