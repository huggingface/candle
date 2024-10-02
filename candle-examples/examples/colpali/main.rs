use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::colpali::Model;
use candle_transformers::models::{colpali, paligemma};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use image::DynamicImage;
use pdf2image::{RenderOptionsBuilder, PDF};
use tokenizers::Tokenizer;

struct PageRetriever {
    model: Model,
    config: paligemma::Config,
    pdf: PDF,
    device: Device,
    tokenizer: Tokenizer,
    range: pdf2image::Pages,
    batch_size: usize,
    top_k: usize,
}

impl PageRetriever {
    fn new(
        model: Model,
        config: paligemma::Config,
        pdf: PDF,
        tokenizer: Tokenizer,
        device: &Device,
        range: Option<pdf2image::Pages>,
        batch_size: usize,
        top_k: usize,
    ) -> Self {
        let page_count = pdf.page_count();
        Self {
            model,
            config,
            pdf,
            device: device.clone(),
            tokenizer,
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

    fn tokenize_batch(&self, prompts: Vec<&str>) -> Result<Tensor> {
        let tokens = self.tokenizer.encode_batch(prompts, true).map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let input = Tensor::stack(&token_ids, 0)?;
        Ok(input)
    }

    fn images_to_tensor(
        &self,
        pages: &[DynamicImage],
        image_size: usize,
    ) -> anyhow::Result<Tensor> {
        let mut images = vec![];
        for page in pages.iter() {
            let img = page.resize_to_fill(
                image_size as u32,
                image_size as u32,
                image::imageops::FilterType::Triangle,
            );
            let img = img.to_rgb8();
            let img = img.into_raw();
            let img = Tensor::from_vec(img, (image_size, image_size, 3), &Device::Cpu)?
                .permute((2, 0, 1))?
                .to_dtype(DType::F32)?
                .affine(2. / 255., -1.)?;
            images.push(img);
        }
        let images = Tensor::stack(&images, 0)?;
        Ok(images)
    }

    fn retrieve(&mut self, prompt: &str) -> Result<Vec<usize>> {
        let dtype = if self.device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let dummy_prompt: &str = "Describe the image";

        let input = self.tokenize_batch(vec![prompt])?;
        let dummy_input = self.tokenize_batch(vec![dummy_prompt])?;

        let pages = self.get_images_from_pdf()?;
        let mut all_scores = Vec::new();
        for batch in pages.chunks(self.batch_size) {
            let page_images = self
                .images_to_tensor(batch, self.config.vision_config.image_size)?
                .to_device(&self.device)?
                .to_dtype(dtype)?;
            let dummy_input = dummy_input.repeat((page_images.dims()[0], 0))?;

            let image_embeddings = self.model.forward_images(&page_images, &dummy_input)?;
            let text_embeddings = self.model.forward_text(&input)?;

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

    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "vidore/colpali-v1.2-merged".to_string(),
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .repo(Repo::with_revision(
                "vidore/colpali".to_string(),
                RepoType::Model,
                "main".to_string(),
            ))
            .get("tokenizer.json")?,
    };

    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    let start = std::time::Instant::now();

    let config: paligemma::Config = paligemma::Config::paligemma_3b_448();

    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let device = candle_examples::device(false)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = colpali::Model::new(&config, vb)?;

    let pdf = PDF::from_file(args.pdf)?;

    // check if start and end given in arg
    let range = if let (Some(start), Some(end)) = (args.start, args.end) {
        pdf2image::Pages::Range(start..=end)
    } else {
        pdf2image::Pages::Range(1..=pdf.page_count()) // can use pdf2image::Pages::All but there is a bug in the library which causes the first page to rendered twice.
    };

    let mut retriever =
        PageRetriever::new(model, config, pdf, tokenizer, &device, Some(range), 4, 3);
    let top_k_indices = retriever.retrieve(&args.prompt)?;

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
