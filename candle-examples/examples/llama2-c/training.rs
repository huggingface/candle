use candle::Result;

pub struct Dataset {
    valid_tokens: Vec<memmap2::Mmap>,
    train_tokens: Vec<memmap2::Mmap>,
}

fn mmap_file(p: &std::path::PathBuf) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(p)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    Ok(mmap)
}

impl Dataset {
    pub fn new<P: AsRef<std::path::Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let mut bin_files = vec![];
        for file in std::fs::read_dir(dir)?.flatten() {
            let file = file.path();
            if let Some(extension) = file.extension() {
                if extension == "bin" {
                    bin_files.push(file)
                }
            }
        }
        if bin_files.len() < 2 {
            candle::bail!("found less than two bin files in {:?}", dir)
        }
        bin_files.sort();
        let valid_tokens = mmap_file(&bin_files[0])?;
        let train_tokens = bin_files[1..]
            .iter()
            .map(mmap_file)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            valid_tokens: vec![valid_tokens],
            train_tokens,
        })
    }
}

pub fn run(args: &crate::TrainingCmd, common_args: &crate::Args) -> Result<()> {
    let _device = candle_examples::device(common_args.cpu)?;
    let dataset = Dataset::new(&args.pretokenized_dir)?;
    println!(
        "loaded dataset, train: {} files, valid: {} files",
        dataset.train_tokens.len(),
        dataset.valid_tokens.len()
    );
    Ok(())
}
