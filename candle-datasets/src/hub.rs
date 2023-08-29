use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use parquet::file::reader::SerializedFileReader;
use std::fs::File;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("ApiError : {0}")]
    ApiError(#[from] hf_hub::api::sync::ApiError),

    #[error("IoError : {0}")]
    IoError(#[from] std::io::Error),

    #[error("ParquetError : {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),
}

fn sibling_to_parquet(
    rfilename: &str,
    repo: &ApiRepo,
) -> Result<SerializedFileReader<File>, Error> {
    let local = repo.get(rfilename)?;
    let file = File::open(local)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader)
}

pub fn from_hub(api: &Api, dataset_id: String) -> Result<Vec<SerializedFileReader<File>>, Error> {
    let repo = Repo::with_revision(
        dataset_id,
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    );
    let repo = api.repo(repo);
    let info = repo.info()?;

    let files: Result<Vec<_>, _> = info
        .siblings
        .into_iter()
        .filter_map(|s| -> Option<Result<_, _>> {
            let filename = s.rfilename;
            if filename.ends_with(".parquet") {
                let reader_result = sibling_to_parquet(&filename, &repo);
                Some(reader_result)
            } else {
                None
            }
        })
        .collect();
    let files = files?;

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::file::reader::FileReader;

    #[test]
    fn test_dataset() {
        let api = Api::new().unwrap();
        let files = from_hub(
            &api,
            "hf-internal-testing/dummy_image_text_data".to_string(),
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].metadata().file_metadata().num_rows(), 20);
    }
}
