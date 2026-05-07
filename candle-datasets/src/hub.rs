use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeDataset};
use parquet::file::reader::SerializedFileReader;
use std::fs::File;

/// Re-export of the `FileReader` trait from the `parquet` crate.
///
/// This trait provides access to Parquet file metadata and row groups:
/// - [`FileReader::metadata`]
/// - [`FileReader::num_row_groups`]
/// - [`FileReader::get_row_group`]
/// - [`FileReader::get_row_iter`]
///
/// This is re-exported so downstream users of [`from_hub`] can use these
/// methods without needing to explicitly add `parquet` as a dependency.
///
/// # Example
/// ```
/// use candle_datasets::hub::{from_hub, FileReader};  // Re-exported trait
/// let client = hf_hub::HFClientSync::new().unwrap();
/// let files = from_hub(&client, "hf-internal-testing/dummy_image_text_data".to_string()).unwrap();
/// let num_rows = files[0].metadata().file_metadata().num_rows();
/// ```
pub use parquet::file::reader::FileReader;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("ApiError : {0}")]
    ApiError(#[from] hf_hub::HFError),

    #[error("IoError : {0}")]
    IoError(#[from] std::io::Error),

    #[error("ParquetError : {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),
}

fn sibling_to_parquet(
    rfilename: &str,
    repo: &HFRepositorySync<RepoTypeDataset>,
) -> Result<SerializedFileReader<File>, Error> {
    let local = repo
        .download_file()
        .filename(rfilename)
        .revision("refs/convert/parquet")
        .send()?;
    let file = File::open(local)?;
    Ok(SerializedFileReader::new(file)?)
}

/// Loads all `.parquet` files from a given dataset ID on the Hugging Face Hub.
///
/// This returns a list of `SerializedFileReader<File>` that can be used to read Parquet content.
///
/// # Example
/// ```
/// use candle_datasets::hub::{from_hub, FileReader};
/// let client = hf_hub::HFClientSync::new().unwrap();
/// let readers = from_hub(&client, "hf-internal-testing/dummy_image_text_data".to_string()).unwrap();
/// let metadata = readers[0].metadata();
/// assert_eq!(metadata.file_metadata().num_rows(), 20);
/// ```
pub fn from_hub(
    client: &HFClientSync,
    dataset_id: String,
) -> Result<Vec<SerializedFileReader<File>>, Error> {
    let repo = client.dataset("", dataset_id);
    let info = repo
        .info()
        .revision("refs/convert/parquet".to_string())
        .send()?;

    info.siblings
        .unwrap_or_default()
        .into_iter()
        .filter(|s| s.rfilename.ends_with(".parquet"))
        .map(|s| sibling_to_parquet(&s.rfilename, &repo))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset() {
        let client = HFClientSync::new().unwrap();
        let files = from_hub(
            &client,
            "hf-internal-testing/dummy_image_text_data".to_string(),
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].metadata().file_metadata().num_rows(), 20);
    }
}
