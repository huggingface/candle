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
    let file = File::open(&local)?;
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

    #[test]
    fn test_dataset() {
        let api = Api::new().unwrap();
        let files = from_hub(
            &api,
            "hf-internal-testing/dummy_image_text_data".to_string(),
        )
        .unwrap();
        assert_eq!(files.len(), 1);

        let mut rows = files.into_iter().flat_map(|r| r.into_iter());

        let row = rows.next().unwrap().unwrap();
        let mut col_iter = row.get_column_iter();

        // First element is an image
        col_iter.next();
        assert_eq!(
            col_iter.next().unwrap().1,
            &parquet::record::Field::Str("a drawing of a green pokemon with red eyes".to_string())
        );

        // Keep for now to showcase how to use.
        //for row in rows {
        //    if let Ok(row) = row {
        //        for (_idx, (_name, field)) in row.get_column_iter().enumerate() {
        //            if let parquet::record::Field::Str(value) = field {
        //                println!("Value {value:?}");
        //            }
        //        }
        //    }
        //}
    }
}
