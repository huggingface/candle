//! The MNIST hand-written digit dataset.
//!
//! The files can be obtained from the following link:
//! <http://yann.lecun.com/exdb/mnist/>
use candle::{CpuDevice, CpuStorage, DType, Error, Result, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{self, BufReader, Read};

type CpuTensor = Tensor<CpuStorage>;

fn read_u32<T: Read>(reader: &mut T) -> std::io::Result<u32> {
    use byteorder::ReadBytesExt;
    reader.read_u32::<byteorder::BigEndian>()
}

fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        Err(io::Error::other(format!(
            "incorrect magic number {magic_number} != {expected}"
        )))?;
    }
    Ok(())
}

fn read_labels(filename: &std::path::Path) -> Result<CpuTensor> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;
    let samples = data.len();
    Tensor::from_vec(data, samples, &CpuDevice)
}

fn read_images(filename: &std::path::Path) -> Result<CpuTensor> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)? as usize;
    let rows = read_u32(&mut buf_reader)? as usize;
    let cols = read_u32(&mut buf_reader)? as usize;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len];
    buf_reader.read_exact(&mut data)?;
    let tensor = Tensor::from_vec(data, (samples, rows * cols), &CpuDevice)?;
    tensor.to_dtype(DType::F32)? / 255.
}

pub fn load_dir<T: AsRef<std::path::Path>>(dir: T) -> Result<crate::vision::Dataset> {
    let dir = dir.as_ref();
    let train_images = read_images(&dir.join("train-images-idx3-ubyte"))?;
    let train_labels = read_labels(&dir.join("train-labels-idx1-ubyte"))?;
    let test_images = read_images(&dir.join("t10k-images-idx3-ubyte"))?;
    let test_labels = read_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
    Ok(crate::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}

fn load_parquet(parquet: SerializedFileReader<std::fs::File>) -> Result<(CpuTensor, CpuTensor)> {
    let samples = parquet.metadata().file_metadata().num_rows() as usize;
    let mut buffer_images: Vec<u8> = Vec::with_capacity(samples * 784);
    let mut buffer_labels: Vec<u8> = Vec::with_capacity(samples);
    for row in parquet.into_iter().flatten() {
        for (_name, field) in row.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                buffer_labels.push(*label as u8);
            }
        }
    }
    let images = (Tensor::from_vec(buffer_images, (samples, 784), &CpuDevice)?
        .to_dtype(DType::F32)?
        / 255.)?;
    let labels = Tensor::from_vec(buffer_labels, (samples,), &CpuDevice)?;
    Ok((images, labels))
}

pub(crate) fn load_mnist_like(
    dataset_id: &str,
    revision: &str,
    test_filename: &str,
    train_filename: &str,
) -> Result<crate::vision::Dataset> {
    let api = Api::new().map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let repo = Repo::with_revision(
        dataset_id.to_string(),
        RepoType::Dataset,
        revision.to_string(),
    );
    let repo = api.repo(repo);
    let test_parquet_filename = repo
        .get(test_filename)
        .map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let train_parquet_filename = repo
        .get(train_filename)
        .map_err(|e| Error::Msg(format!("Api error: {e}")))?;
    let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)
        .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
    let (test_images, test_labels) = load_parquet(test_parquet)?;
    let (train_images, train_labels) = load_parquet(train_parquet)?;
    Ok(crate::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}

pub fn load() -> Result<crate::vision::Dataset> {
    load_mnist_like(
        "ylecun/mnist",
        "refs/convert/parquet",
        "mnist/test/0000.parquet",
        "mnist/train/0000.parquet",
    )
}
