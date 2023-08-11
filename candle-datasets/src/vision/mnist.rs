//! The MNIST hand-written digit dataset.
//!
//! The files can be obtained from the following link:
//! <http://yann.lecun.com/exdb/mnist/>
use candle::{DType, Device, Error, Result, Tensor};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{self, BufReader, Read};

fn read_u32<T: Read>(reader: &mut T) -> Result<u32> {
    let mut b = vec![0u8; 4];
    reader.read_exact(&mut b)?;
    let (result, _) = b.iter().rev().fold((0u64, 1u64), |(s, basis), &x| {
        (s + basis * u64::from(x), basis * 256)
    });
    Ok(result as u32)
}

fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("incorrect magic number {magic_number} != {expected}"),
        ))?;
    }
    Ok(())
}

fn read_labels(filename: &std::path::Path) -> Result<Tensor> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    read_labels_from_reader(&mut buf_reader)
}

fn read_labels_from_reader<T: Read>(reader: &mut T) -> Result<Tensor> {
    check_magic_number(reader, 2049)?;
    let samples = read_u32(reader)?;
    let mut data = vec![0u8; samples as usize];
    reader.read_exact(&mut data)?;
    let samples = data.len();
    Tensor::from_vec(data, samples, &Device::Cpu)
}

fn read_images(filename: &std::path::Path) -> Result<Tensor> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    read_images_from_reader(&mut buf_reader)
}

fn read_images_from_reader<T: Read>(reader: &mut T) -> Result<Tensor> {
    check_magic_number(reader, 2051)?;
    let samples = read_u32(reader)? as usize;
    let rows = read_u32(reader)? as usize;
    let cols = read_u32(reader)? as usize;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len];
    reader.read_exact(&mut data)?;
    let tensor = Tensor::from_vec(data, (samples, rows * cols), &Device::Cpu)?;
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

pub fn load() -> Result<crate::vision::Dataset> {
    // reference https://huggingface.co/datasets/mnist/blob/main/mnist.py
    let train_images_buf =
        ureq::get("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz")
            .call()
            .map_err(Error::wrap)?
            .into_reader();
    let train_images = read_images_from_reader(&mut GzDecoder::new(train_images_buf))?;

    let train_labels_buf =
        ureq::get("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz")
            .call()
            .map_err(Error::wrap)?
            .into_reader();
    let train_labels = read_labels_from_reader(&mut GzDecoder::new(train_labels_buf))?;

    let test_images_buf =
        ureq::get("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz")
            .call()
            .map_err(Error::wrap)?
            .into_reader();
    let test_images = read_images_from_reader(&mut GzDecoder::new(test_images_buf))?;

    let test_labels_buf =
        ureq::get("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz")
            .call()
            .map_err(Error::wrap)?
            .into_reader();
    let test_labels = read_labels_from_reader(&mut GzDecoder::new(test_labels_buf))?;

    Ok(crate::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}
