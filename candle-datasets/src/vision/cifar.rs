//! The CIFAR-10 dataset.
//!
//! The files can be downloaded from the following page:
//! <https://www.cs.toronto.edu/~kriz/cifar.html>
//! The binary version of the dataset is used.
use crate::vision::Dataset;
use candle::{DType, Device, Result, Tensor};
use std::fs::File;
use std::io::{BufReader, Read};

const W: usize = 32;
const H: usize = 32;
const C: usize = 3;
const BYTES_PER_IMAGE: usize = W * H * C + 1;
const SAMPLES_PER_FILE: usize = 10000;

fn read_file(filename: &std::path::Path) -> Result<(Tensor, Tensor)> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    let mut data = vec![0u8; SAMPLES_PER_FILE * BYTES_PER_IMAGE];
    buf_reader.read_exact(&mut data)?;
    let mut images = vec![];
    let mut labels = vec![];
    for index in 0..SAMPLES_PER_FILE {
        let content_offset = BYTES_PER_IMAGE * index;
        labels.push(data[content_offset]);
        images.push(&data[1 + content_offset..content_offset + BYTES_PER_IMAGE]);
    }
    let images: Vec<u8> = images
        .iter()
        .copied()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let labels = Tensor::from_vec(labels, SAMPLES_PER_FILE, &Device::Cpu)?;
    let images = Tensor::from_vec(images, (SAMPLES_PER_FILE, C, H, W), &Device::Cpu)?;
    let images = (images.to_dtype(DType::F32)? / 255.)?;
    Ok((images, labels))
}

pub fn load_dir<T: AsRef<std::path::Path>>(dir: T) -> Result<Dataset> {
    let dir = dir.as_ref();
    let (test_images, test_labels) = read_file(&dir.join("test_batch.bin"))?;
    let train_images_and_labels = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ]
    .iter()
    .map(|x| read_file(&dir.join(x)))
    .collect::<Result<Vec<_>>>()?;
    let (train_images, train_labels): (Vec<_>, Vec<_>) =
        train_images_and_labels.into_iter().unzip();
    Ok(Dataset {
        train_images: Tensor::cat(&train_images, 0)?,
        train_labels: Tensor::cat(&train_labels, 0)?,
        test_images,
        test_labels,
        labels: 10,
    })
}
