use candle_core::npy::NpzTensors;
use candle_core::{DType, Device, Error, Result, Tensor};
use std::borrow::Cow;

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "candle-{}-{}-{:?}",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl std::convert::AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap()
    }
}

#[test]
fn npy() -> Result<()> {
    let npy = Tensor::read_npy("tests/test.npy")?;
    assert_eq!(
        npy.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}

#[test]
fn npz() -> Result<()> {
    let npz = Tensor::read_npz("tests/test.npz")?;
    assert_eq!(npz.len(), 2);
    assert_eq!(npz[0].0, "x");
    assert_eq!(npz[1].0, "x_plus_one");
    assert_eq!(
        npz[1].1.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}

#[test]
fn safetensors() -> Result<()> {
    use candle_core::safetensors::Load;

    let tmp_file = TmpFile::create("st");
    let t = Tensor::arange(0f32, 24f32, &candle_core::Device::Cpu)?;
    t.save_safetensors("t", &tmp_file)?;
    // Load from file.
    let st = candle_core::safetensors::load(&tmp_file, &candle_core::Device::Cpu)?;
    let t2 = st.get("t").unwrap();
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    // Load from bytes.
    let bytes = std::fs::read(tmp_file)?;
    let st = candle_core::safetensors::SliceSafetensors::new(&bytes)?;
    let t2 = st.get("t").unwrap().load(&candle_core::Device::Cpu);
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    Ok(())
}

fn write_named_tensor(path: &TmpFile, name: &str) -> Result<()> {
    let tensor = Tensor::new(&[2u32, 4, 6], &Device::Cpu)?;
    Tensor::write_npz(&[(name, &tensor)], path)?;
    Ok(())
}

fn assert_named_tensor(path: &TmpFile, name: &str) -> Result<()> {
    let loaded = Tensor::read_npz(path)?;
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, name);
    assert_eq!(loaded[0].1.to_vec1::<u32>()?, vec![2, 4, 6]);
    let lazy = NpzTensors::new(path)?;
    assert_eq!(
        lazy.names().iter().map(|n| n.as_str()).collect::<Vec<_>>(),
        vec![name]
    );
    let tensor = lazy.get(name)?.expect("the named tensor must be indexed");
    assert_eq!(tensor.to_vec1::<u32>()?, vec![2, 4, 6]);
    Ok(())
}

#[test]
fn zip9_npz_borrowed_utf8_names() -> Result<()> {
    let file = TmpFile::create("zip9-utf8");
    write_named_tensor(&file, "données")?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&file.0)?)?;
    let entry = archive.by_index(0)?;
    assert!(matches!(entry.name()?, Cow::Borrowed("données.npy")));
    assert_named_tensor(&file, "données")
}

#[test]
fn zip9_npz_owned_cp437_names() -> Result<()> {
    let file = TmpFile::create("zip9-cp437");
    write_named_tensor(&file, "a")?;
    let mut bytes = std::fs::read(&file.0)?;
    let positions: Vec<_> = bytes
        .windows(5)
        .enumerate()
        .filter_map(|(i, x)| (x == b"a.npy").then_some(i))
        .collect();
    assert_eq!(positions.len(), 2);
    for index in positions {
        bytes[index] = 0x82;
    }
    std::fs::write(&file.0, bytes)?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&file.0)?)?;
    let entry = archive.by_index(0)?;
    assert!(matches!(entry.name()?, Cow::Owned(name) if name == "é.npy"));
    assert_named_tensor(&file, "é")
}

#[test]
fn zip9_invalid_npz_returns_zip_errors() -> Result<()> {
    let file = TmpFile::create("zip9-invalid-npz");
    std::fs::write(&file.0, b"not a zip archive")?;
    assert!(matches!(Tensor::read_npz(&file.0), Err(Error::Zip(_))));
    assert!(matches!(NpzTensors::new(&file.0), Err(Error::Zip(_))));
    Ok(())
}

#[test]
fn zip9_invalid_pth_returns_zip_error() -> Result<()> {
    let file = TmpFile::create("zip9-invalid-pth");
    std::fs::write(&file.0, b"not a zip archive")?;
    assert!(matches!(
        candle_core::pickle::read_pth_tensor_info(&file.0, false, None),
        Err(Error::Zip(_))
    ));
    Ok(())
}
