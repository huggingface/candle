use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use candle::{Error, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter as ParquetRowIter;
use parquet::record::Row;

pub trait FromParquetRow: Sized {
    fn from_row(row: &Row) -> Result<Self>;
}

pub struct ParquetSource<T> {
    path: PathBuf,
    num_rows: usize,
    _marker: PhantomData<T>,
}

impl<T: FromParquetRow> ParquetSource<T> {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let reader = SerializedFileReader::new(file)
            .map_err(|e| Error::Msg(format!("parquet open error: {e}")))?;
        let meta = reader.metadata();
        let num_rows: i64 = meta.file_metadata().num_rows();
        if num_rows < 0 {
            candle::bail!("parquet file reports negative row count: {num_rows}");
        }
        Ok(Self {
            path,
            num_rows: num_rows as usize,
            _marker: PhantomData,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn iter(&self) -> Result<RowIter<T>> {
        let file = File::open(&self.path)?;
        let reader = SerializedFileReader::new(file)
            .map_err(|e| Error::Msg(format!("parquet open error: {e}")))?;
        let inner = ParquetRowIter::from_file_into(Box::new(reader));
        Ok(RowIter {
            inner,
            _marker: PhantomData,
        })
    }
}

pub struct RowIter<T> {
    inner: ParquetRowIter<'static>,
    _marker: PhantomData<T>,
}

impl<T: FromParquetRow> Iterator for RowIter<T> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next()? {
            Ok(row) => Some(T::from_row(&row)),
            Err(e) => Some(Err(Error::Msg(format!("parquet decode error: {e}")))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::data_type::Int64Type;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::record::Field;
    use parquet::schema::parser::parse_message_type;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempPath(PathBuf);

    impl TempPath {
        fn new(stem: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("candle_datasets_test_{pid}_{nanos}_{n}"));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir.join(format!("{stem}.parquet")))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            if let Some(parent) = self.0.parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }

    struct TestRow {
        id: i64,
        value: i64,
    }

    impl FromParquetRow for TestRow {
        fn from_row(row: &Row) -> Result<Self> {
            let mut id = None;
            let mut value = None;
            for (name, field) in row.get_column_iter() {
                match (name.as_str(), field) {
                    ("id", Field::Long(v)) => id = Some(*v),
                    ("value", Field::Long(v)) => value = Some(*v),
                    _ => {}
                }
            }
            Ok(Self {
                id: id.ok_or_else(|| Error::Msg("missing id".into()))?,
                value: value.ok_or_else(|| Error::Msg("missing value".into()))?,
            })
        }
    }

    fn write_test_parquet(path: &Path, n: usize) {
        let message_type = "
            message schema {
                REQUIRED INT64 id;
                REQUIRED INT64 value;
            }
        ";
        let schema = Arc::new(parse_message_type(message_type).unwrap());
        let props = Arc::new(WriterProperties::builder().build());
        let file = File::create(path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();

        let ids: Vec<i64> = (0..n as i64).collect();
        let values: Vec<i64> = (0..n as i64).map(|i| i * 10).collect();

        let mut row_group = writer.next_row_group().unwrap();

        let mut id_writer = row_group.next_column().unwrap().unwrap();
        id_writer
            .typed::<Int64Type>()
            .write_batch(&ids, None, None)
            .unwrap();
        id_writer.close().unwrap();

        let mut value_writer = row_group.next_column().unwrap().unwrap();
        value_writer
            .typed::<Int64Type>()
            .write_batch(&values, None, None)
            .unwrap();
        value_writer.close().unwrap();

        row_group.close().unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn roundtrip_and_num_rows() {
        let tmp = TempPath::new("roundtrip");
        write_test_parquet(tmp.path(), 5);

        let src: ParquetSource<TestRow> = ParquetSource::open(tmp.path()).unwrap();
        assert_eq!(src.num_rows(), 5);

        let rows: Result<Vec<TestRow>> = src.iter().unwrap().collect();
        let rows = rows.unwrap();
        assert_eq!(rows.len(), 5);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.id, i as i64);
            assert_eq!(r.value, i as i64 * 10);
        }
    }

    #[test]
    fn iter_twice_for_multiple_epochs() {
        let tmp = TempPath::new("iter_twice");
        write_test_parquet(tmp.path(), 3);

        let src: ParquetSource<TestRow> = ParquetSource::open(tmp.path()).unwrap();
        let first: Vec<i64> = src.iter().unwrap().map(|r| r.unwrap().id).collect();
        let second: Vec<i64> = src.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(first, second);
        assert_eq!(first, vec![0, 1, 2]);
    }

    #[test]
    fn open_nonexistent_file_errors() {
        let result: Result<ParquetSource<TestRow>> = ParquetSource::open("/does/not/exist.parquet");
        assert!(result.is_err());
    }
}
