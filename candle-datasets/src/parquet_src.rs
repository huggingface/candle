//! Generic parquet row reader for training datasets.
//!
//! [`ParquetSource`] wraps a parquet file and yields one strongly-typed row
//! at a time. Users plug in their own row struct via the [`FromParquetRow`]
//! trait, which converts a `parquet::record::Row` into the target type.
//!
//! This is deliberately unopinionated about what rows look like — it stays
//! out of the business of DataFrame manipulation. For heavy-duty dataframe
//! work on the parquet contents, use `polars` or `arrow-rs` directly; for a
//! lean "iterate rows and build tensors" path, use this module.
//!
//! # Example
//!
//! ```no_run
//! use candle::Result;
//! use candle_datasets::parquet_src::{FromParquetRow, ParquetSource};
//! use parquet::record::Row;
//!
//! struct Example {
//!     id: i64,
//!     label: i32,
//! }
//!
//! impl FromParquetRow for Example {
//!     fn from_row(row: &Row) -> Result<Self> {
//!         let mut id = None;
//!         let mut label = None;
//!         for (name, field) in row.get_column_iter() {
//!             match (name.as_str(), field) {
//!                 ("id", parquet::record::Field::Long(v)) => id = Some(*v),
//!                 ("label", parquet::record::Field::Int(v)) => label = Some(*v),
//!                 _ => {}
//!             }
//!         }
//!         Ok(Self {
//!             id: id.ok_or_else(|| candle::Error::Msg("missing id".into()))?,
//!             label: label.ok_or_else(|| candle::Error::Msg("missing label".into()))?,
//!         })
//!     }
//! }
//!
//! fn run() -> Result<()> {
//!     let src: ParquetSource<Example> = ParquetSource::open("data.parquet")?;
//!     for row in src.iter()? {
//!         let example = row?;
//!         let _ = (example.id, example.label);
//!         // ... build tensors, append to batch, etc.
//!     }
//!     Ok(())
//! }
//! ```

use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use candle::{Error, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter as ParquetRowIter;
use parquet::record::Row;

/// Trait implemented by user row types so [`ParquetSource`] knows how to
/// decode one parquet record into that type.
///
/// Implementations typically walk `row.get_column_iter()` and match on
/// `parquet::record::Field` variants. Return a [`candle::Error::Msg`] for
/// missing or mis-typed columns; the error surfaces through the iterator as
/// a `Result::Err`.
pub trait FromParquetRow: Sized {
    fn from_row(row: &Row) -> Result<Self>;
}

/// A parquet file opened for row-by-row reading, parameterized by the target
/// row type `T`.
///
/// Opens and parses the file footer on construction. Reading itself is lazy
/// — call [`ParquetSource::iter`] to get an iterator that decodes rows on
/// demand.
pub struct ParquetSource<T> {
    path: PathBuf,
    num_rows: usize,
    _marker: PhantomData<T>,
}

impl<T: FromParquetRow> ParquetSource<T> {
    /// Open a parquet file and read its footer to determine the row count.
    ///
    /// The file is re-opened for each call to [`ParquetSource::iter`] so a
    /// single `ParquetSource` can be iterated multiple times (useful for
    /// multiple training epochs).
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

    /// Total number of rows in the parquet file.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Return a fresh iterator over all rows in the file.
    ///
    /// The file is re-opened each time this method is called, so iterating
    /// twice is safe (e.g. for two training epochs) at the cost of one
    /// syscall and a new parquet metadata parse per epoch.
    pub fn iter(&self) -> Result<RowIter<T>> {
        let file = File::open(&self.path)?;
        let reader = SerializedFileReader::new(file)
            .map_err(|e| Error::Msg(format!("parquet open error: {e}")))?;
        // `from_file_into` takes ownership of the reader via `Box<dyn FileReader>`
        // so the resulting `RowIter` is self-contained and has no borrow of
        // the `ParquetSource`.
        let inner = ParquetRowIter::from_file_into(Box::new(reader));
        Ok(RowIter {
            inner,
            _marker: PhantomData,
        })
    }
}

/// Iterator yielded by [`ParquetSource::iter`]. Decodes one row at a time
/// into `T` via [`FromParquetRow::from_row`].
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

    /// Unique temp path scoped to this test run; parent dir is created and
    /// the file is removed on `Drop`. Avoids adding a dev-dependency on the
    /// `tempfile` crate.
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
