//! Datasets & Dataloaders for Candle.
//!
//! This crate augments the minimal [`Batcher`] with a small number of
//! general-purpose helpers for training-loop data plumbing:
//!
//! - [`pad`]         — variable-length sequence padding and batch collation
//! - [`shuffle`]     — seeded shuffled-index and shuffled-iterator helpers
//! - [`parquet_src`] — generic parquet row reader via the
//!   [`FromParquetRow`](parquet_src::FromParquetRow) trait
//! - [`ragged`]      — flat-ragged tensor cache persisted via safetensors
//!
//! These are intentionally narrow. They do not reimplement DataFrame-style
//! indexing or filtering — pair them with `polars` or `arrow-rs` if you
//! need those.
//!
//! # Typical training-loop skeleton
//!
//! ```no_run
//! use candle::{Device, Result, Tensor};
//! use candle_datasets::pad::pad_and_stack_2d;
//! use candle_datasets::parquet_src::{FromParquetRow, ParquetSource};
//! use candle_datasets::shuffle::shuffled_indices;
//! use parquet::record::Row;
//!
//! struct Example {
//!     sequence: Vec<f32>,
//!     feature_dim: usize,
//!     label: i64,
//! }
//!
//! impl FromParquetRow for Example {
//!     fn from_row(_row: &Row) -> Result<Self> {
//!         // ... decode user fields here ...
//!         Ok(Self { sequence: vec![], feature_dim: 64, label: 0 })
//!     }
//! }
//!
//! fn one_epoch(src: &ParquetSource<Example>, epoch: u64, dev: &Device) -> Result<()> {
//!     let examples: Vec<Example> =
//!         src.iter()?.collect::<Result<Vec<_>>>()?;
//!     let order = shuffled_indices(examples.len(), 42 + epoch);
//!     for batch_start in (0..order.len()).step_by(16) {
//!         let batch_end = (batch_start + 16).min(order.len());
//!         let items: Vec<Tensor> = order[batch_start..batch_end]
//!             .iter()
//!             .map(|&i| {
//!                 let ex = &examples[i];
//!                 let t_len = ex.sequence.len() / ex.feature_dim;
//!                 Tensor::from_slice(&ex.sequence, (t_len, ex.feature_dim), dev)
//!             })
//!             .collect::<Result<Vec<_>>>()?;
//!         let (_padded, _mask) = pad_and_stack_2d(&items, 0.0)?;
//!         // ... forward / backward / step ...
//!     }
//!     Ok(())
//! }
//! ```

pub mod batcher;
pub mod hub;
pub mod nlp;
pub mod pad;
pub mod parquet_src;
pub mod ragged;
pub mod shuffle;
pub mod vision;

pub use batcher::Batcher;
