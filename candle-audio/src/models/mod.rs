//! Native Candle adapters. Implement the crate's traits to add other models;
//! the session, PCM conversion, event sinks, and CLI protocol need not change.
pub mod codec_stream;
pub mod codecs;
pub mod csm;
pub mod lfm3;
pub mod load;
pub mod metavoice;
pub mod mimi;
pub mod parler;
pub mod registry;
pub mod whisper;

pub mod musicgen;
pub mod voxtral;

pub mod gemma4;
