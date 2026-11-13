//! Native Candle adapters. Implement the crate's traits to add other models;
//! the session, PCM conversion, event sinks, and CLI protocol need not change.
pub mod load;
pub mod mimi;
pub mod parler;
pub mod whisper;
