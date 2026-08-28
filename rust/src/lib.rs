//! Library-first TreeSearch API with optional CLI, indexer, output, tokenizer, and parser layers.

pub mod config;
pub mod document;
pub mod engine;
#[cfg(feature = "output")]
pub mod output;
pub mod parser;
pub mod scorer;
pub mod tokenizer;
