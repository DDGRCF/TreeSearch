//! Pure in-memory tree search plus optional SQLite FTS and directory-indexing adapters.

pub mod candidate_search;
#[cfg(feature = "sqlite-fts")]
pub mod fts;
#[cfg(feature = "directory-indexer")]
pub mod indexer;
#[cfg(feature = "sqlite-fts")]
pub mod search;
pub mod tree_walker;
