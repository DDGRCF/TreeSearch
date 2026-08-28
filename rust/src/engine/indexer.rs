//! Optional directory indexer with independently selectable progress reporting.
//!
//! Pipeline:
//!   File Discovery (ignore crate) → Parallel Parse (rayon) → Batch Insert (FTS5)
//!                                    ↓
//!                            mtime/size fingerprinting for incremental index

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use ignore::WalkBuilder;
#[cfg(feature = "progress")]
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tracing::{info, warn};

use crate::config::TreeSearchConfig;
use crate::document::Document;
use crate::engine::fts::FTS5Index;
use crate::parser::ParserRegistry;

/// File fingerprint for incremental indexing.
fn file_fingerprint(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let size = meta.len();
    Some(format!("{}:{}", mtime, size))
}

/// Discover files respecting .gitignore and .treesearchignore.
pub fn discover_files(
    root: &Path,
    config: &TreeSearchConfig,
    follow_symlinks: bool,
) -> Vec<PathBuf> {
    discover_files_with_status(root, config, follow_symlinks).files
}

/// One bounded discovery result plus whether the whole tree was observed.
struct FileDiscovery {
    files: Vec<PathBuf>,
    complete: bool,
}

/// Discovers supported files and records whether pruning evidence is complete.
fn discover_files_with_status(
    root: &Path,
    config: &TreeSearchConfig,
    follow_symlinks: bool,
) -> FileDiscovery {
    if config.max_dir_files == 0 {
        return FileDiscovery {
            files: Vec::new(),
            complete: false,
        };
    }
    let parser_registry = ParserRegistry::new();

    let mut walker_builder = WalkBuilder::new(root);
    walker_builder
        .follow_links(follow_symlinks)
        .hidden(true) // skip hidden files
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .add_custom_ignore_filename(".treesearchignore")
        .max_depth(None)
        .sort_by_file_path(|left, right| left.cmp(right));
    let walker = walker_builder.build();

    let mut files: Vec<PathBuf> = Vec::new();
    let mut complete = true;
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Walk error: {}", e);
                complete = false;
                continue;
            }
        };
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        if parser_registry.supports(&path) {
            files.push(path);
        }
        if files.len() >= config.max_dir_files {
            warn!(
                "Reached max_dir_files limit ({}), stopping discovery",
                config.max_dir_files
            );
            complete = false;
            break;
        }
    }
    FileDiscovery { files, complete }
}

/// Index a directory into an FTS5 database.
pub fn index_directory(
    root: &Path,
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    follow_symlinks: bool,
    show_progress: bool,
) -> Result<IndexStats> {
    let start = Instant::now();

    // Discover files
    let discovery = discover_files_with_status(root, config, follow_symlinks);
    let files = discovery.files;
    if files.is_empty() {
        if discovery.complete {
            let existing_meta = fts_index.get_all_index_meta()?;
            for old_path in existing_meta.keys() {
                if let Some(doc_id) = fts_index.get_doc_id_by_source_path(old_path)? {
                    fts_index.delete_document(&doc_id)?;
                }
            }
        }
        info!("No supported files found in {:?}", root);
        return Ok(IndexStats {
            files_found: 0,
            files_indexed: 0,
            files_skipped: 0,
            files_failed: 0,
            nodes_indexed: 0,
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    // Check existing fingerprints for incremental indexing
    let existing_meta = fts_index.get_all_index_meta()?;

    // Filter to files that need (re-)indexing
    let mut to_index: Vec<PathBuf> = Vec::new();
    let mut skipped = 0usize;
    for file in &files {
        let path_str = file.to_string_lossy().to_string();
        let fp = file_fingerprint(file);
        match (&fp, existing_meta.get(&path_str)) {
            (Some(new_fp), Some(old_fp)) if new_fp == old_fp => {
                skipped += 1;
            }
            _ => {
                to_index.push(file.clone());
            }
        }
    }

    info!(
        "Discovered {} files, {} unchanged (skipping), {} to index",
        files.len(),
        skipped,
        to_index.len()
    );

    if to_index.is_empty() {
        return Ok(IndexStats {
            files_found: files.len(),
            files_indexed: 0,
            files_skipped: skipped,
            files_failed: 0,
            nodes_indexed: 0,
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    #[cfg(not(feature = "progress"))]
    let _ = show_progress;

    #[cfg(feature = "progress")]
    let pb = if show_progress {
        let pb = ProgressBar::new(to_index.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        Some(pb)
    } else {
        None
    };

    // Parallel parsing in a request-local pool, rather than Rayon global defaults.
    let parser_registry = ParserRegistry::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(effective_concurrency(
            config.max_concurrency,
            to_index.len(),
        ))
        .build()
        .context("build bounded parser thread pool")?;
    let parse_results: Vec<(PathBuf, Result<Document>)> = pool.install(|| {
        to_index
            .par_iter()
            .map(|path| {
                let result = parse_file(&parser_registry, path);
                #[cfg(feature = "progress")]
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                (path.clone(), result)
            })
            .collect()
    });

    #[cfg(feature = "progress")]
    if let Some(ref pb) = pb {
        pb.finish_with_message("Parsing complete");
    }

    // Batch insert into FTS5 (single-threaded for SQLite)
    let mut indexed = 0usize;
    let mut failed = 0usize;
    let mut total_nodes = 0usize;
    let mut new_meta: HashMap<String, String> = HashMap::new();

    for (path, result) in parse_results {
        match result {
            Ok(doc) => {
                let path_str = path.to_string_lossy().to_string();
                match fts_index.index_document(&doc, false) {
                    Ok(count) => {
                        total_nodes += count;
                        indexed += 1;
                        if let Some(fp) = file_fingerprint(&path) {
                            new_meta.insert(path_str, fp);
                        }
                    }
                    Err(e) => {
                        warn!("Index error for {:?}: {}", path, e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Parse error for {:?}: {}", path, e);
                failed += 1;
            }
        }
    }

    // Batch commit fingerprints
    fts_index.commit()?;
    if !new_meta.is_empty() {
        fts_index.set_index_meta_batch(&new_meta)?;
    }

    let mut pruned = 0;
    if discovery.complete {
        // Only a complete walk proves that a previously indexed path was deleted.
        let all_paths: std::collections::HashSet<String> = files
            .iter()
            .map(|f| f.to_string_lossy().to_string())
            .collect();
        for old_path in existing_meta.keys() {
            if !all_paths.contains(old_path) {
                if let Some(doc_id) = fts_index.get_doc_id_by_source_path(old_path)? {
                    fts_index.delete_document(&doc_id)?;
                    pruned += 1;
                }
            }
        }
    } else {
        warn!("Skipping stale-document pruning because discovery was incomplete");
    }
    if pruned > 0 {
        info!("Pruned {} deleted documents from index", pruned);
    }

    let duration = start.elapsed().as_millis() as u64;
    info!(
        "Indexed {} files ({} nodes) in {}ms ({} failed, {} skipped)",
        indexed, total_nodes, duration, failed, skipped
    );

    Ok(IndexStats {
        files_found: files.len(),
        files_indexed: indexed,
        files_skipped: skipped,
        files_failed: failed,
        nodes_indexed: total_nodes,
        duration_ms: duration,
    })
}

/// Clamps parser parallelism to a non-zero value and the current work size.
fn effective_concurrency(configured: usize, work_items: usize) -> usize {
    configured.max(1).min(work_items.max(1))
}

/// Parse a single file using the parser registry.
fn parse_file(registry: &ParserRegistry, path: &Path) -> Result<Document> {
    let content = fs::read_to_string(path).with_context(|| format!("Failed to read {:?}", path))?;
    registry
        .parse_content(path, &content)?
        .ok_or_else(|| anyhow::anyhow!("No parser found for {:?}", path))
}

/// Index statistics.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub files_found: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_failed: usize,
    pub nodes_indexed: usize,
    pub duration_ms: u64,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Indexed: {} files ({} nodes) in {}ms | {} skipped, {} failed",
            self.files_indexed,
            self.nodes_indexed,
            self.duration_ms,
            self.files_skipped,
            self.files_failed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{discover_files_with_status, effective_concurrency};
    use crate::config::TreeSearchConfig;

    #[test]
    fn parser_concurrency_is_bounded_by_policy_and_work() {
        for (configured, work_items, expected) in [(0, 0, 1), (0, 5, 1), (2, 5, 2), (20, 5, 5)] {
            assert_eq!(
                effective_concurrency(configured, work_items),
                expected,
                "configured={configured}, work_items={work_items}"
            );
        }
    }

    #[cfg(feature = "parser-markdown")]
    #[test]
    fn bounded_discovery_is_marked_incomplete() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.md"), "# A").unwrap();
        std::fs::write(directory.path().join("b.md"), "# B").unwrap();
        let config = TreeSearchConfig {
            max_dir_files: 1,
            ..TreeSearchConfig::default()
        };

        let discovery = discover_files_with_status(directory.path(), &config, false);
        assert_eq!(discovery.files.len(), 1);
        assert!(!discovery.complete);
        assert!(discovery.files[0].ends_with("a.md"));
    }

    #[cfg(feature = "parser-markdown")]
    #[test]
    fn incomplete_discovery_never_prunes_and_complete_empty_walk_does() {
        use super::index_directory;
        use crate::engine::fts::FTS5Index;

        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("a.md");
        let second = directory.path().join("b.md");
        std::fs::write(&first, "# A").unwrap();
        std::fs::write(&second, "# B").unwrap();
        let index = FTS5Index::new(None, None).unwrap();
        let full_config = TreeSearchConfig {
            max_dir_files: 10,
            ..TreeSearchConfig::default()
        };
        index_directory(directory.path(), &index, &full_config, false, false).unwrap();
        assert_eq!(index.get_stats().unwrap().document_count, 2);

        let bounded_config = TreeSearchConfig {
            max_dir_files: 1,
            ..full_config.clone()
        };
        index_directory(directory.path(), &index, &bounded_config, false, false).unwrap();
        assert_eq!(index.get_stats().unwrap().document_count, 2);

        std::fs::remove_file(first).unwrap();
        std::fs::remove_file(second).unwrap();
        index_directory(directory.path(), &index, &full_config, false, false).unwrap();
        assert_eq!(index.get_stats().unwrap().document_count, 0);
    }
}
