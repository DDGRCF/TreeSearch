//! SQLite FTS5 full-text search engine for tree-structured documents.
//!
//! Rust port of Python `treesearch/fts.py`. Single-file storage: tree structures,
//! FTS5 indexes, and incremental metadata are all stored in one SQLite database.
//!
//! Key features:
//!   - WAL mode for concurrent read/write
//!   - MD-aware schema: front_matter, title, summary, body, code_blocks
//!   - Hierarchical column weighting via FTS5 bm25()
//!   - Content hash for incremental indexing
//!   - Ancestor score propagation for tree search

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use md5::{Digest, Md5};
use regex::Regex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::config::{CjkTokenizerMode, TreeSearchConfig};
use crate::document::{Document, Node, NodeId, SourceType};

/// Safety bound for one SQLite-to-memory candidate transfer.
const DEFAULT_BATCH_CANDIDATE_LIMIT: usize = 5_000;

/// Metadata keyed by `(node_id, doc_id)` for batched result assembly.
type NodeMetadataMap = HashMap<(String, String), (String, String, u32)>;

// ---------------------------------------------------------------------------
// FTS5 column weights
// ---------------------------------------------------------------------------

/// Column weights for FTS5 bm25() ranking.
#[derive(Debug, Clone)]
pub struct FtsWeights {
    pub title: f64,
    pub summary: f64,
    pub body: f64,
    pub code_blocks: f64,
    pub front_matter: f64,
}

impl Default for FtsWeights {
    fn default() -> Self {
        Self {
            title: 5.0,
            summary: 2.0,
            body: 10.0,
            code_blocks: 1.0,
            front_matter: 2.0,
        }
    }
}

impl FtsWeights {
    /// Create weights from a `TreeSearchConfig`.
    pub fn from_config(config: &TreeSearchConfig) -> Self {
        Self {
            title: config.fts_title_weight,
            summary: config.fts_summary_weight,
            body: config.fts_body_weight,
            code_blocks: config.fts_code_weight,
            front_matter: config.fts_front_matter_weight,
        }
    }

    /// Return weight arguments for `bm25(fts_nodes, ...)`.
    fn bm25_args(&self) -> String {
        format!(
            "{}, {}, {}, {}, {}",
            self.title, self.summary, self.body, self.code_blocks, self.front_matter
        )
    }

    /// Rejects values that cannot safely participate in SQLite BM25 ranking.
    fn validate(&self) -> Result<()> {
        let weights = [
            self.title,
            self.summary,
            self.body,
            self.code_blocks,
            self.front_matter,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            bail!("FTS5 weights must be finite and non-negative");
        }
        if weights.iter().all(|weight| *weight == 0.0) {
            bail!("at least one FTS5 weight must be positive");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FTS result type
// ---------------------------------------------------------------------------

/// A single FTS5 search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsResult {
    pub node_id: NodeId,
    pub doc_id: String,
    pub title: String,
    pub summary: String,
    pub depth: u32,
    pub fts_score: f64,
}

// ---------------------------------------------------------------------------
// Index statistics
// ---------------------------------------------------------------------------

/// Statistics about the FTS5 index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub db_path: String,
    pub document_count: usize,
    pub node_count: usize,
}

// ---------------------------------------------------------------------------
// Markdown field parser
// ---------------------------------------------------------------------------

/// Parsed markdown-aware fields from a node's text.
#[derive(Debug, Clone, Default)]
pub struct MdFields {
    pub front_matter: String,
    pub body: String,
    pub code_blocks: String,
}

// Lazily-compiled regex singletons.
fn re_front_matter() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)^---\s*\n(.*?\n)---\s*\n").unwrap())
}

fn re_code_block() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)```[\w]*\n(.*?)```").unwrap())
}

fn re_blank_lines() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n{3,}").unwrap())
}

fn re_fts5_special() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\w\u{4e00}-\u{9fff}\u{3400}-\u{4dbf}]").unwrap())
}

/// Parse a node's text into MD-aware structured fields.
///
/// Extracts YAML front matter (between `---` fences), fenced code blocks,
/// and the remaining body text.
pub fn parse_md_node_text(text: &str) -> MdFields {
    if text.is_empty() {
        return MdFields::default();
    }

    let mut front_matter = String::new();
    let remaining;

    if let Some(cap) = re_front_matter().captures(text) {
        front_matter = cap[1].trim().to_string();
        remaining = &text[cap.get(0).unwrap().end()..];
    } else {
        remaining = text;
    }

    // Extract code blocks
    let mut code_parts: Vec<String> = Vec::new();
    let body = re_code_block().replace_all(remaining, |caps: &regex::Captures| {
        if let Some(code) = caps.get(1) {
            code_parts.push(code.as_str().trim().to_string());
        }
        String::new()
    });
    let code_blocks = code_parts.join("\n");

    // Collapse excessive blank lines
    let body = re_blank_lines()
        .replace_all(&body, "\n\n")
        .trim()
        .to_string();

    MdFields {
        front_matter,
        body,
        code_blocks,
    }
}

// ---------------------------------------------------------------------------
// Tokenizer bridge
// ---------------------------------------------------------------------------

/// Tokenize text for FTS5 indexing. Returns space-separated tokens.
///
/// Delegates to `crate::tokenizer::tokenize_for_fts` for CJK support.
fn tokenize_for_fts(text: &str, cjk_mode: CjkTokenizerMode) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    crate::tokenizer::tokenize_for_fts(text, cjk_mode)
}

/// FTS5 operators that must NOT be tokenized.
const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT", "NEAR"];

fn is_fts5_operator(word: &str) -> bool {
    FTS5_OPERATORS.contains(&word.to_uppercase().as_str())
}

/// Tokenize terms in an FTS5 expression while preserving operators.
fn tokenize_fts_expression(expr: &str, cjk_mode: CjkTokenizerMode) -> String {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    let mut result = Vec::new();
    for part in parts {
        if FTS5_OPERATORS.contains(&part.to_uppercase().as_str()) {
            result.push(part.to_uppercase());
        } else {
            let tokenized = tokenize_for_fts(part, cjk_mode);
            let trimmed = tokenized.trim().to_string();
            if !trimmed.is_empty() {
                result.push(trimmed);
            }
        }
    }
    result.join(" ")
}

// ---------------------------------------------------------------------------
// Content hashing
// ---------------------------------------------------------------------------

fn md5_hex(data: &[u8]) -> String {
    let hash = Md5::digest(data);
    format!("{:x}", hash)
}

fn content_hash_for_doc(doc: &Document) -> Result<String> {
    let json = serde_json::to_string(&doc.structure)?;
    Ok(md5_hex(json.as_bytes()))
}

// ---------------------------------------------------------------------------
// FTS5Index
// ---------------------------------------------------------------------------

/// SQLite FTS5 full-text search index for tree-structured documents.
///
/// - WAL journal mode for concurrent read/write
/// - MD-aware columns: title, summary, body, code_blocks, front_matter
/// - Hierarchical column weighting via bm25() rank function
/// - Content-hash incremental indexing
/// - Ancestor score propagation for tree search
///
/// In-memory mode (`db_path = None`): all indexes in `:memory:`, no file on disk.
pub struct FTS5Index {
    conn: Connection,
    db_path: String,
    weights: FtsWeights,
    cjk_tokenizer: CjkTokenizerMode,
    max_node_chars: usize,
}

impl FTS5Index {
    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    /// Create a new FTS5 index.
    ///
    /// - `db_path = None` → in-memory (`:memory:`).
    /// - `weights = None` → default column weights.
    pub fn new(db_path: Option<&str>, weights: Option<FtsWeights>) -> Result<Self> {
        Self::new_path(db_path.map(std::path::Path::new), weights)
    }

    /// Creates an index from an OS-native path without requiring UTF-8.
    pub fn new_path(
        db_path: Option<&std::path::Path>,
        weights: Option<FtsWeights>,
    ) -> Result<Self> {
        Self::open(
            db_path,
            weights.unwrap_or_default(),
            CjkTokenizerMode::Auto,
            TreeSearchConfig::default().max_node_chars,
        )
    }

    /// Creates an index using both FTS weights and tokenizer settings from config.
    pub fn with_config(db_path: Option<&str>, config: &TreeSearchConfig) -> Result<Self> {
        Self::with_config_path(db_path.map(std::path::Path::new), config)
    }

    /// Creates a configured index from an OS-native path without requiring UTF-8.
    pub fn with_config_path(
        db_path: Option<&std::path::Path>,
        config: &TreeSearchConfig,
    ) -> Result<Self> {
        crate::tokenizer::cjk::configure_from(config);
        Self::open(
            db_path,
            FtsWeights::from_config(config),
            config.cjk_tokenizer,
            config.max_node_chars,
        )
    }

    /// Opens SQLite after validating all ranking inputs.
    fn open(
        db_path: Option<&std::path::Path>,
        weights: FtsWeights,
        cjk_tokenizer: CjkTokenizerMode,
        max_node_chars: usize,
    ) -> Result<Self> {
        weights.validate()?;
        let path = db_path.unwrap_or_else(|| std::path::Path::new(":memory:"));

        if path != std::path::Path::new(":memory:") {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create dir for {}", path.display()))?;
                }
            }
        }

        let conn =
            Connection::open(path).with_context(|| format!("open SQLite at {}", path.display()))?;

        let mut index = Self {
            conn,
            db_path: path.to_string_lossy().into_owned(),
            weights,
            cjk_tokenizer,
            max_node_chars,
        };
        index.init_db()?;
        Ok(index)
    }

    /// Database path (or `:memory:`).
    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    // ---------------------------------------------------------------
    // Schema initialization
    // ---------------------------------------------------------------

    fn init_db(&mut self) -> Result<()> {
        self.conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        self.conn.execute_batch("PRAGMA synchronous=NORMAL;")?;

        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                depth INTEGER DEFAULT 0,
                line_start INTEGER,
                line_end INTEGER,
                parent_node_id TEXT,
                content_hash TEXT,
                PRIMARY KEY (doc_id, node_id)
            );",
        )?;

        self.conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(
                node_id UNINDEXED,
                doc_id UNINDEXED,
                title,
                summary,
                body,
                code_blocks,
                front_matter,
                tokenize='unicode61 remove_diacritics 2'
            );",
        )?;

        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_name TEXT DEFAULT '',
                doc_description TEXT DEFAULT '',
                source_path TEXT DEFAULT '',
                source_type TEXT DEFAULT '',
                structure_json TEXT DEFAULT '',
                node_count INTEGER DEFAULT 0,
                index_hash TEXT
            );",
        )?;

        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS index_meta (
                source_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL
            );",
        )?;

        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_nodes_doc_id ON nodes (doc_id);
             CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents (source_path);",
        )?;

        Ok(())
    }

    // ---------------------------------------------------------------
    // Indexing
    // ---------------------------------------------------------------

    /// Index all nodes from a Document into FTS5.
    ///
    /// Returns the number of nodes indexed (0 if content hash matches
    /// and `force` is false).
    pub fn index_document(&self, doc: &Document, force: bool) -> Result<usize> {
        doc.validate_structure()
            .with_context(|| format!("validate document `{}`", doc.doc_id))?;
        let content_hash = content_hash_for_doc(doc)?;

        // Incremental check
        if !force {
            let existing: Option<String> = self
                .conn
                .query_row(
                    "SELECT index_hash FROM documents WHERE doc_id = ?1",
                    params![doc.doc_id],
                    |row| row.get(0),
                )
                .ok();
            if existing.as_deref() == Some(content_hash.as_str()) {
                return Ok(0);
            }
        }

        let transaction = self.conn.unchecked_transaction()?;

        // Clear old entries inside the same transaction as replacement rows.
        delete_fts_rows_for_doc(&transaction, &doc.doc_id)?;
        transaction.execute("DELETE FROM nodes WHERE doc_id = ?1", params![doc.doc_id])?;

        let parent_map = doc.build_parent_map();
        let depth_map = doc.build_depth_map();
        let all_nodes = doc.flatten_nodes();
        let mut count = 0usize;

        for node in &all_nodes {
            if node.node_id.is_empty() {
                continue;
            }
            let depth = depth_map.get(&node.node_id).copied().unwrap_or(0);
            let parent_id = parent_map
                .get(&node.node_id)
                .and_then(|p| p.as_deref())
                .unwrap_or("");
            let node_hash = &md5_hex(node.text.as_bytes())[..16];

            transaction.execute(
                "INSERT OR REPLACE INTO nodes
                 (node_id, doc_id, title, summary, depth, line_start, line_end, parent_node_id, content_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    node.node_id,
                    doc.doc_id,
                    node.title,
                    node.summary,
                    depth,
                    node.line_start,
                    node.line_end,
                    parent_id,
                    node_hash,
                ],
            )?;

            let parsed = parse_md_node_text(bounded_prefix(&node.text, self.max_node_chars));
            let title_tok = tokenize_for_fts(
                bounded_prefix(&node.title, self.max_node_chars),
                self.cjk_tokenizer,
            );
            let summary_tok = tokenize_for_fts(
                bounded_prefix(&node.summary, self.max_node_chars),
                self.cjk_tokenizer,
            );
            let body_tok = tokenize_for_fts(&parsed.body, self.cjk_tokenizer);
            let code_tok = tokenize_for_fts(&parsed.code_blocks, self.cjk_tokenizer);
            let fm_tok = tokenize_for_fts(&parsed.front_matter, self.cjk_tokenizer);

            transaction.execute(
                "INSERT INTO fts_nodes
                 (node_id, doc_id, title, summary, body, code_blocks, front_matter)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    node.node_id,
                    doc.doc_id,
                    title_tok,
                    summary_tok,
                    body_tok,
                    code_tok,
                    fm_tok,
                ],
            )?;
            count += 1;
        }

        // Upsert document record
        let structure_json = serde_json::to_string(&doc.structure)?;
        transaction.execute(
            "INSERT OR REPLACE INTO documents
             (doc_id, doc_name, doc_description, source_path, source_type,
              structure_json, node_count, index_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                doc.doc_id,
                doc.doc_name,
                doc.doc_description,
                doc.source_path,
                doc.source_type.as_str(),
                structure_json,
                count as i64,
                content_hash,
            ],
        )?;

        transaction.commit()?;
        Ok(count)
    }

    /// Batch index multiple documents. Returns total nodes indexed.
    pub fn index_documents(&self, documents: &[Document], force: bool) -> Result<usize> {
        let mut total = 0;
        for doc in documents {
            total += self.index_document(doc, force)?;
        }
        Ok(total)
    }

    /// Manually commit pending changes to the database.
    pub fn commit(&self) -> Result<()> {
        self.conn.execute_batch("-- noop; rusqlite auto-commits")?;
        Ok(())
    }

    /// Batch store/update file hashes. Single transaction for performance.
    pub fn set_index_meta_batch(&self, meta: &HashMap<String, String>) -> Result<()> {
        let transaction = self.conn.unchecked_transaction()?;
        for (path, hash) in meta {
            transaction.execute(
                "INSERT OR REPLACE INTO index_meta (source_path, file_hash) VALUES (?1, ?2)",
                params![path, hash],
            )?;
        }
        transaction.commit()?;
        Ok(())
    }

    /// Look up a doc_id from a source file path.
    pub fn get_doc_id_by_source_path(&self, source_path: &str) -> Result<Option<String>> {
        match self.conn.query_row(
            "SELECT doc_id FROM documents WHERE source_path = ?1",
            params![source_path],
            |row| row.get(0),
        ) {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ---------------------------------------------------------------
    // Search
    // ---------------------------------------------------------------

    /// Build an FTS5 MATCH expression from a query string.
    /// Returns `None` if no valid tokens could be extracted.
    fn build_match_expr(&self, query: &str, fts_expression: Option<&str>) -> Option<String> {
        if let Some(expr) = fts_expression {
            let tokenized = tokenize_fts_expression(expr, self.cjk_tokenizer);
            if tokenized.trim().is_empty() {
                return None;
            }
            return Some(tokenized);
        }

        let tokens = tokenize_for_fts(query, self.cjk_tokenizer);
        if tokens.trim().is_empty() {
            return None;
        }
        let re = re_fts5_special();
        let clean_words: Vec<String> = tokens
            .split_whitespace()
            .filter_map(|w| {
                let cleaned = re.replace_all(w, "").trim().to_string();
                if cleaned.is_empty() || is_fts5_operator(&cleaned) {
                    None
                } else {
                    Some(cleaned)
                }
            })
            .collect();

        if clean_words.is_empty() {
            return None;
        }
        if clean_words.len() > 1 {
            Some(clean_words.join(" OR "))
        } else {
            Some(clean_words.into_iter().next().unwrap())
        }
    }

    /// Search nodes using FTS5 BM25 ranking.
    ///
    /// Returns results sorted by relevance (highest score first).
    pub fn search(
        &self,
        query: &str,
        doc_id: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<FtsResult>> {
        self.search_with_expr(query, doc_id, top_k, None)
    }

    pub fn search_with_expr(
        &self,
        query: &str,
        doc_id: Option<&str>,
        top_k: usize,
        fts_expression: Option<&str>,
    ) -> Result<Vec<FtsResult>> {
        let match_expr = match self.build_match_expr(query, fts_expression) {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let weight_args = self.weights.bm25_args();

        // Phase 1: phrase boosting for multi-word queries
        let phrase_boost_nids = self.collect_phrase_boost_nids(query, doc_id);

        // Phase 2: main FTS5 MATCH query
        struct RawRow {
            node_id: String,
            doc_id: String,
            title: String,
            summary: String,
            rank_score: f64,
        }

        let raw_rows: Vec<RawRow> = if let Some(did) = doc_id {
            let sql = format!(
                "SELECT f.node_id, f.doc_id, f.title, f.summary,
                        bm25(fts_nodes, {}) AS rank_score
                 FROM fts_nodes f
                 WHERE fts_nodes MATCH ?1 AND f.doc_id = ?2
                 ORDER BY rank_score LIMIT ?3",
                weight_args
            );
            let mut stmt = match self.conn.prepare(&sql) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("FTS5 query error: {}, query={:?}", e, match_expr);
                    return Ok(Vec::new());
                }
            };
            let rows = stmt.query_map(params![match_expr, did, top_k as i64], |row| {
                Ok(RawRow {
                    node_id: row.get(0)?,
                    doc_id: row.get(1)?,
                    title: row.get(2)?,
                    summary: row.get(3)?,
                    rank_score: row.get(4)?,
                })
            })?;
            rows.filter_map(|r| r.ok()).collect()
        } else {
            let sql = format!(
                "SELECT f.node_id, f.doc_id, f.title, f.summary,
                        bm25(fts_nodes, {}) AS rank_score
                 FROM fts_nodes f
                 WHERE fts_nodes MATCH ?1
                 ORDER BY rank_score LIMIT ?2",
                weight_args
            );
            let mut stmt = match self.conn.prepare(&sql) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("FTS5 query error: {}, query={:?}", e, match_expr);
                    return Ok(Vec::new());
                }
            };
            let rows = stmt.query_map(params![match_expr, top_k as i64], |row| {
                Ok(RawRow {
                    node_id: row.get(0)?,
                    doc_id: row.get(1)?,
                    title: row.get(2)?,
                    summary: row.get(3)?,
                    rank_score: row.get(4)?,
                })
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        // Batch-fetch node metadata (depth, canonical title/summary)
        let keys: Vec<(String, String)> = raw_rows
            .iter()
            .map(|r| (r.node_id.clone(), r.doc_id.clone()))
            .collect();
        let node_meta = self.batch_lookup_node_meta(&keys)?;

        // Deduplicate and assemble results
        let mut results: Vec<FtsResult> = Vec::new();
        let mut seen: HashMap<(String, String), usize> = HashMap::new();

        for raw in &raw_rows {
            let mut fts_score = if raw.rank_score != 0.0 {
                -raw.rank_score
            } else {
                0.0
            };
            let identity = (raw.doc_id.clone(), raw.node_id.clone());
            if phrase_boost_nids.contains(&identity) {
                fts_score *= 1.5;
            }

            if let Some(&idx) = seen.get(&identity) {
                if fts_score > results[idx].fts_score {
                    results[idx].fts_score = round6(fts_score);
                }
                continue;
            }

            let meta = node_meta.get(&(raw.node_id.clone(), raw.doc_id.clone()));
            seen.insert(identity, results.len());
            results.push(FtsResult {
                node_id: raw.node_id.clone(),
                doc_id: raw.doc_id.clone(),
                title: meta.map_or_else(|| raw.title.clone(), |m| m.0.clone()),
                summary: meta.map_or_else(|| raw.summary.clone(), |m| m.1.clone()),
                depth: meta.map_or(0, |m| m.2),
                fts_score: round6(fts_score),
            });
        }

        results.sort_by(|left, right| {
            right
                .fts_score
                .total_cmp(&left.fts_score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
                .then_with(|| left.node_id.cmp(&right.node_id))
        });

        Ok(results)
    }

    /// Collect node IDs matching the exact phrase for score boosting.
    fn collect_phrase_boost_nids(
        &self,
        query: &str,
        doc_id: Option<&str>,
    ) -> std::collections::HashSet<(String, String)> {
        let mut nids = std::collections::HashSet::new();
        let words: Vec<&str> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .collect();
        if words.len() < 2 {
            return nids;
        }
        let phrase_expr = format!("\"{}\"", words.join(" ").to_lowercase());

        let result: std::result::Result<Vec<(String, String)>, _> = if let Some(did) = doc_id {
            self.conn
                .prepare(
                    "SELECT f.node_id, f.doc_id FROM fts_nodes f \
                     WHERE fts_nodes MATCH ?1 AND f.doc_id = ?2 LIMIT 50",
                )
                .and_then(|mut stmt| {
                    let rows: Vec<(String, String)> = stmt
                        .query_map(params![phrase_expr, did], |row| {
                            Ok((row.get(1)?, row.get(0)?))
                        })?
                        .filter_map(|r| r.ok())
                        .collect();
                    Ok(rows)
                })
        } else {
            self.conn
                .prepare(
                    "SELECT f.node_id, f.doc_id FROM fts_nodes f \
                     WHERE fts_nodes MATCH ?1 LIMIT 50",
                )
                .and_then(|mut stmt| {
                    let rows: Vec<(String, String)> = stmt
                        .query_map(params![phrase_expr], |row| Ok((row.get(1)?, row.get(0)?)))?
                        .filter_map(|r| r.ok())
                        .collect();
                    Ok(rows)
                })
        };

        if let Ok(rows) = result {
            nids.extend(rows);
        }
        nids
    }

    /// Batch lookup node metadata (title, summary, depth) from the `nodes` table.
    fn batch_lookup_node_meta(&self, keys: &[(String, String)]) -> Result<NodeMetadataMap> {
        let mut meta = HashMap::new();
        for (nid, did) in keys {
            let key = (nid.clone(), did.clone());
            if meta.contains_key(&key) {
                continue;
            }
            let result = self.conn.query_row(
                "SELECT title, summary, depth FROM nodes WHERE node_id = ?1 AND doc_id = ?2",
                params![nid, did],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, u32>(2)?,
                    ))
                },
            );
            if let Ok(val) = result {
                meta.insert(key, val);
            }
        }
        Ok(meta)
    }

    // ---------------------------------------------------------------
    // Batch scoring (tree search integration)
    // ---------------------------------------------------------------

    /// Batch scoring with ancestor propagation.
    ///
    /// Returns `{doc_id: {node_id: score}}` for all matched documents.
    /// Scores are normalized to `[0, 1]` across the whole candidate set, so
    /// cross-document relevance is preserved before tree reranking.
    pub fn score_nodes_batch(
        &self,
        query: &str,
        doc_ids: Option<&[String]>,
        ancestor_decay: f64,
    ) -> Result<HashMap<String, HashMap<String, f64>>> {
        self.score_nodes_batch_with_expr(query, doc_ids, ancestor_decay, None)
    }

    pub fn score_nodes_batch_with_expr(
        &self,
        query: &str,
        doc_ids: Option<&[String]>,
        ancestor_decay: f64,
        fts_expression: Option<&str>,
    ) -> Result<HashMap<String, HashMap<String, f64>>> {
        let ancestor_decay = if ancestor_decay.is_finite() {
            ancestor_decay.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let match_expr = match self.build_match_expr(query, fts_expression) {
            Some(e) => e,
            None => return Ok(HashMap::new()),
        };

        let weight_args = self.weights.bm25_args();
        let phrase_boost_nodes = self.collect_phrase_boost_nids(query, None);

        // Build SQL depending on whether doc_ids filter is given.
        let (sql, extra_params): (String, Vec<String>) = if let Some(ids) = doc_ids {
            if ids.is_empty() {
                return Ok(HashMap::new());
            }
            let placeholders: String = (0..ids.len())
                .map(|i| format!("?{}", i + 2))
                .collect::<Vec<_>>()
                .join(",");
            (
                format!(
                    "SELECT f.node_id, f.doc_id,
                                bm25(fts_nodes, {w}) AS rank_score
                         FROM fts_nodes f
                         WHERE fts_nodes MATCH ?1
                           AND f.doc_id IN ({p})
                         ORDER BY rank_score LIMIT {limit}",
                    w = weight_args,
                    p = placeholders,
                    limit = DEFAULT_BATCH_CANDIDATE_LIMIT,
                ),
                ids.to_vec(),
            )
        } else {
            (
                format!(
                    "SELECT f.node_id, f.doc_id,
                                bm25(fts_nodes, {w}) AS rank_score
                         FROM fts_nodes f
                         WHERE fts_nodes MATCH ?1
                         ORDER BY rank_score LIMIT {limit}",
                    w = weight_args,
                    limit = DEFAULT_BATCH_CANDIDATE_LIMIT,
                ),
                Vec::new(),
            )
        };

        let mut stmt = match self.conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return Ok(HashMap::new()),
        };

        stmt.raw_bind_parameter(1, match_expr.as_str())?;
        for (i, id) in extra_params.iter().enumerate() {
            stmt.raw_bind_parameter(i + 2, id.as_str())?;
        }

        // Collect raw scores grouped by doc_id.
        let mut per_doc_raw: HashMap<String, HashMap<String, f64>> = HashMap::new();
        {
            let mut rows = stmt.raw_query();
            while let Some(row) = rows.next()? {
                let node_id: String = row.get(0)?;
                let doc_id: String = row.get(1)?;
                let rank_score: f64 = row.get(2)?;
                let mut fts_score = if rank_score != 0.0 { -rank_score } else { 0.0 };
                if phrase_boost_nodes.contains(&(doc_id.clone(), node_id.clone())) {
                    fts_score *= 1.5;
                }
                let entry = per_doc_raw.entry(doc_id).or_default();
                let old = entry.get(&node_id).copied().unwrap_or(0.0);
                entry.insert(node_id, old.max(fts_score));
            }
        }
        drop(stmt);

        if per_doc_raw.is_empty() {
            return Ok(HashMap::new());
        }

        // Fetch parent→children maps for ancestor propagation.
        let doc_children_map = if ancestor_decay > 0.0 {
            self.fetch_children_maps(&per_doc_raw.keys().cloned().collect::<Vec<_>>())?
        } else {
            HashMap::new()
        };

        // Normalize globally before ancestor propagation so weaker documents do
        // not each receive an artificial score of 1.0.
        let global_max = per_doc_raw
            .values()
            .flat_map(HashMap::values)
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1e-10);
        let mut result: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for (doc_id, raw_scores) in &per_doc_raw {
            let mut scores: HashMap<String, f64> = raw_scores
                .iter()
                .map(|(nid, &score)| (nid.clone(), score / global_max))
                .collect();

            if ancestor_decay > 0.0 {
                if let Some(children_map) = doc_children_map.get(doc_id) {
                    scores = propagate_ancestor_scores(&scores, children_map, ancestor_decay);
                }
            }
            result.insert(doc_id.clone(), scores);
        }

        let propagated_max = result
            .values()
            .flat_map(HashMap::values)
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1e-10);
        for scores in result.values_mut() {
            for score in scores.values_mut() {
                if propagated_max > 1.0 {
                    *score /= propagated_max;
                }
                *score = round6(*score);
            }
        }

        Ok(result)
    }

    /// Fetch parent→children maps from the `nodes` table.
    fn fetch_children_maps(
        &self,
        doc_ids: &[String],
    ) -> Result<HashMap<String, HashMap<String, Vec<String>>>> {
        if doc_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..doc_ids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT doc_id, node_id, parent_node_id FROM nodes WHERE doc_id IN ({})",
            placeholders
        );
        let mut stmt = self.conn.prepare(&sql)?;
        for (i, id) in doc_ids.iter().enumerate() {
            stmt.raw_bind_parameter(i + 1, id.as_str())?;
        }

        let mut map: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
        let mut rows = stmt.raw_query();
        while let Some(row) = rows.next()? {
            let did: String = row.get(0)?;
            let nid: String = row.get(1)?;
            let pid: String = row.get(2)?;
            if !pid.is_empty() {
                map.entry(did)
                    .or_default()
                    .entry(pid)
                    .or_default()
                    .push(nid);
            }
        }
        Ok(map)
    }

    // ---------------------------------------------------------------
    // Document persistence
    // ---------------------------------------------------------------

    /// Save a Document's tree structure (without FTS indexing).
    pub fn save_document(&self, doc: &Document) -> Result<()> {
        doc.validate_structure()
            .with_context(|| format!("validate document `{}`", doc.doc_id))?;
        let structure_json = serde_json::to_string(&doc.structure)?;
        let content_hash = md5_hex(structure_json.as_bytes());
        let node_count = doc.flatten_nodes().len();

        self.conn.execute(
            "INSERT OR REPLACE INTO documents
             (doc_id, doc_name, doc_description, source_path, source_type,
              structure_json, node_count, index_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                doc.doc_id,
                doc.doc_name,
                doc.doc_description,
                doc.source_path,
                doc.source_type.as_str(),
                structure_json,
                node_count as i64,
                content_hash,
            ],
        )?;
        Ok(())
    }

    /// Load a single Document by doc_id. Returns `None` if not found.
    pub fn load_document(&self, doc_id: &str) -> Result<Option<Document>> {
        let result = self.conn.query_row(
            "SELECT doc_id, doc_name, doc_description, source_path, source_type, structure_json
             FROM documents WHERE doc_id = ?1",
            params![doc_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        );

        match result {
            Ok((did, dname, ddesc, spath, stype, sjson)) => {
                let structure: Vec<Node> = if sjson.is_empty() {
                    Vec::new()
                } else {
                    serde_json::from_str(&sjson)
                        .with_context(|| format!("decode stored structure for document `{did}`"))?
                };
                let source_type = source_type_from_str(&stype);
                let mut doc = Document::new(did, dname, source_type);
                doc.doc_description = ddesc;
                doc.source_path = spath;
                doc.structure = structure;
                doc.validate_structure()
                    .with_context(|| format!("validate stored document `{}`", doc.doc_id))?;
                Ok(Some(doc))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Load all Documents stored in the DB.
    pub fn load_all_documents(&self) -> Result<Vec<Document>> {
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, doc_name, doc_description, source_path, source_type, structure_json
             FROM documents ORDER BY doc_id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;

        let mut documents = Vec::new();
        for r in rows {
            let (did, dname, ddesc, spath, stype, sjson) = r?;
            let structure: Vec<Node> = if sjson.is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(&sjson)
                    .with_context(|| format!("decode stored structure for document `{did}`"))?
            };
            let source_type = source_type_from_str(&stype);
            let mut doc = Document::new(did, dname, source_type);
            doc.doc_description = ddesc;
            doc.source_path = spath;
            doc.structure = structure;
            doc.validate_structure()
                .with_context(|| format!("validate stored document `{}`", doc.doc_id))?;
            documents.push(doc);
        }
        Ok(documents)
    }

    /// Delete a document and all its indexed data atomically.
    ///
    /// Returns `true` if the document existed and was deleted.
    pub fn delete_document(&self, doc_id: &str) -> Result<bool> {
        let source_path: Option<String> = self
            .conn
            .query_row(
                "SELECT source_path FROM documents WHERE doc_id = ?1",
                params![doc_id],
                |row| row.get(0),
            )
            .ok();

        if source_path.is_none() {
            return Ok(false);
        }
        let source_path = source_path.unwrap();

        let transaction = self.conn.unchecked_transaction()?;
        delete_fts_rows_for_doc(&transaction, doc_id)?;
        transaction.execute("DELETE FROM nodes WHERE doc_id = ?1", params![doc_id])?;
        transaction.execute("DELETE FROM documents WHERE doc_id = ?1", params![doc_id])?;
        if !source_path.is_empty() {
            transaction.execute(
                "DELETE FROM index_meta WHERE source_path = ?1",
                params![source_path],
            )?;
        }
        transaction.commit()?;

        Ok(true)
    }

    // ---------------------------------------------------------------
    // Index metadata (fingerprint management)
    // ---------------------------------------------------------------

    /// Get the stored file hash for a source path.
    pub fn get_index_meta(&self, source_path: &str) -> Result<Option<String>> {
        match self.conn.query_row(
            "SELECT file_hash FROM index_meta WHERE source_path = ?1",
            params![source_path],
            |row| row.get(0),
        ) {
            Ok(hash) => Ok(Some(hash)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Store/update the file hash for a source path.
    pub fn set_index_meta(&self, source_path: &str, file_hash: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO index_meta (source_path, file_hash) VALUES (?1, ?2)",
            params![source_path, file_hash],
        )?;
        Ok(())
    }

    /// Get all stored file hashes: `{source_path: file_hash}`.
    pub fn get_all_index_meta(&self) -> Result<HashMap<String, String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT source_path, file_hash FROM index_meta")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut map = HashMap::new();
        for r in rows {
            let (path, hash) = r?;
            map.insert(path, hash);
        }
        Ok(map)
    }

    // ---------------------------------------------------------------
    // Statistics & maintenance
    // ---------------------------------------------------------------

    /// Get index statistics.
    pub fn get_stats(&self) -> Result<IndexStats> {
        let doc_count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        let node_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))?;
        Ok(IndexStats {
            db_path: self.db_path.clone(),
            document_count: doc_count,
            node_count,
        })
    }

    /// Run FTS5 merge optimization.
    pub fn optimize(&self) -> Result<()> {
        self.conn
            .execute("INSERT INTO fts_nodes(fts_nodes) VALUES('optimize')", [])?;
        Ok(())
    }

    /// Rebuild FTS5 index from scratch.
    pub fn rebuild(&self) -> Result<()> {
        self.conn
            .execute("INSERT INTO fts_nodes(fts_nodes) VALUES('rebuild')", [])?;
        Ok(())
    }

    /// Clear all indexed data.
    pub fn clear(&self) -> Result<()> {
        let transaction = self.conn.unchecked_transaction()?;
        transaction.execute("DELETE FROM fts_nodes", [])?;
        transaction.execute("DELETE FROM nodes", [])?;
        transaction.execute("DELETE FROM documents", [])?;
        transaction.execute("DELETE FROM index_meta", [])?;
        transaction.commit()?;
        Ok(())
    }

    /// Close the database connection.
    ///
    /// Consumes the index. After calling `close()`, no further operations
    /// are possible. SQLite flushes WAL and releases all locks.
    pub fn close(self) {
        // `Connection::close` returns Result<(), (Connection, Error)>.
        // On error we just drop the connection (best-effort).
        let _ = self.conn.close();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deletes every FTS row for one document using the caller's transaction.
fn delete_fts_rows_for_doc(connection: &Connection, doc_id: &str) -> Result<()> {
    let rowids: Vec<i64> = {
        let mut statement = connection.prepare("SELECT rowid FROM fts_nodes WHERE doc_id = ?1")?;
        let rows = statement
            .query_map(params![doc_id], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    if rowids.is_empty() {
        return Ok(());
    }
    let placeholders = rowids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!("DELETE FROM fts_nodes WHERE rowid IN ({placeholders})");
    let mut statement = connection.prepare(&sql)?;
    for (index, row_id) in rowids.iter().enumerate() {
        statement.raw_bind_parameter(index + 1, *row_id)?;
    }
    statement.raw_execute()?;
    Ok(())
}

/// Returns an indexing prefix without splitting UTF-8 code points.
fn bounded_prefix(text: &str, max_characters: usize) -> &str {
    match text.char_indices().nth(max_characters) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// Propagates child relevance to every ancestor in deterministic bottom-up order.
fn propagate_ancestor_scores(
    base_scores: &HashMap<String, f64>,
    children_map: &HashMap<String, Vec<String>>,
    ancestor_decay: f64,
) -> HashMap<String, f64> {
    let mut node_ids: Vec<String> = base_scores
        .keys()
        .chain(children_map.keys())
        .chain(children_map.values().flatten())
        .cloned()
        .collect();
    node_ids.sort();
    node_ids.dedup();

    let mut memo = HashMap::new();
    let mut visit_state = HashMap::<String, u8>::new();
    for root_id in node_ids {
        if memo.contains_key(&root_id) {
            continue;
        }
        let mut stack = vec![(root_id, false)];
        while let Some((node_id, exiting)) = stack.pop() {
            if exiting {
                let max_child = children_map
                    .get(&node_id)
                    .into_iter()
                    .flatten()
                    .map(|child| {
                        memo.get(child)
                            .copied()
                            .unwrap_or_else(|| base_scores.get(child).copied().unwrap_or(0.0))
                    })
                    .fold(0.0_f64, f64::max);
                let base = base_scores.get(&node_id).copied().unwrap_or(0.0);
                memo.insert(node_id.clone(), base + ancestor_decay * max_child);
                visit_state.insert(node_id, 2);
                continue;
            }
            if visit_state.get(&node_id) == Some(&2) {
                continue;
            }
            visit_state.insert(node_id.clone(), 1);
            stack.push((node_id.clone(), true));
            let mut children = children_map.get(&node_id).cloned().unwrap_or_default();
            children.sort();
            children.dedup();
            for child in children.into_iter().rev() {
                if !visit_state.contains_key(&child) {
                    stack.push((child, false));
                }
            }
        }
    }
    memo.retain(|_, score| score.is_finite() && *score > 0.0);
    memo
}

fn source_type_from_str(s: &str) -> SourceType {
    match s {
        "code" => SourceType::Code,
        "markdown" => SourceType::Markdown,
        "html" => SourceType::Html,
        "text" => SourceType::Text,
        "json" => SourceType::Json,
        "yaml" => SourceType::Yaml,
        "toml" => SourceType::Toml,
        _ => SourceType::Unknown,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, Node, SourceType};

    fn sample_doc() -> Document {
        let mut doc = Document::new("doc1", "test_doc.md", SourceType::Markdown);
        doc.source_path = "/tmp/test_doc.md".to_string();
        doc.doc_description = "A test document".to_string();

        let mut root = Node::new("0", "Introduction");
        root.text = "This is the introduction to machine learning.".to_string();
        root.summary = "Intro to ML".to_string();

        let mut child1 = Node::new("1", "Deep Learning");
        child1.text =
            "Deep learning is a subset of machine learning using neural networks.".to_string();
        child1.summary = "DL overview".to_string();

        let mut child2 = Node::new("2", "Reinforcement Learning");
        child2.text = "Reinforcement learning trains agents through reward signals.".to_string();
        child2.summary = "RL overview".to_string();

        let mut grandchild = Node::new("3", "Transformers");
        grandchild.text =
            "Transformers use self-attention mechanisms for sequence modeling.".to_string();
        grandchild.summary = "Transformer architecture".to_string();
        child1.children.push(grandchild);

        root.children.push(child1);
        root.children.push(child2);
        doc.structure.push(root);
        doc
    }

    #[test]
    fn test_ancestor_propagation_is_stable_across_insertion_orders() {
        let base_scores = HashMap::from([("leaf".to_string(), 1.0)]);
        let mut parent_first = HashMap::new();
        parent_first.insert("root".to_string(), vec!["section".to_string()]);
        parent_first.insert("section".to_string(), vec!["leaf".to_string()]);
        let mut child_first = HashMap::new();
        child_first.insert("section".to_string(), vec!["leaf".to_string()]);
        child_first.insert("root".to_string(), vec!["section".to_string()]);

        let first = propagate_ancestor_scores(&base_scores, &parent_first, 0.6);
        let second = propagate_ancestor_scores(&base_scores, &child_first, 0.6);

        assert_eq!(first, second);
        assert_eq!(first.get("leaf"), Some(&1.0));
        assert_eq!(first.get("section"), Some(&0.6));
        assert_eq!(first.get("root"), Some(&0.36));
    }

    #[test]
    fn test_batch_scoring_reaches_all_ancestors_deterministically() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.index_document(&doc, false).unwrap();

        for _ in 0..10 {
            let scores = index.score_nodes_batch("transformers", None, 0.6).unwrap();
            let doc_scores = scores.get("doc1").expect("document scores");
            assert_eq!(doc_scores.get("3"), Some(&1.0));
            assert_eq!(doc_scores.get("1"), Some(&0.6));
            assert_eq!(doc_scores.get("0"), Some(&0.36));
        }
    }

    #[test]
    fn test_ancestor_propagation_is_bounded_for_cycles_and_deep_trees() {
        let cycle = HashMap::from([
            ("a".to_string(), vec!["b".to_string()]),
            ("b".to_string(), vec!["a".to_string()]),
        ]);
        let cycle_scores =
            propagate_ancestor_scores(&HashMap::from([("a".to_string(), 1.0)]), &cycle, 0.6);
        assert!(cycle_scores.values().all(|score| score.is_finite()));
        assert!(cycle_scores.values().all(|score| *score <= 1.6));

        let mut deep = HashMap::new();
        for index in 0..4095 {
            deep.insert(index.to_string(), vec![(index + 1).to_string()]);
        }
        let scores =
            propagate_ancestor_scores(&HashMap::from([("4095".to_string(), 1.0)]), &deep, 0.6);
        assert_eq!(scores.get("4095"), Some(&1.0));
        assert!(scores.get("0").is_some_and(|score| score.is_finite()));
    }

    #[test]
    fn test_create_and_search() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();

        let count = index.index_document(&doc, false).unwrap();
        assert_eq!(count, 4);

        let results = index.search("machine learning", None, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.doc_id == "doc1"));

        let results = index.search("transformers", Some("doc1"), 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.node_id == "3"));

        let results = index.search("quantum computing", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_keeps_same_node_id_from_different_documents_distinct() {
        let index = FTS5Index::new(None, None).unwrap();
        let mut first = Document::new("a", "A", SourceType::Text);
        let mut first_node = Node::new("0", "Exact phrase");
        first_node.text = "alpha beta appears together".into();
        first.structure.push(first_node);
        let mut second = Document::new("b", "B", SourceType::Text);
        let mut second_node = Node::new("0", "Separated terms");
        second_node.text = "alpha filler filler beta".into();
        second.structure.push(second_node);
        index.index_document(&first, false).unwrap();
        index.index_document(&second, false).unwrap();

        let results = index.search("alpha beta", None, 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results
                .iter()
                .map(|result| (result.doc_id.as_str(), result.node_id.as_str()))
                .collect::<Vec<_>>(),
            vec![("a", "0"), ("b", "0")]
        );
        let phrase_boost = index.collect_phrase_boost_nids("alpha beta", None);
        assert!(phrase_boost.contains(&("a".to_string(), "0".to_string())));
        assert!(!phrase_boost.contains(&("b".to_string(), "0".to_string())));
    }

    #[test]
    fn batch_scores_preserve_cross_document_calibration() {
        let index = FTS5Index::new(None, None).unwrap();
        let mut strong = Document::new("strong", "Strong", SourceType::Text);
        let mut strong_node = Node::new("0", "target target target");
        strong_node.text = "target target target target".into();
        strong.structure.push(strong_node);
        let mut weak = Document::new("weak", "Weak", SourceType::Text);
        let mut weak_node = Node::new("0", "Other");
        weak_node.text = "target with a much longer unrelated body of filler words".into();
        weak.structure.push(weak_node);
        index.index_document(&strong, false).unwrap();
        index.index_document(&weak, false).unwrap();

        let scores = index.score_nodes_batch("target", None, 0.0).unwrap();
        let strong_score = scores["strong"]["0"];
        let weak_score = scores["weak"]["0"];
        assert_eq!(strong_score.max(weak_score), 1.0);
        assert_ne!(strong_score, weak_score);
        assert!(strong_score > weak_score);
    }

    #[test]
    fn test_incremental_indexing() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();

        let count1 = index.index_document(&doc, false).unwrap();
        assert_eq!(count1, 4);

        // Hash match → skip
        let count2 = index.index_document(&doc, false).unwrap();
        assert_eq!(count2, 0);

        // Force re-index
        let count3 = index.index_document(&doc, true).unwrap();
        assert_eq!(count3, 4);
    }

    #[test]
    fn invalid_weights_and_documents_fail_before_replacing_valid_data() {
        let invalid_weights = FtsWeights {
            body: f64::NAN,
            ..FtsWeights::default()
        };
        assert!(FTS5Index::new(None, Some(invalid_weights)).is_err());

        let index = FTS5Index::new(None, None).unwrap();
        let valid = sample_doc();
        index.index_document(&valid, false).unwrap();
        let mut invalid = valid.clone();
        invalid.structure[0].children[1].node_id = "1".into();
        assert!(index.index_document(&invalid, true).is_err());

        let results = index.search("transformers", Some("doc1"), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "3");
    }

    #[test]
    fn configured_indexing_bounds_node_text_and_honors_cjk_mode() {
        let config = TreeSearchConfig {
            max_node_chars: 3,
            cjk_tokenizer: CjkTokenizerMode::Bigram,
            ..TreeSearchConfig::default()
        };
        let index = FTS5Index::with_config(None, &config).unwrap();
        let mut document = Document::new("bounded", "Bounded", SourceType::Text);
        let mut node = Node::new("0", "Section");
        node.text = "知识库目标不会进入索引".into();
        document.structure.push(node);
        index.index_document(&document, false).unwrap();

        assert!(!index.search("知识", None, 10).unwrap().is_empty());
        assert!(index.search("目标", None, 10).unwrap().is_empty());
    }

    #[test]
    fn test_delete_document() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.index_document(&doc, false).unwrap();

        let stats = index.get_stats().unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.node_count, 4);

        let deleted = index.delete_document("doc1").unwrap();
        assert!(deleted);

        let stats = index.get_stats().unwrap();
        assert_eq!(stats.document_count, 0);
        assert_eq!(stats.node_count, 0);

        // Idempotent
        let deleted = index.delete_document("doc1").unwrap();
        assert!(!deleted);

        let results = index.search("machine learning", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_score_nodes_batch() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.index_document(&doc, false).unwrap();

        let scores = index
            .score_nodes_batch("deep learning neural", Some(&["doc1".to_string()]), 0.6)
            .unwrap();

        assert!(scores.contains_key("doc1"));
        let doc_scores = &scores["doc1"];
        assert!(doc_scores.contains_key("1"));
        assert!(*doc_scores.get("1").unwrap() > 0.0);

        // Ancestor propagation: parent "0" should inherit score from child "1"
        assert!(
            doc_scores.contains_key("0"),
            "Parent node should receive propagated score"
        );
    }

    #[test]
    fn test_document_persistence() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.save_document(&doc).unwrap();

        let loaded = index.load_document("doc1").unwrap().unwrap();
        assert_eq!(loaded.doc_id, "doc1");
        assert_eq!(loaded.doc_name, "test_doc.md");
        assert_eq!(loaded.source_type, SourceType::Markdown);
        assert_eq!(loaded.structure.len(), 1);
        assert_eq!(loaded.structure[0].children.len(), 2);

        let all = index.load_all_documents().unwrap();
        assert_eq!(all.len(), 1);

        assert!(index.load_document("nonexistent").unwrap().is_none());
    }

    #[test]
    fn corrupt_persisted_structure_is_reported_instead_of_hidden() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.save_document(&doc).unwrap();
        index
            .conn
            .execute(
                "UPDATE documents SET structure_json = ?1 WHERE doc_id = ?2",
                params!["{not-json", "doc1"],
            )
            .unwrap();

        let error = index.load_document("doc1").unwrap_err();
        assert!(error.to_string().contains("decode stored structure"));
        assert!(index.load_all_documents().is_err());
    }

    #[test]
    fn test_index_meta() {
        let index = FTS5Index::new(None, None).unwrap();

        assert!(index.get_index_meta("/foo.md").unwrap().is_none());
        assert!(index.get_all_index_meta().unwrap().is_empty());

        index.set_index_meta("/foo.md", "abc123").unwrap();
        assert_eq!(
            index.get_index_meta("/foo.md").unwrap().as_deref(),
            Some("abc123")
        );

        index.set_index_meta("/foo.md", "def456").unwrap();
        assert_eq!(
            index.get_index_meta("/foo.md").unwrap().as_deref(),
            Some("def456")
        );

        index.set_index_meta("/bar.md", "ghi789").unwrap();
        let all = index.get_all_index_meta().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all["/foo.md"], "def456");
        assert_eq!(all["/bar.md"], "ghi789");
    }

    #[test]
    fn test_stats_and_clear() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.index_document(&doc, false).unwrap();

        let stats = index.get_stats().unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.node_count, 4);
        assert_eq!(stats.db_path, ":memory:");

        index.clear().unwrap();
        let stats = index.get_stats().unwrap();
        assert_eq!(stats.document_count, 0);
        assert_eq!(stats.node_count, 0);
    }

    #[test]
    fn test_parse_md_node_text() {
        let text = "---\ntitle: Hello World\nauthor: Test\n---\n\n# Introduction\n\nSome body text.\n\n```python\ndef hello():\n    print(\"Hi\")\n```\n\nMore body.\n\n```rust\nfn main() {\n    println!(\"Hello!\");\n}\n```\n";
        let fields = parse_md_node_text(text);
        assert!(fields.front_matter.contains("title: Hello World"));
        assert!(fields.body.contains("Some body text."));
        assert!(fields.body.contains("# Introduction"));
        assert!(!fields.body.contains("```"));
        assert!(fields.code_blocks.contains("def hello()"));
        assert!(fields.code_blocks.contains("fn main()"));
    }

    #[test]
    fn test_parse_md_empty() {
        let fields = parse_md_node_text("");
        assert!(fields.front_matter.is_empty());
        assert!(fields.body.is_empty());
        assert!(fields.code_blocks.is_empty());
    }

    #[test]
    fn test_optimize_and_rebuild() {
        let index = FTS5Index::new(None, None).unwrap();
        let doc = sample_doc();
        index.index_document(&doc, false).unwrap();

        index.optimize().unwrap();
        index.rebuild().unwrap();

        let results = index.search("machine learning", None, 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_batch_index_multi_doc() {
        let index = FTS5Index::new(None, None).unwrap();

        let doc1 = sample_doc();
        let mut doc2 = Document::new("doc2", "other.md", SourceType::Markdown);
        let mut node = Node::new("0", "Quantum Computing");
        node.text = "Quantum computing uses qubits for parallel computation.".to_string();
        doc2.structure.push(node);

        let total = index.index_documents(&[doc1, doc2], false).unwrap();
        assert_eq!(total, 5);

        let stats = index.get_stats().unwrap();
        assert_eq!(stats.document_count, 2);

        let results = index.search("quantum", None, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.doc_id == "doc2"));
    }
}
