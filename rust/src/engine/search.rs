//! Search pipeline with UTF-8-safe query routing and FTS5-backed flat/tree scoring.
//!
//! Two-stage pipeline:
//! 1. FTS5 pre-scoring: batch score all documents
//! 2. Mode routing: flat mode (FTS5 results directly) or tree mode (tree walk + reranking)

use std::collections::HashMap;

use anyhow::{bail, Result};
use regex::{Regex, RegexBuilder};

use crate::config::{SearchMode, TreeSearchConfig};
use crate::document::{Document, Node, SearchResult};
use crate::engine::candidate_search;
use crate::engine::fts::FTS5Index;

#[derive(Debug, Clone)]
struct QueryMode {
    effective_query: String,
    fts_expression: Option<String>,
    regex_pattern: Option<String>,
}

fn classify_query_mode(query: &str, fts_expression: Option<&str>, regex: bool) -> QueryMode {
    if regex {
        return QueryMode {
            effective_query: query.to_string(),
            fts_expression: None,
            regex_pattern: Some(query.to_string()),
        };
    }
    if let Some(expr) = fts_expression {
        return QueryMode {
            effective_query: if query.is_empty() {
                expr.to_string()
            } else {
                query.to_string()
            },
            fts_expression: Some(expr.to_string()),
            regex_pattern: None,
        };
    }
    let trimmed = query.trim();
    let prefix_body = trimmed.strip_suffix('*').unwrap_or(trimmed);
    let no_internal_star = !prefix_body.contains('*');
    let middle = trimmed
        .strip_prefix('*')
        .and_then(|value| value.strip_suffix('*'))
        .unwrap_or("");

    if trimmed.starts_with('*')
        && trimmed.ends_with('*')
        && trimmed.chars().count() > 2
        && !middle.contains('*')
        && !middle.chars().any(|c| c.is_whitespace())
    {
        let term = middle.to_string();
        return QueryMode {
            effective_query: term.clone(),
            fts_expression: None,
            regex_pattern: Some(regex::escape(&term)),
        };
    }

    if trimmed.ends_with('*')
        && !trimmed.starts_with('*')
        && trimmed.chars().count() > 1
        && no_internal_star
        && !prefix_body.chars().any(|c| c.is_whitespace())
    {
        return QueryMode {
            effective_query: prefix_body.to_string(),
            fts_expression: Some(trimmed.to_string()),
            regex_pattern: None,
        };
    }

    QueryMode {
        effective_query: query.to_string(),
        fts_expression: None,
        regex_pattern: None,
    }
}

fn compile_contains_regex(pattern: &str) -> Result<Regex> {
    Ok(RegexBuilder::new(pattern).case_insensitive(true).build()?)
}

fn regex_score_doc(doc: &Document, regex: &Regex) -> HashMap<String, f64> {
    fn count_matches(regex: &Regex, text: &str) -> usize {
        regex.find_iter(text).count()
    }

    fn score_node(node: &Node, regex: &Regex, scores: &mut HashMap<String, f64>) {
        let hit_count = count_matches(regex, &node.title)
            + count_matches(regex, &node.summary)
            + count_matches(regex, &node.text);
        if hit_count > 0 {
            scores.insert(node.node_id.clone(), hit_count as f64);
        }
    }

    let mut scores = HashMap::new();
    let mut stack: Vec<&Node> = doc.structure.iter().rev().collect();
    while let Some(node) = stack.pop() {
        score_node(node, regex, &mut scores);
        stack.extend(node.children.iter().rev());
    }

    if let Some(max_score) = scores.values().copied().reduce(f64::max) {
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }
    }
    scores
}

/// Unified search entry point.
pub fn search(
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
) -> Result<Vec<SearchResult>> {
    search_with_options(query, documents, fts_index, config, None, false)
}

pub fn search_with_options(
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    fts_expression: Option<&str>,
    regex: bool,
) -> Result<Vec<SearchResult>> {
    if regex && fts_expression.is_some() {
        bail!("regex and fts_expression cannot be used together");
    }
    if query.trim().is_empty() && fts_expression.is_none() {
        return Ok(Vec::new());
    }

    let query_mode = classify_query_mode(query, fts_expression, regex);
    let mode = candidate_search::resolve_search_mode(config.search_mode, documents);
    match mode {
        SearchMode::Flat => search_flat(documents, fts_index, config, &query_mode),
        SearchMode::Tree => search_tree(documents, fts_index, config, &query_mode),
        SearchMode::Auto => unreachable!("resolve_search_mode should never return Auto"),
    }
}

/// Flat search: FTS5 results directly, ranked by BM25.
fn search_flat(
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    query_mode: &QueryMode,
) -> Result<Vec<SearchResult>> {
    let scores = if let Some(pattern) = &query_mode.regex_pattern {
        let regex = compile_contains_regex(pattern)?;
        documents
            .iter()
            .filter_map(|document| {
                let scores = regex_score_doc(document, &regex);
                (!scores.is_empty()).then_some((document.doc_id.clone(), scores))
            })
            .collect()
    } else {
        let doc_ids: Vec<String> = documents.iter().map(|d| d.doc_id.clone()).collect();
        fts_index.score_nodes_batch_with_expr(
            &query_mode.effective_query,
            Some(&doc_ids),
            0.0,
            query_mode.fts_expression.as_deref(),
        )?
    };
    let mut flat_config = config.clone();
    flat_config.search_mode = SearchMode::Flat;
    Ok(candidate_search::search_with_scores(
        &query_mode.effective_query,
        documents,
        &scores,
        &flat_config,
    )
    .results)
}

/// Tree search: anchor retrieval + tree walk + path scoring + flat reranking.
fn search_tree(
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    query_mode: &QueryMode,
) -> Result<Vec<SearchResult>> {
    // Get FTS5 scores for all documents (single batch query)
    let doc_ids: Vec<String> = documents.iter().map(|d| d.doc_id.clone()).collect();
    let fts_score_map = if let Some(pattern) = &query_mode.regex_pattern {
        let regex = compile_contains_regex(pattern)?;
        documents
            .iter()
            .filter_map(|doc| {
                let scores = regex_score_doc(doc, &regex);
                if scores.is_empty() {
                    None
                } else {
                    Some((doc.doc_id.clone(), scores))
                }
            })
            .collect()
    } else {
        fts_index.score_nodes_batch_with_expr(
            &query_mode.effective_query,
            Some(&doc_ids),
            0.6,
            query_mode.fts_expression.as_deref(),
        )?
    };

    Ok(candidate_search::search_with_scores(
        &query_mode.effective_query,
        documents,
        &fts_score_map,
        config,
    )
    .results)
}

#[cfg(test)]
mod tests {
    use super::{classify_query_mode, search, search_with_options};
    use crate::config::{SearchMode, TreeSearchConfig};
    use crate::document::{Document, Node, SourceType};
    use crate::engine::fts::FTS5Index;

    fn wildcard_documents() -> Vec<Document> {
        let mut exact = Document::new("exact", "Exact Auth", SourceType::Text);
        let mut exact_root = Node::new("0", "Exact Auth");
        exact_root.summary = "Contains the exact auth token.".to_string();
        exact_root.text = "Use auth tokens for API access.".to_string();
        exact.structure.push(exact_root);

        let mut prefix = Document::new("prefix", "Authentication Guide", SourceType::Text);
        let mut prefix_root = Node::new("0", "Authentication");
        prefix_root.summary = "Authentication and authorizer details.".to_string();
        prefix_root.text = "Authentication depends on an authorizer service.".to_string();
        prefix.structure.push(prefix_root);

        let mut contains = Document::new("contains", "OAuth Guide", SourceType::Text);
        let mut contains_root = Node::new("0", "OAuth");
        contains_root.summary = "OAuth callback handling.".to_string();
        contains_root.text = "OAuth callbacks must be validated.".to_string();
        contains.structure.push(contains_root);

        vec![exact, prefix, contains]
    }

    fn default_config() -> TreeSearchConfig {
        TreeSearchConfig {
            search_mode: SearchMode::Flat,
            top_k_docs: 3,
            max_nodes_per_doc: 5,
            ..TreeSearchConfig::default()
        }
    }

    #[test]
    fn test_plain_query_preserves_exact_term_behavior() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("auth", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(!doc_names.contains(&"Authentication Guide"));
        assert!(!doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_suffix_star_query_uses_prefix_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("auth*", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(!doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_explicit_fts_expression_uses_prefix_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search_with_options(
            "ignored",
            &docs,
            &index,
            &default_config(),
            Some("auth*"),
            false,
        )
        .unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(!doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_surrounded_star_query_uses_contains_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("*auth*", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_explicit_regex_query_uses_regex_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results =
            search_with_options("o?auth", &docs, &index, &default_config(), None, true).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_explicit_regex_invalid_pattern_returns_error() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let error =
            search_with_options("(", &docs, &index, &default_config(), None, true).unwrap_err();

        assert!(error.to_string().contains("regex parse error"));
    }

    #[test]
    fn test_unsupported_wildcard_shape_falls_back_to_plain_query() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("au*th", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();
        assert_eq!(doc_names, vec!["Exact Auth"]);
    }

    #[test]
    fn test_cjk_punctuation_query_classification_is_utf8_safe() {
        let mode = classify_query_mode("如何自行报价？", None, false);
        assert_eq!(mode.effective_query, "如何自行报价？");
        assert!(mode.fts_expression.is_none());
        assert!(mode.regex_pattern.is_none());
    }
}
