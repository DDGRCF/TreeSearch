//! Runs tree traversal and final ranking from host-provided in-memory candidate scores.

use std::collections::{HashMap, HashSet};

use crate::config::{SearchMode, TreeSearchConfig};
use crate::document::{Document, SearchResult};
use crate::engine::tree_walker::{PathResult, TreeSearcher};

/// Initial lexical or hybrid scores grouped by document ID and then node ID.
pub type CandidateScoreMap = HashMap<String, HashMap<String, f64>>;

/// One scored node candidate supplied by an external retrieval adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredNodeCandidate {
    /// Document identity matching [`Document::doc_id`].
    pub doc_id: String,
    /// Node identity within the document.
    pub node_id: String,
    /// Normalized retrieval score in `0.0..=1.0`.
    pub score: f64,
}

impl ScoredNodeCandidate {
    /// Creates one externally ranked candidate.
    pub fn new(doc_id: impl Into<String>, node_id: impl Into<String>, score: f64) -> Self {
        Self {
            doc_id: doc_id.into(),
            node_id: node_id.into(),
            score,
        }
    }
}

/// Tree paths and flattened ranked nodes produced by one in-memory search.
#[derive(Debug, Clone)]
pub struct CandidateSearchOutcome {
    /// Highest-ranked root-to-answer tree paths; empty in Flat mode.
    pub paths: Vec<PathResult>,
    /// Stable transport-ready node results after mode-specific ranking.
    pub results: Vec<SearchResult>,
}

/// Validation failure returned by strict external-score entry points.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CandidateSearchError {
    #[error("candidate search query cannot be empty")]
    EmptyQuery,
    #[error("candidate references unknown document `{doc_id}`")]
    UnknownDocument { doc_id: String },
    #[error("candidate references unknown node `{node_id}` in document `{doc_id}`")]
    UnknownNode { doc_id: String, node_id: String },
    #[error("candidate score for `{doc_id}/{node_id}` must be finite and within 0..=1")]
    InvalidScore { doc_id: String, node_id: String },
    #[error("candidate `{doc_id}/{node_id}` occurs more than once")]
    DuplicateCandidate { doc_id: String, node_id: String },
    #[error("document ID `{doc_id}` occurs more than once")]
    DuplicateDocument { doc_id: String },
    #[error("document `{doc_id}` has an invalid structure: {reason}")]
    InvalidDocument { doc_id: String, reason: String },
    #[error("candidate search configuration field `{field}` is invalid")]
    InvalidConfiguration { field: &'static str },
}

impl CandidateSearchOutcome {
    /// Creates an empty outcome for an empty or invalid candidate set.
    fn empty() -> Self {
        Self {
            paths: Vec::new(),
            results: Vec::new(),
        }
    }
}

/// Searches a document forest using a host-provided nested score map.
///
/// Scores must be finite and positive. If any accepted value is above `1.0`,
/// all accepted scores are divided by the global maximum, preserving cross-node
/// and cross-document ordering. Unknown documents/nodes, zero/negative scores,
/// and non-finite values are ignored. The documents may still contain zero-score
/// ancestors: traversal reaches them through the tree.
pub fn search_with_scores(
    query: &str,
    documents: &[Document],
    scores: &CandidateScoreMap,
    config: &TreeSearchConfig,
) -> CandidateSearchOutcome {
    let normalized = normalize_scores(documents, scores);
    search_normalized(query, documents, &normalized, config)
}

/// Strictly validates a nested score map before running candidate search.
///
/// Unlike [`search_with_scores`], this returns an error for an empty query,
/// unknown IDs, non-finite scores, negative scores, or scores above `1.0`.
/// Zero-score ancestors are accepted and ignored as retrieval seeds.
pub fn search_with_scores_strict(
    query: &str,
    documents: &[Document],
    scores: &CandidateScoreMap,
    config: &TreeSearchConfig,
) -> Result<CandidateSearchOutcome, CandidateSearchError> {
    validate_config(config)?;
    validate_scores(query, documents, scores)?;
    Ok(search_with_scores(query, documents, scores, config))
}

/// Searches from a flat candidate list, merging duplicate document/node pairs by maximum score.
///
/// This convenience API applies the same validation as [`search_with_scores`].
pub fn search_with_candidates(
    query: &str,
    documents: &[Document],
    candidates: &[ScoredNodeCandidate],
    config: &TreeSearchConfig,
) -> CandidateSearchOutcome {
    let mut scores = CandidateScoreMap::new();
    for candidate in candidates {
        if !candidate.score.is_finite() || candidate.score <= 0.0 {
            continue;
        }
        let score = candidate.score;
        scores
            .entry(candidate.doc_id.clone())
            .or_default()
            .entry(candidate.node_id.clone())
            .and_modify(|current| *current = current.max(score))
            .or_insert(score);
    }
    search_with_scores(query, documents, &scores, config)
}

/// Strictly validates a flat candidate list and rejects duplicate node identities.
pub fn search_with_candidates_strict(
    query: &str,
    documents: &[Document],
    candidates: &[ScoredNodeCandidate],
    config: &TreeSearchConfig,
) -> Result<CandidateSearchOutcome, CandidateSearchError> {
    let mut identities = HashSet::new();
    let mut scores = CandidateScoreMap::new();
    for candidate in candidates {
        let identity = (candidate.doc_id.clone(), candidate.node_id.clone());
        if !identities.insert(identity) {
            return Err(CandidateSearchError::DuplicateCandidate {
                doc_id: candidate.doc_id.clone(),
                node_id: candidate.node_id.clone(),
            });
        }
        scores
            .entry(candidate.doc_id.clone())
            .or_default()
            .insert(candidate.node_id.clone(), candidate.score);
    }
    search_with_scores_strict(query, documents, &scores, config)
}

/// Resolves Auto mode with the same hierarchy ratio used by the SQLite adapter.
pub fn resolve_search_mode(mode: SearchMode, documents: &[Document]) -> SearchMode {
    const MIN_TREE_DEPTH: u32 = 2;
    const TREE_RATIO_THRESHOLD: f64 = 0.3;

    if mode != SearchMode::Auto {
        return mode;
    }
    if documents.is_empty() {
        return SearchMode::Flat;
    }
    let tree_count = documents
        .iter()
        .filter(|document| {
            let benefits_from_tree = matches!(
                document.source_type,
                crate::document::SourceType::Markdown
                    | crate::document::SourceType::Json
                    | crate::document::SourceType::Yaml
                    | crate::document::SourceType::Toml
                    | crate::document::SourceType::Html
            );
            benefits_from_tree && document.max_depth() >= MIN_TREE_DEPTH
        })
        .count();
    let ratio = tree_count as f64 / documents.len() as f64;
    if ratio >= TREE_RATIO_THRESHOLD {
        SearchMode::Tree
    } else {
        SearchMode::Flat
    }
}

/// Validates numeric policies that participate in score comparisons.
fn validate_config(config: &TreeSearchConfig) -> Result<(), CandidateSearchError> {
    for (field, value) in [
        ("min_frontier_score", config.min_frontier_score),
        ("early_stop_score", config.early_stop_score),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CandidateSearchError::InvalidConfiguration { field });
        }
    }
    if config.early_stop_score < config.min_frontier_score {
        return Err(CandidateSearchError::InvalidConfiguration {
            field: "early_stop_score",
        });
    }
    Ok(())
}

/// Validates strict host-score invariants without mutating the caller's map.
fn validate_scores(
    query: &str,
    documents: &[Document],
    scores: &CandidateScoreMap,
) -> Result<(), CandidateSearchError> {
    if query.trim().is_empty() {
        return Err(CandidateSearchError::EmptyQuery);
    }
    let mut known_documents: HashMap<&str, HashSet<&str>> = HashMap::new();
    for document in documents {
        if known_documents.contains_key(document.doc_id.as_str()) {
            return Err(CandidateSearchError::DuplicateDocument {
                doc_id: document.doc_id.clone(),
            });
        }
        if let Err(error) = document.validate_structure() {
            return Err(CandidateSearchError::InvalidDocument {
                doc_id: document.doc_id.clone(),
                reason: error.to_string(),
            });
        }
        known_documents.insert(
            document.doc_id.as_str(),
            document
                .flatten_nodes()
                .into_iter()
                .map(|node| node.node_id.as_str())
                .collect(),
        );
    }
    for (doc_id, node_scores) in scores {
        let Some(known_nodes) = known_documents.get(doc_id.as_str()) else {
            return Err(CandidateSearchError::UnknownDocument {
                doc_id: doc_id.clone(),
            });
        };
        for (node_id, score) in node_scores {
            if !known_nodes.contains(node_id.as_str()) {
                return Err(CandidateSearchError::UnknownNode {
                    doc_id: doc_id.clone(),
                    node_id: node_id.clone(),
                });
            }
            if !score.is_finite() || *score < 0.0 || *score > 1.0 {
                return Err(CandidateSearchError::InvalidScore {
                    doc_id: doc_id.clone(),
                    node_id: node_id.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Filters host scores to known nodes and normalizes the range expected by the scorer.
fn normalize_scores(documents: &[Document], scores: &CandidateScoreMap) -> CandidateScoreMap {
    let mut document_counts: HashMap<&str, usize> = HashMap::new();
    for document in documents {
        *document_counts.entry(document.doc_id.as_str()).or_default() += 1;
    }
    let known_documents: HashMap<&str, HashSet<&str>> = documents
        .iter()
        .filter(|document| {
            document_counts.get(document.doc_id.as_str()) == Some(&1)
                && document.validate_structure().is_ok()
        })
        .map(|document| {
            (
                document.doc_id.as_str(),
                document
                    .flatten_nodes()
                    .into_iter()
                    .map(|node| node.node_id.as_str())
                    .collect(),
            )
        })
        .collect();

    let mut normalized: CandidateScoreMap = scores
        .iter()
        .filter_map(|(doc_id, node_scores)| {
            let known_nodes = known_documents.get(doc_id.as_str())?;
            let normalized: HashMap<String, f64> = node_scores
                .iter()
                .filter_map(|(node_id, score)| {
                    if !known_nodes.contains(node_id.as_str())
                        || !score.is_finite()
                        || *score <= 0.0
                    {
                        return None;
                    }
                    Some((node_id.clone(), *score))
                })
                .collect();
            (!normalized.is_empty()).then_some((doc_id.clone(), normalized))
        })
        .collect();
    let global_max = normalized
        .values()
        .flat_map(HashMap::values)
        .copied()
        .fold(0.0_f64, f64::max);
    if global_max > 1.0 {
        for score in normalized.values_mut().flat_map(HashMap::values_mut) {
            *score /= global_max;
        }
    }
    normalized
}

/// Runs the existing tree walker and maps its output to the stable search-result contract.
fn search_normalized(
    query: &str,
    documents: &[Document],
    scores: &CandidateScoreMap,
    config: &TreeSearchConfig,
) -> CandidateSearchOutcome {
    if query.trim().is_empty() || documents.is_empty() || scores.is_empty() {
        return CandidateSearchOutcome::empty();
    }

    match resolve_search_mode(config.search_mode, documents) {
        SearchMode::Flat => return flat_search(documents, scores, config),
        SearchMode::Tree => {}
        SearchMode::Auto => unreachable!("resolve_search_mode must return a concrete mode"),
    }

    let searcher = TreeSearcher::new(config);
    let (paths, flat_nodes) = searcher.search(query, documents, scores);
    let doc_map: HashMap<&str, &Document> = documents
        .iter()
        .map(|document| (document.doc_id.as_str(), document))
        .collect();
    let depth_maps: HashMap<&str, HashMap<String, u32>> = documents
        .iter()
        .map(|document| (document.doc_id.as_str(), document.build_depth_map()))
        .collect();
    let parent_maps: HashMap<&str, HashMap<String, Option<String>>> = documents
        .iter()
        .map(|document| (document.doc_id.as_str(), document.build_parent_map()))
        .collect();
    let node_maps: HashMap<&str, HashMap<&str, &crate::document::Node>> = documents
        .iter()
        .map(|document| (document.doc_id.as_str(), document.build_node_map()))
        .collect();
    let top_k = config.max_nodes_per_doc.saturating_mul(config.top_k_docs);
    let mut results: Vec<SearchResult> = flat_nodes
        .into_iter()
        .filter_map(|flat| {
            let document = doc_map.get(flat.doc_id.as_str())?;
            let node_map = node_maps.get(flat.doc_id.as_str())?;
            let node = node_map.get(flat.node_id.as_str())?;
            let breadcrumb = breadcrumb_titles(
                &flat.node_id,
                parent_maps.get(flat.doc_id.as_str())?,
                node_map,
            );
            let depth = depth_maps
                .get(flat.doc_id.as_str())
                .and_then(|depths| depths.get(&flat.node_id))
                .copied()
                .unwrap_or(0);
            Some(SearchResult {
                node_id: flat.node_id,
                doc_id: flat.doc_id,
                doc_name: flat.doc_name,
                title: flat.title,
                summary: node.summary.clone(),
                text: bounded_result_text(&node.text, config.max_result_chars),
                source_type: document.source_type.to_string(),
                source_path: document.source_path.clone(),
                line_start: node.line_start,
                line_end: node.line_end,
                score: flat.score,
                depth,
                breadcrumb,
            })
        })
        .collect();

    for path in &paths {
        if results
            .iter()
            .any(|result| result.doc_id == path.doc_id && result.node_id == path.target_node_id)
        {
            continue;
        }
        let Some(document) = doc_map.get(path.doc_id.as_str()) else {
            continue;
        };
        let Some(node) = node_maps
            .get(path.doc_id.as_str())
            .and_then(|nodes| nodes.get(path.target_node_id.as_str()))
        else {
            continue;
        };
        results.push(SearchResult {
            node_id: path.target_node_id.clone(),
            doc_id: path.doc_id.clone(),
            doc_name: path.doc_name.clone(),
            title: node.title.clone(),
            summary: node.summary.clone(),
            text: bounded_result_text(&node.text, config.max_result_chars),
            source_type: document.source_type.to_string(),
            source_path: document.source_path.clone(),
            line_start: node.line_start,
            line_end: node.line_end,
            score: path.score,
            depth: depth_maps
                .get(path.doc_id.as_str())
                .and_then(|depths| depths.get(&path.target_node_id))
                .copied()
                .unwrap_or(0),
            breadcrumb: path.path.iter().map(|item| item.title.clone()).collect(),
        });
    }

    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    limit_ranked_results(&mut results, config);
    let returned_targets: HashSet<(&str, &str)> = results
        .iter()
        .map(|result| (result.doc_id.as_str(), result.node_id.as_str()))
        .collect();
    let paths = paths
        .into_iter()
        .filter(|path| {
            returned_targets.contains(&(path.doc_id.as_str(), path.target_node_id.as_str()))
        })
        .collect();
    debug_assert!(results.len() <= top_k);
    CandidateSearchOutcome { paths, results }
}

/// Maps host-ranked nodes directly into the stable result contract for Flat mode.
fn flat_search(
    documents: &[Document],
    scores: &CandidateScoreMap,
    config: &TreeSearchConfig,
) -> CandidateSearchOutcome {
    let mut results = Vec::new();
    for document in documents {
        let Some(node_scores) = scores.get(&document.doc_id) else {
            continue;
        };
        let depth_map = document.build_depth_map();
        let parent_map = document.build_parent_map();
        let node_map = document.build_node_map();
        for (node_id, score) in node_scores {
            let Some(node) = node_map.get(node_id.as_str()) else {
                continue;
            };
            let breadcrumb = breadcrumb_titles(node_id, &parent_map, &node_map);
            results.push(SearchResult {
                node_id: node_id.clone(),
                doc_id: document.doc_id.clone(),
                doc_name: document.doc_name.clone(),
                title: node.title.clone(),
                summary: node.summary.clone(),
                text: bounded_result_text(&node.text, config.max_result_chars),
                source_type: document.source_type.to_string(),
                source_path: document.source_path.clone(),
                line_start: node.line_start,
                line_end: node.line_end,
                score: *score,
                depth: depth_map.get(node_id).copied().unwrap_or(0),
                breadcrumb,
            });
        }
    }
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    limit_ranked_results(&mut results, config);
    CandidateSearchOutcome {
        paths: Vec::new(),
        results,
    }
}

/// Resolves one breadcrumb from prebuilt maps in O(depth), with cycle defense.
fn breadcrumb_titles(
    node_id: &str,
    parent_map: &HashMap<String, Option<String>>,
    node_map: &HashMap<&str, &crate::document::Node>,
) -> Vec<String> {
    let mut titles = Vec::new();
    let mut visited = HashSet::new();
    let mut current = Some(node_id);
    while let Some(current_id) = current {
        if !visited.insert(current_id) {
            return Vec::new();
        }
        let Some(node) = node_map.get(current_id) else {
            return Vec::new();
        };
        titles.push(node.title.clone());
        current = parent_map
            .get(current_id)
            .and_then(|parent| parent.as_deref());
    }
    titles.reverse();
    titles
}

/// Enforces both the distinct-document and per-document result limits.
fn limit_ranked_results(results: &mut Vec<SearchResult>, config: &TreeSearchConfig) {
    if config.top_k_docs == 0 || config.max_nodes_per_doc == 0 {
        results.clear();
        return;
    }
    let mut selected_documents = HashSet::new();
    let mut per_document = HashMap::<String, usize>::new();
    results.retain(|result| {
        if !selected_documents.contains(&result.doc_id)
            && selected_documents.len() >= config.top_k_docs
        {
            return false;
        }
        let count = per_document.entry(result.doc_id.clone()).or_default();
        if *count >= config.max_nodes_per_doc {
            return false;
        }
        selected_documents.insert(result.doc_id.clone());
        *count += 1;
        true
    });
}

/// Copies a result body up to a UTF-8-safe character limit.
fn bounded_result_text(text: &str, max_characters: usize) -> String {
    match text.char_indices().nth(max_characters) {
        Some((end, _)) => text[..end].to_string(),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Node, SourceType};

    /// Builds one nested document for candidate-search behavior tests.
    fn nested_document() -> Document {
        let mut document = Document::new("guide", "Guide", SourceType::Markdown);
        let mut root = Node::new("root", "Root");
        root.text = "General introduction".into();
        let mut section = Node::new("section", "Authentication");
        section.text = "Authentication overview".into();
        let mut leaf = Node::new("leaf", "Token validation");
        leaf.text = "Validate bearer tokens before serving protected data.".into();
        section.children.push(leaf);
        root.children.push(section);
        document.structure.push(root);
        document
    }

    /// Uses deterministic tree-search limits shared by these tests.
    fn tree_config() -> TreeSearchConfig {
        TreeSearchConfig {
            top_k_docs: 1,
            max_nodes_per_doc: 5,
            anchor_top_k: 3,
            path_top_k: 3,
            ..TreeSearchConfig::default()
        }
    }

    #[test]
    fn scored_leaf_produces_root_to_leaf_path() {
        let document = nested_document();
        let candidates = [ScoredNodeCandidate::new("guide", "leaf", 0.9)];
        let mut config = tree_config();
        config.search_mode = SearchMode::Tree;
        let outcome =
            search_with_candidates("validate bearer tokens", &[document], &candidates, &config);

        assert!(!outcome.results.is_empty());
        assert!(outcome.paths.iter().any(|path| {
            path.path
                .iter()
                .map(|item| item.node_id.as_str())
                .eq(["root", "section", "leaf"])
        }));
    }

    #[test]
    fn empty_unknown_and_non_finite_candidates_are_ignored() {
        let document = nested_document();
        let candidates = [
            ScoredNodeCandidate::new("missing", "leaf", 1.0),
            ScoredNodeCandidate::new("guide", "missing", 1.0),
            ScoredNodeCandidate::new("guide", "leaf", f64::NAN),
            ScoredNodeCandidate::new("guide", "leaf", f64::INFINITY),
            ScoredNodeCandidate::new("guide", "leaf", 0.0),
            ScoredNodeCandidate::new("guide", "leaf", -1.0),
        ];

        let outcome = search_with_candidates("tokens", &[document], &candidates, &tree_config());
        assert!(outcome.paths.is_empty());
        assert!(outcome.results.is_empty());
    }

    #[test]
    fn duplicate_candidates_keep_the_highest_score_once() {
        let document = nested_document();
        let duplicates = [
            ScoredNodeCandidate::new("guide", "leaf", 0.2),
            ScoredNodeCandidate::new("guide", "leaf", 0.9),
            ScoredNodeCandidate::new("guide", "leaf", 0.4),
        ];
        let single = [ScoredNodeCandidate::new("guide", "leaf", 0.9)];

        let duplicate_outcome = search_with_candidates(
            "tokens",
            std::slice::from_ref(&document),
            &duplicates,
            &tree_config(),
        );
        let single_outcome = search_with_candidates("tokens", &[document], &single, &tree_config());
        let duplicate_pairs: Vec<(&str, &str, f64)> = duplicate_outcome
            .results
            .iter()
            .map(|result| {
                (
                    result.doc_id.as_str(),
                    result.node_id.as_str(),
                    result.score,
                )
            })
            .collect();
        let single_pairs: Vec<(&str, &str, f64)> = single_outcome
            .results
            .iter()
            .map(|result| {
                (
                    result.doc_id.as_str(),
                    result.node_id.as_str(),
                    result.score,
                )
            })
            .collect();

        assert_eq!(duplicate_pairs, single_pairs);
    }

    #[test]
    fn flat_mode_maps_only_supplied_candidates_without_paths() {
        let document = nested_document();
        let candidates = [ScoredNodeCandidate::new("guide", "leaf", 0.9)];
        let mut config = tree_config();
        config.search_mode = SearchMode::Flat;

        let outcome = search_with_candidates("tokens", &[document], &candidates, &config);

        assert!(outcome.paths.is_empty());
        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.results[0].node_id, "leaf");
        assert_eq!(outcome.results[0].depth, 2);
    }

    #[test]
    fn auto_mode_uses_hierarchy_and_source_type() {
        let nested = nested_document();
        let mut plain = Document::new("plain", "Plain", SourceType::Text);
        plain.structure.push(Node::new("only", "Only node"));
        let config = tree_config();

        assert_eq!(
            resolve_search_mode(SearchMode::Auto, std::slice::from_ref(&nested)),
            SearchMode::Tree
        );
        assert_eq!(
            resolve_search_mode(SearchMode::Auto, std::slice::from_ref(&plain)),
            SearchMode::Flat
        );

        let nested_outcome = search_with_candidates(
            "tokens",
            &[nested],
            &[ScoredNodeCandidate::new("guide", "leaf", 0.9)],
            &config,
        );
        let plain_outcome = search_with_candidates(
            "only",
            &[plain],
            &[ScoredNodeCandidate::new("plain", "only", 0.9)],
            &config,
        );
        assert!(!nested_outcome.paths.is_empty());
        assert!(plain_outcome.paths.is_empty());
        assert_eq!(plain_outcome.results.len(), 1);
    }

    #[test]
    fn strict_scores_accept_zero_ancestors_and_reject_invalid_ids_and_values() {
        let document = nested_document();
        let mut config = tree_config();
        config.search_mode = SearchMode::Tree;
        let valid = CandidateScoreMap::from([(
            "guide".to_string(),
            HashMap::from([
                ("root".to_string(), 0.0),
                ("section".to_string(), 0.0),
                ("leaf".to_string(), 0.9),
            ]),
        )]);
        let valid_outcome =
            search_with_scores_strict("tokens", std::slice::from_ref(&document), &valid, &config)
                .expect("valid strict scores");
        assert!(valid_outcome
            .results
            .iter()
            .all(|result| result.score.is_finite() && (0.0..=1.0).contains(&result.score)));

        let unknown = CandidateScoreMap::from([(
            "guide".to_string(),
            HashMap::from([("missing".to_string(), 0.5)]),
        )]);
        assert!(matches!(
            search_with_scores_strict("tokens", std::slice::from_ref(&document), &unknown, &config),
            Err(CandidateSearchError::UnknownNode { .. })
        ));

        let invalid = CandidateScoreMap::from([(
            "guide".to_string(),
            HashMap::from([("leaf".to_string(), f64::NAN)]),
        )]);
        assert!(matches!(
            search_with_scores_strict("tokens", &[document], &invalid, &config),
            Err(CandidateSearchError::InvalidScore { .. })
        ));
    }

    #[test]
    fn strict_candidates_reject_duplicates() {
        let document = nested_document();
        let duplicates = [
            ScoredNodeCandidate::new("guide", "leaf", 0.2),
            ScoredNodeCandidate::new("guide", "leaf", 0.9),
        ];

        assert!(matches!(
            search_with_candidates_strict("tokens", &[document], &duplicates, &tree_config()),
            Err(CandidateSearchError::DuplicateCandidate { .. })
        ));
    }

    #[test]
    fn lenient_raw_scores_are_globally_calibrated_without_losing_order() {
        let document = nested_document();
        let mut config = tree_config();
        config.search_mode = SearchMode::Flat;
        let candidates = [
            ScoredNodeCandidate::new("guide", "section", 20.0),
            ScoredNodeCandidate::new("guide", "leaf", 10.0),
        ];

        let outcome = search_with_candidates("tokens", &[document], &candidates, &config);
        assert_eq!(outcome.results[0].node_id, "section");
        assert_eq!(outcome.results[0].score, 1.0);
        assert_eq!(outcome.results[1].score, 0.5);
    }

    #[test]
    fn strict_search_rejects_duplicate_document_and_node_identities() {
        let document = nested_document();
        let scores = CandidateScoreMap::from([(
            "guide".to_string(),
            HashMap::from([("leaf".to_string(), 0.9)]),
        )]);
        assert!(matches!(
            search_with_scores_strict(
                "tokens",
                &[document.clone(), document.clone()],
                &scores,
                &tree_config(),
            ),
            Err(CandidateSearchError::DuplicateDocument { .. })
        ));

        let mut duplicate_node = document;
        duplicate_node.structure[0].children[0].node_id = "root".into();
        assert!(matches!(
            search_with_scores_strict("tokens", &[duplicate_node], &scores, &tree_config()),
            Err(CandidateSearchError::InvalidDocument { .. })
        ));
    }

    #[test]
    fn result_limits_apply_per_document_with_stable_ties() {
        let first = nested_document();
        let mut second = nested_document();
        second.doc_id = "second".into();
        second.doc_name = "Second".into();
        let candidates = [
            ScoredNodeCandidate::new("guide", "section", 0.9),
            ScoredNodeCandidate::new("guide", "leaf", 0.9),
            ScoredNodeCandidate::new("second", "section", 0.9),
            ScoredNodeCandidate::new("second", "leaf", 0.9),
        ];
        let mut config = tree_config();
        config.search_mode = SearchMode::Flat;
        config.top_k_docs = 1;
        config.max_nodes_per_doc = 1;

        for candidates in [candidates.to_vec(), candidates.into_iter().rev().collect()] {
            let outcome = search_with_candidates(
                "tokens",
                &[first.clone(), second.clone()],
                &candidates,
                &config,
            );
            assert_eq!(outcome.results.len(), 1);
            assert_eq!(outcome.results[0].doc_id, "guide");
            assert_eq!(outcome.results[0].node_id, "leaf");
        }
    }

    #[test]
    fn strict_search_rejects_non_finite_thresholds() {
        let document = nested_document();
        let scores = CandidateScoreMap::from([(
            "guide".to_string(),
            HashMap::from([("leaf".to_string(), 0.9)]),
        )]);
        let mut config = tree_config();
        config.min_frontier_score = f64::NAN;
        assert!(matches!(
            search_with_scores_strict("tokens", &[document], &scores, &config),
            Err(CandidateSearchError::InvalidConfiguration {
                field: "min_frontier_score"
            })
        ));
    }

    #[test]
    fn tree_paths_use_the_actual_anchor_and_utf8_safe_snippets() {
        let mut document = nested_document();
        document.structure[0].children[0].children[0].text = "报价".repeat(200);
        let candidates = [ScoredNodeCandidate::new("guide", "leaf", 0.9)];
        let mut config = tree_config();
        config.search_mode = SearchMode::Tree;
        config.early_stop_score = 1.0;

        let outcome = search_with_candidates("报价", &[document], &candidates, &config);
        let leaf_path = outcome
            .paths
            .iter()
            .find(|path| path.target_node_id == "leaf")
            .expect("leaf path");
        assert_eq!(leaf_path.anchor_node_id, "leaf");
        assert_eq!(leaf_path.snippet.chars().count(), 300);
    }

    #[test]
    fn result_text_respects_utf8_character_limit() {
        let mut document = nested_document();
        document.structure[0].children[0].children[0].text = "证据".repeat(20);
        let mut config = tree_config();
        config.search_mode = SearchMode::Flat;
        config.max_result_chars = 7;
        let outcome = search_with_candidates(
            "证据",
            &[document],
            &[ScoredNodeCandidate::new("guide", "leaf", 0.9)],
            &config,
        );
        assert_eq!(outcome.results[0].text.chars().count(), 7);
        assert_eq!(outcome.results[0].text, "证据证据证据证");
    }

    #[test]
    fn tree_ranking_is_stable_across_document_and_candidate_orders() {
        let first = nested_document();
        let mut second = nested_document();
        second.doc_id = "second".into();
        second.doc_name = "Second".into();
        let candidates = vec![
            ScoredNodeCandidate::new("guide", "leaf", 0.8),
            ScoredNodeCandidate::new("second", "leaf", 0.8),
        ];
        let mut config = tree_config();
        config.search_mode = SearchMode::Tree;
        config.top_k_docs = 2;
        config.path_top_k = 4;

        let mut expected = None;
        for (documents, candidates) in [
            (vec![first.clone(), second.clone()], candidates.clone()),
            (
                vec![second.clone(), first.clone()],
                candidates.iter().cloned().rev().collect(),
            ),
        ] {
            let outcome = search_with_candidates("tokens", &documents, &candidates, &config);
            let projection = (
                outcome
                    .results
                    .iter()
                    .map(|result| (result.doc_id.clone(), result.node_id.clone(), result.score))
                    .collect::<Vec<_>>(),
                outcome
                    .paths
                    .iter()
                    .map(|path| {
                        (
                            path.doc_id.clone(),
                            path.anchor_node_id.clone(),
                            path.target_node_id.clone(),
                            path.score,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            if let Some(expected) = &expected {
                assert_eq!(&projection, expected);
            } else {
                expected = Some(projection);
            }
        }
    }

    #[test]
    #[ignore = "explicit local scalability baseline"]
    fn benchmark_large_candidate_forest() {
        let mut documents = Vec::new();
        let mut candidates = Vec::new();
        for document_index in 0..100 {
            let doc_id = format!("doc-{document_index:03}");
            let mut document = Document::new(&doc_id, &doc_id, SourceType::Markdown);
            let mut root = Node::new("root", "Knowledge base");
            for node_index in 0..20 {
                let node_id = format!("node-{node_index:02}");
                let mut node = Node::new(&node_id, format!("Section {node_index}"));
                node.text = format!(
                    "bounded retrieval evidence for document {document_index} node {node_index}"
                );
                root.children.push(node);
                candidates.push(ScoredNodeCandidate::new(
                    &doc_id,
                    node_id,
                    (node_index + 1) as f64,
                ));
            }
            document.structure.push(root);
            documents.push(document);
        }
        let mut config = tree_config();
        config.search_mode = SearchMode::Tree;
        config.top_k_docs = 10;
        config.max_nodes_per_doc = 5;
        let iterations = 100;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            let outcome = search_with_candidates(
                "bounded retrieval evidence",
                &documents,
                &candidates,
                &config,
            );
            assert!(!outcome.results.is_empty());
        }
        let elapsed = started.elapsed();
        eprintln!(
            "candidate_search docs=100 nodes=2000 iterations={iterations} avg_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0 / iterations as f64
        );
    }

    #[cfg(feature = "sqlite-fts")]
    #[test]
    fn in_memory_scores_match_sqlite_adapter_in_flat_and_tree_modes() {
        use crate::config::SearchMode;
        use crate::engine::{fts::FTS5Index, search};

        let document = nested_document();
        let documents = vec![document];
        let index = FTS5Index::new(None, None).expect("in-memory FTS index");
        for document in &documents {
            index
                .index_document(document, false)
                .expect("index test document");
        }
        let doc_ids = vec!["guide".to_string()];
        for mode in [SearchMode::Flat, SearchMode::Tree] {
            let scores = index
                .score_nodes_batch_with_expr(
                    "tokens",
                    Some(&doc_ids),
                    if mode == SearchMode::Tree { 0.6 } else { 0.0 },
                    None,
                )
                .expect("score test candidates");
            let mut config = tree_config();
            config.search_mode = mode;
            let sqlite = search::search("tokens", &documents, &index, &config)
                .expect("SQLite-backed search");
            let in_memory = search_with_scores("tokens", &documents, &scores, &config);
            let sqlite_projection: Vec<(&str, &str, f64)> = sqlite
                .iter()
                .map(|result| {
                    (
                        result.doc_id.as_str(),
                        result.node_id.as_str(),
                        result.score,
                    )
                })
                .collect();
            let memory_projection: Vec<(&str, &str, f64)> = in_memory
                .results
                .iter()
                .map(|result| {
                    (
                        result.doc_id.as_str(),
                        result.node_id.as_str(),
                        result.score,
                    )
                })
                .collect();

            assert_eq!(memory_projection, sqlite_projection, "mode={mode:?}");
        }
    }
}
