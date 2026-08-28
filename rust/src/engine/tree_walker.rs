//! Tree Walker — Best-First Search over document trees.
//!
//! Core algorithm (ported from Python tree_searcher.py):
//! 1. Anchor Retrieval: use host-provided scores to find high-value entry nodes
//! 2. Tree Walk: BFS expansion from anchors along parent/child/sibling edges
//! 3. Path Aggregation: select best root-to-leaf paths as results

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::config::TreeSearchConfig;
use crate::document::{Document, Node};
use crate::scorer::heuristics::{
    build_query_plan_with_mode, check_phrase_match, check_title_match, compute_term_overlap,
    estimate_idf, is_generic_section, score_anchor, score_path, score_walk_node_with_input,
    QueryPlan, WalkScoreInput,
};

/// State in the Best-First Search frontier.
#[derive(Debug, Clone)]
pub struct SearchState {
    pub doc_id: String,
    pub node_id: String,
    pub score: f64,
    pub hop: usize,
    pub source: String,
    pub path: Vec<String>,
    pub max_ancestor_score: f64,
    /// Original retrieval anchor that initiated this walk branch.
    pub anchor_node_id: String,
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for SearchState {}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher score first, then lower hop and lexical identity.
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.hop.cmp(&self.hop))
            .then_with(|| other.doc_id.cmp(&self.doc_id))
            .then_with(|| other.node_id.cmp(&self.node_id))
            .then_with(|| self.max_ancestor_score.total_cmp(&other.max_ancestor_score))
            .then_with(|| other.anchor_node_id.cmp(&self.anchor_node_id))
            .then_with(|| other.source.cmp(&self.source))
            .then_with(|| other.path.cmp(&self.path))
    }
}

/// A scored root-to-answer path.
#[derive(Debug, Clone)]
pub struct PathResult {
    pub doc_id: String,
    pub doc_name: String,
    pub score: f64,
    pub anchor_node_id: String,
    pub target_node_id: String,
    pub path: Vec<PathNode>,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub struct PathNode {
    pub node_id: String,
    pub title: String,
}

/// Flat node from tree walker reranking.
#[derive(Debug, Clone)]
pub struct FlatNode {
    pub node_id: String,
    pub doc_id: String,
    pub doc_name: String,
    pub title: String,
    pub score: f64,
    pub text: String,
}

/// Tree searcher engine.
pub struct TreeSearcher<'a> {
    config: &'a TreeSearchConfig,
}

/// Reuses document lookups across every phase of one tree-search request.
struct DocumentTreeIndex<'a> {
    document: &'a Document,
    nodes: HashMap<&'a str, &'a Node>,
    parents: HashMap<String, Option<String>>,
    children: HashMap<String, Vec<String>>,
    depths: HashMap<String, u32>,
}

impl<'a> DocumentTreeIndex<'a> {
    /// Builds all structural maps in linear time for one document.
    fn new(document: &'a Document) -> Self {
        Self {
            document,
            nodes: document.build_node_map(),
            parents: document.build_parent_map(),
            children: document.build_children_map(),
            depths: document.build_depth_map(),
        }
    }

    /// Returns one uniquely identified node from the prebuilt lookup.
    fn node(&self, node_id: &str) -> Option<&'a Node> {
        self.nodes.get(node_id).copied()
    }
}

impl<'a> TreeSearcher<'a> {
    pub fn new(config: &'a TreeSearchConfig) -> Self {
        Self { config }
    }

    /// Run tree search across documents.
    pub fn search(
        &self,
        query: &str,
        documents: &[Document],
        initial_score_map: &HashMap<String, HashMap<String, f64>>,
    ) -> (Vec<PathResult>, Vec<FlatNode>) {
        let normalized_scores = initial_score_map
            .values()
            .flat_map(HashMap::values)
            .any(|score| !score.is_finite() || *score <= 0.0 || *score > 1.0)
            .then(|| normalize_initial_scores(initial_score_map));
        let initial_score_map = normalized_scores.as_ref().unwrap_or(initial_score_map);
        let plan = build_query_plan_with_mode(query, self.config.cjk_tokenizer);
        let mut all_paths: Vec<PathResult> = Vec::new();
        let mut all_walked_nodes: Vec<(String, String, f64, f64, usize)> = Vec::new();
        let indexes: Vec<DocumentTreeIndex<'_>> =
            documents.iter().map(DocumentTreeIndex::new).collect();

        // Sort documents by maximum initial score descending.
        let mut scored_docs: Vec<(f64, &DocumentTreeIndex<'_>, &HashMap<String, f64>)> = indexes
            .iter()
            .filter_map(|index| {
                let scores = initial_score_map.get(&index.document.doc_id)?;
                if scores.is_empty() {
                    return None;
                }
                let max_score = scores.values().cloned().fold(0.0_f64, f64::max);
                Some((max_score, index, scores))
            })
            .collect();
        scored_docs.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.document.doc_id.cmp(&right.1.document.doc_id))
        });
        scored_docs.truncate(self.config.top_k_docs);

        for (_, index, doc_scores) in &scored_docs {
            // IDF estimation for large documents
            let idf = if !plan.terms.is_empty() && index.nodes.len() > 20 && doc_scores.len() >= 5 {
                let corpus: Vec<&str> = index
                    .nodes
                    .values()
                    .map(|node| bounded_prefix(&node.text, self.config.max_node_chars))
                    .collect();
                Some(estimate_idf(&plan.terms, &corpus))
            } else {
                None
            };

            // Stage 1: Anchor retrieval
            let anchors = self.select_anchors(index, doc_scores, &plan, idf.as_ref());
            if anchors.is_empty() {
                continue;
            }

            // Stage 2: Tree walk
            let (doc_paths, walked_states) =
                self.tree_walk(index, &anchors, doc_scores, &plan, idf.as_ref());
            all_paths.extend(doc_paths);

            for state in &walked_states {
                let initial_score = doc_scores.get(&state.node_id).copied().unwrap_or(0.0);
                let combined = 0.3 * state.score + 0.7 * initial_score;
                all_walked_nodes.push((
                    index.document.doc_id.clone(),
                    state.node_id.clone(),
                    combined,
                    initial_score,
                    state.hop,
                ));
            }
        }

        // Stage 3: Select top paths globally
        all_paths.sort_by(compare_paths);
        all_paths.truncate(self.config.path_top_k);

        // Build flat nodes with reranking
        let document_indexes: HashMap<&str, &DocumentTreeIndex<'_>> = indexes
            .iter()
            .map(|index| (index.document.doc_id.as_str(), index))
            .collect();
        let flat_nodes = self.build_flat_nodes(
            &all_paths,
            &all_walked_nodes,
            &document_indexes,
            initial_score_map,
            &plan,
        );

        (all_paths, flat_nodes)
    }

    // ---------------------------------------------------------------
    // Stage 1: Anchor Retrieval
    // ---------------------------------------------------------------

    fn select_anchors(
        &self,
        index: &DocumentTreeIndex<'_>,
        doc_scores: &HashMap<String, f64>,
        plan: &QueryPlan,
        idf: Option<&HashMap<String, f64>>,
    ) -> Vec<SearchState> {
        let max_candidates = self.config.anchor_top_k.saturating_mul(3);
        let threshold = if doc_scores.len() > max_candidates {
            let mut scores: Vec<f64> = doc_scores.values().copied().collect();
            scores.sort_by(|left, right| right.total_cmp(left));
            scores
                .get(max_candidates.saturating_sub(1))
                .copied()
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let mut candidates: Vec<(f64, String, &Node)> = Vec::new();
        for (nid, &initial_score) in doc_scores {
            if initial_score < threshold {
                continue;
            }
            let node = match index.node(nid) {
                Some(n) => n,
                None => continue,
            };
            let depth = index.depths.get(nid).copied().unwrap_or(0);
            let node_text = bounded_prefix(&node.text, self.config.max_node_chars);
            let full_text = format!("{} {}", node.title, node_text);
            let a_score = score_anchor(
                initial_score,
                depth,
                check_title_match(&node.title, &plan.terms),
                check_phrase_match(&full_text, &plan.phrases),
                compute_term_overlap(node_text, &plan.terms, idf),
                6,
            );
            candidates.push((a_score, nid.clone(), node));
        }

        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let mut selected: Vec<SearchState> = Vec::new();
        let mut selected_paths: HashSet<String> = HashSet::new();

        let anchor_limit = self.config.anchor_top_k.min(self.config.max_anchor_per_doc);
        for (a_score, nid, _node) in &candidates {
            if selected.len() >= anchor_limit {
                break;
            }
            let path_to_root = path_to_root_via_map(nid, &index.parents);
            let path_key = path_to_root
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(">");
            if selected_paths.contains(&path_key) {
                continue;
            }
            selected_paths.insert(path_key);

            selected.push(SearchState {
                doc_id: index.document.doc_id.clone(),
                node_id: nid.clone(),
                score: *a_score,
                hop: 0,
                source: "anchor".into(),
                path: path_to_root,
                max_ancestor_score: maximum_ancestor_score(nid, &index.parents, doc_scores),
                anchor_node_id: nid.clone(),
            });
        }

        selected
    }

    // ---------------------------------------------------------------
    // Stage 2: Tree Walk (Best-First Search)
    // ---------------------------------------------------------------

    fn tree_walk(
        &self,
        index: &DocumentTreeIndex<'_>,
        anchors: &[SearchState],
        doc_scores: &HashMap<String, f64>,
        plan: &QueryPlan,
        idf: Option<&HashMap<String, f64>>,
    ) -> (Vec<PathResult>, Vec<SearchState>) {
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: BinaryHeap<SearchState> = BinaryHeap::new();
        let mut best_states: Vec<SearchState> = Vec::new();
        let mut expansion_count = 0;

        // Pre-cache term overlap for initially scored nodes.
        let mut overlap_cache: HashMap<String, f64> = HashMap::new();
        if !plan.terms.is_empty() {
            for nid in doc_scores.keys() {
                if let Some(node) = index.node(nid) {
                    overlap_cache.insert(
                        nid.clone(),
                        compute_term_overlap(
                            bounded_prefix(&node.text, self.config.max_node_chars),
                            &plan.terms,
                            idf,
                        ),
                    );
                }
            }
        }

        // Initialize frontier
        for anchor in anchors {
            frontier.push(anchor.clone());
        }

        while let Some(state) = frontier.pop() {
            if expansion_count >= self.config.max_expansions {
                break;
            }
            if visited.contains(&state.node_id) {
                continue;
            }
            visited.insert(state.node_id.clone());
            best_states.push(state.clone());
            expansion_count += 1;

            if state.score >= self.config.early_stop_score {
                break;
            }
            if state.score < self.config.min_frontier_score {
                continue;
            }
            if state.hop >= self.config.max_hops {
                continue;
            }

            // Expand neighbors
            let neighbors = get_neighbors(
                &state.node_id,
                &index.parents,
                &index.children,
                self.config.max_siblings,
            );
            for (nid, relation) in neighbors {
                if visited.contains(&nid) {
                    continue;
                }
                let node = match index.node(&nid) {
                    Some(n) => n,
                    None => continue,
                };
                let lexical = doc_scores.get(&nid).copied().unwrap_or(0.0);
                let overlap = overlap_cache.get(&nid).copied().unwrap_or_else(|| {
                    if !plan.terms.is_empty() {
                        let ov = compute_term_overlap(
                            bounded_prefix(&node.text, self.config.max_node_chars),
                            &plan.terms,
                            idf,
                        );
                        overlap_cache.insert(nid.clone(), ov);
                        ov
                    } else {
                        0.0
                    }
                });
                let new_max_anc = maximum_ancestor_score(&nid, &index.parents, doc_scores);
                let full_text = format!(
                    "{} {}",
                    node.title,
                    bounded_prefix(&node.text, self.config.max_node_chars)
                );
                let w_score = score_walk_node_with_input(WalkScoreInput {
                    lexical_score: lexical,
                    has_title_match: check_title_match(&node.title, &plan.terms),
                    has_phrase_match: check_phrase_match(&full_text, &plan.phrases),
                    body_term_overlap: overlap,
                    ancestor_support: new_max_anc,
                    hop: u32::try_from(state.hop.saturating_add(1)).unwrap_or(u32::MAX),
                    is_redundant: false,
                    max_hops: u32::try_from(self.config.max_hops).unwrap_or(u32::MAX),
                });

                let new_path = if relation == "child" {
                    let mut p = state.path.clone();
                    p.push(nid.clone());
                    p
                } else {
                    path_to_root_via_map(&nid, &index.parents)
                };

                frontier.push(SearchState {
                    doc_id: index.document.doc_id.clone(),
                    node_id: nid,
                    score: w_score,
                    hop: state.hop.saturating_add(1),
                    source: relation,
                    path: new_path,
                    max_ancestor_score: new_max_anc,
                    anchor_node_id: state.anchor_node_id.clone(),
                });
            }
        }

        let paths = self.states_to_paths(index, &mut best_states, doc_scores, plan);
        (paths, best_states)
    }

    fn states_to_paths(
        &self,
        index: &DocumentTreeIndex<'_>,
        states: &mut [SearchState],
        doc_scores: &HashMap<String, f64>,
        plan: &QueryPlan,
    ) -> Vec<PathResult> {
        states.sort_by(compare_states);
        let mut results: Vec<PathResult> = Vec::new();
        let mut seen_targets: HashSet<String> = HashSet::new();
        let max_to_process = self.config.path_top_k.saturating_mul(2);

        for state in states.iter() {
            if results.len() >= max_to_process {
                break;
            }
            if seen_targets.contains(&state.node_id) {
                continue;
            }
            seen_targets.insert(state.node_id.clone());

            let full_path = path_to_root_via_map(&state.node_id, &index.parents);
            let mut path_titles = Vec::new();
            let mut path_texts = Vec::new();
            let mut path_nodes = Vec::new();
            for pid in &full_path {
                if let Some(pnode) = index.node(pid) {
                    path_titles.push(pnode.title.clone());
                    path_texts
                        .push(bounded_prefix(&pnode.text, self.config.max_node_chars).to_string());
                    path_nodes.push(PathNode {
                        node_id: pid.clone(),
                        title: pnode.title.clone(),
                    });
                }
            }

            let p_score = score_path(
                state.score,
                &path_titles,
                &path_texts,
                &plan.terms,
                full_path.len(),
                doc_scores.get(&state.node_id).copied().unwrap_or(0.0),
                6,
            );

            let snippet = index
                .node(&state.node_id)
                .map(|node| {
                    bounded_prefix(&node.text, self.config.max_result_chars.min(300)).to_string()
                })
                .unwrap_or_default();

            results.push(PathResult {
                doc_id: index.document.doc_id.clone(),
                doc_name: index.document.doc_name.clone(),
                score: (p_score * 10000.0).round() / 10000.0,
                anchor_node_id: state.anchor_node_id.clone(),
                target_node_id: state.node_id.clone(),
                path: path_nodes,
                snippet,
            });
        }

        results.sort_by(compare_paths);
        results.truncate(self.config.path_top_k);
        results
    }

    // ---------------------------------------------------------------
    // Stage 3: Build flat nodes with reranking
    // ---------------------------------------------------------------

    fn build_flat_nodes(
        &self,
        _paths: &[PathResult],
        walked_nodes: &[(String, String, f64, f64, usize)],
        document_indexes: &HashMap<&str, &DocumentTreeIndex<'_>>,
        initial_score_map: &HashMap<String, HashMap<String, f64>>,
        plan: &QueryPlan,
    ) -> Vec<FlatNode> {
        let mut node_scores: HashMap<(String, String), f64> = HashMap::new();
        // 1. Base: host-provided initial scores.
        for (doc_id, doc_scores) in initial_score_map {
            for (nid, &initial_score) in doc_scores {
                node_scores.insert((doc_id.clone(), nid.clone()), initial_score);
            }
        }

        // 2. Generic section demotion + leaf preference (merged pass)
        for ((doc_id, nid), score) in node_scores.iter_mut() {
            let index = match document_indexes.get(doc_id.as_str()) {
                Some(index) => index,
                None => continue,
            };
            let node = match index.node(nid) {
                Some(n) => n,
                None => continue,
            };
            let depth = index.depths.get(nid.as_str()).copied().unwrap_or(0);

            // Generic section demotion
            if depth > 0 && is_generic_section(&node.title, depth) {
                let demote = if !plan.terms.is_empty() {
                    let base_title = node.title.to_lowercase();
                    !plan.terms.iter().any(|t| base_title.contains(t.as_str()))
                } else {
                    true
                };
                if demote {
                    *score *= 0.70;
                }
            }

            // Leaf preference
            if node.children.is_empty() && node.text.len() > 100 {
                *score *= 1.08;
            }
        }

        // 3. Walk boost
        for (doc_id, nid, combined_score, _initial_score, _hop) in walked_nodes {
            let key = (doc_id.clone(), nid.clone());
            if let Some(score) = node_scores.get_mut(&key) {
                *score += 0.15 * combined_score;
            }
        }

        // 4. Title match boost
        if !plan.terms.is_empty() {
            let keys: Vec<(String, String)> = node_scores.keys().cloned().collect();
            for key in &keys {
                let score = match node_scores.get(key) {
                    Some(&s) if s >= 0.05 => s,
                    _ => continue,
                };
                let index = match document_indexes.get(key.0.as_str()) {
                    Some(index) => index,
                    None => continue,
                };
                let node = match index.node(&key.1) {
                    Some(n) => n,
                    None => continue,
                };
                let title_lower = node.title.to_lowercase();
                let title_hits = plan
                    .terms
                    .iter()
                    .filter(|t| title_lower.contains(t.as_str()))
                    .count();
                if title_hits > 0 {
                    let title_overlap = title_hits as f64 / plan.terms.len() as f64;
                    let title_bonus = 0.15 * title_overlap * score.max(0.10);
                    node_scores.insert(key.clone(), score + title_bonus);
                }
            }
        }

        // Preserve relative ordering while restoring the public score range
        // after leaf, walk, and title boosts.
        let global_max = node_scores.values().copied().fold(0.0_f64, f64::max);
        if global_max > 1.0 {
            for score in node_scores.values_mut() {
                *score /= global_max;
            }
        }

        // Build flat node list
        let mut flat_nodes: Vec<FlatNode> = node_scores
            .into_iter()
            .filter_map(|((doc_id, nid), score)| {
                let index = document_indexes.get(doc_id.as_str())?;
                let node = index.node(&nid)?;
                Some(FlatNode {
                    node_id: nid,
                    doc_id: doc_id.clone(),
                    doc_name: index.document.doc_name.clone(),
                    title: node.title.clone(),
                    score: (score * 10000.0).round() / 10000.0,
                    text: bounded_prefix(&node.text, self.config.max_result_chars).to_string(),
                })
            })
            .collect();

        flat_nodes.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        flat_nodes
    }
}

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

/// Filters invalid direct walker inputs and globally calibrates raw scores.
fn normalize_initial_scores(
    scores: &HashMap<String, HashMap<String, f64>>,
) -> HashMap<String, HashMap<String, f64>> {
    let mut normalized: HashMap<String, HashMap<String, f64>> = scores
        .iter()
        .filter_map(|(document_id, node_scores)| {
            let valid: HashMap<String, f64> = node_scores
                .iter()
                .filter(|(_, score)| score.is_finite() && **score > 0.0)
                .map(|(node_id, score)| (node_id.clone(), *score))
                .collect();
            (!valid.is_empty()).then_some((document_id.clone(), valid))
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

/// Returns at most `max_characters` without splitting a UTF-8 code point.
fn bounded_prefix(text: &str, max_characters: usize) -> &str {
    match text.char_indices().nth(max_characters) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

fn path_to_root_via_map(
    node_id: &str,
    parent_map: &HashMap<String, Option<String>>,
) -> Vec<String> {
    let mut path = Vec::new();
    let mut visited = HashSet::new();
    let mut current = Some(node_id.to_string());
    while let Some(nid) = current {
        if !visited.insert(nid.clone()) {
            return Vec::new();
        }
        path.push(nid.clone());
        current = parent_map.get(&nid).and_then(|p| p.clone());
    }
    path.reverse();
    path
}

/// Returns the strongest score on the strict ancestor chain of one node.
fn maximum_ancestor_score(
    node_id: &str,
    parent_map: &HashMap<String, Option<String>>,
    scores: &HashMap<String, f64>,
) -> f64 {
    let mut maximum = 0.0_f64;
    let mut visited = HashSet::new();
    let mut current = parent_map.get(node_id).and_then(Clone::clone);
    while let Some(node_id) = current {
        if !visited.insert(node_id.clone()) {
            break;
        }
        maximum = maximum.max(scores.get(&node_id).copied().unwrap_or(0.0));
        current = parent_map.get(&node_id).and_then(Clone::clone);
    }
    maximum
}

/// Orders traversal states deterministically for stable ranking and tests.
fn compare_states(left: &SearchState, right: &SearchState) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.hop.cmp(&right.hop))
        .then_with(|| left.doc_id.cmp(&right.doc_id))
        .then_with(|| left.node_id.cmp(&right.node_id))
        .then_with(|| left.anchor_node_id.cmp(&right.anchor_node_id))
}

/// Orders path results by score and stable transport identities.
fn compare_paths(left: &PathResult, right: &PathResult) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.doc_id.cmp(&right.doc_id))
        .then_with(|| left.target_node_id.cmp(&right.target_node_id))
        .then_with(|| left.anchor_node_id.cmp(&right.anchor_node_id))
}

fn get_neighbors(
    node_id: &str,
    parent_map: &HashMap<String, Option<String>>,
    children_map: &HashMap<String, Vec<String>>,
    max_siblings: usize,
) -> Vec<(String, String)> {
    let mut neighbors = Vec::new();

    // Children
    if let Some(children) = children_map.get(node_id) {
        for cid in children {
            neighbors.push((cid.clone(), "child".into()));
        }
    }

    // Parent
    if let Some(Some(pid)) = parent_map.get(node_id) {
        neighbors.push((pid.clone(), "parent".into()));

        // Siblings (via parent's children)
        if let Some(siblings) = children_map.get(pid.as_str()) {
            let mut count = 0;
            for sid in siblings {
                if sid != node_id && count < max_siblings {
                    neighbors.push((sid.clone(), "sibling".into()));
                    count += 1;
                }
            }
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::normalize_initial_scores;
    use std::collections::HashMap;

    #[test]
    fn direct_walker_scores_are_filtered_and_globally_calibrated() {
        let scores = HashMap::from([
            (
                "a".to_string(),
                HashMap::from([
                    ("high".to_string(), 20.0),
                    ("low".to_string(), 10.0),
                    ("nan".to_string(), f64::NAN),
                    ("negative".to_string(), -1.0),
                ]),
            ),
            (
                "b".to_string(),
                HashMap::from([("infinite".to_string(), f64::INFINITY)]),
            ),
        ]);

        let normalized = normalize_initial_scores(&scores);
        assert_eq!(normalized["a"]["high"], 1.0);
        assert_eq!(normalized["a"]["low"], 0.5);
        assert!(!normalized["a"].contains_key("nan"));
        assert!(!normalized["a"].contains_key("negative"));
        assert!(!normalized.contains_key("b"));
    }
}
