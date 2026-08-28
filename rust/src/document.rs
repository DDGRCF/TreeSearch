//! Defines serialized document trees and iterative structural lookup helpers.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a node within a document.
pub type NodeId = String;

/// Source type classification for documents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Code,
    Markdown,
    Html,
    Text,
    Json,
    Yaml,
    Toml,
    #[serde(other)]
    Unknown,
}

impl SourceType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Code => "code",
            Self::Markdown => "markdown",
            Self::Html => "html",
            Self::Text => "text",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Toml => "toml",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "md" | "mdx" | "markdown" => Self::Markdown,
            "html" | "htm" => Self::Html,
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "toml" => Self::Toml,
            "txt" | "text" | "log" | "csv" => Self::Text,
            "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "java" | "c" | "cpp" | "h"
            | "hpp" | "cs" | "rb" | "php" | "swift" | "kt" | "scala" | "sh" | "bash" | "zsh"
            | "fish" | "lua" | "r" | "m" | "mm" | "pl" | "pm" | "ex" | "exs" | "erl" | "hs"
            | "ml" | "mli" | "clj" | "cljs" | "el" | "vim" | "sql" | "graphql" | "proto" | "tf"
            | "hcl" | "zig" | "nim" | "v" | "d" | "dart" | "cmake" | "makefile" | "dockerfile"
            | "css" | "scss" | "sass" | "less" | "vue" | "svelte" => Self::Code,
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single node in the document tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub node_id: NodeId,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub line_start: Option<u32>,
    #[serde(default)]
    pub line_end: Option<u32>,
    #[serde(default)]
    pub children: Vec<Node>,
}

impl Node {
    /// Creates a node with empty content and no children.
    pub fn new(node_id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            title: title.into(),
            summary: String::new(),
            text: String::new(),
            line_start: None,
            line_end: None,
            children: Vec::new(),
        }
    }

    /// Flatten this node and all descendants into a vec.
    pub fn flatten(&self) -> Vec<&Node> {
        let mut result = Vec::new();
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            result.push(node);
            stack.extend(node.children.iter().rev());
        }
        result
    }
}

impl Drop for Node {
    /// Drains descendants iteratively so destroying host-built deep trees cannot overflow the stack.
    fn drop(&mut self) {
        let mut pending = std::mem::take(&mut self.children);
        while let Some(mut node) = pending.pop() {
            pending.append(&mut node.children);
        }
    }
}

/// A structural invariant violation in one owned document tree.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DocumentStructureError {
    #[error("document ID cannot be empty")]
    EmptyDocumentId,
    #[error("node ID cannot be empty")]
    EmptyNodeId,
    #[error("node ID `{node_id}` occurs more than once")]
    DuplicateNodeId { node_id: String },
    #[error("node `{node_id}` has line_end before line_start")]
    InvalidLineRange { node_id: String },
}

/// A parsed document with its tree structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub doc_id: String,
    pub doc_name: String,
    pub source_type: SourceType,
    #[serde(default)]
    pub doc_description: String,
    #[serde(default)]
    pub source_path: String,
    pub structure: Vec<Node>,
}

impl Document {
    /// Creates an empty document with stable transport metadata.
    pub fn new(
        doc_id: impl Into<String>,
        doc_name: impl Into<String>,
        source_type: SourceType,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            doc_name: doc_name.into(),
            source_type,
            doc_description: String::new(),
            source_path: String::new(),
            structure: Vec::new(),
        }
    }

    /// Flatten all nodes in the document tree.
    pub fn flatten_nodes(&self) -> Vec<&Node> {
        let mut nodes = Vec::new();
        let mut stack: Vec<&Node> = self.structure.iter().rev().collect();
        while let Some(node) = stack.pop() {
            nodes.push(node);
            stack.extend(node.children.iter().rev());
        }
        nodes
    }

    /// Builds a borrowed node lookup table for repeated search operations.
    ///
    /// Call [`Document::validate_structure`] first when identities come from an
    /// untrusted host; duplicate IDs would otherwise keep the last node.
    pub fn build_node_map(&self) -> HashMap<&str, &Node> {
        self.flatten_nodes()
            .into_iter()
            .map(|node| (node.node_id.as_str(), node))
            .collect()
    }

    /// Validates identities and source ranges required by search maps.
    pub fn validate_structure(&self) -> Result<(), DocumentStructureError> {
        if self.doc_id.trim().is_empty() {
            return Err(DocumentStructureError::EmptyDocumentId);
        }
        let mut node_ids = HashSet::new();
        for node in self.flatten_nodes() {
            if node.node_id.trim().is_empty() {
                return Err(DocumentStructureError::EmptyNodeId);
            }
            if !node_ids.insert(node.node_id.as_str()) {
                return Err(DocumentStructureError::DuplicateNodeId {
                    node_id: node.node_id.clone(),
                });
            }
            if matches!((node.line_start, node.line_end), (Some(start), Some(end)) if end < start) {
                return Err(DocumentStructureError::InvalidLineRange {
                    node_id: node.node_id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Build parent map: node_id -> parent_node_id (None for roots).
    pub fn build_parent_map(&self) -> HashMap<String, Option<String>> {
        let mut map = HashMap::new();
        let mut stack: Vec<(&Node, Option<&str>)> = self
            .structure
            .iter()
            .rev()
            .map(|root| (root, None))
            .collect();
        while let Some((node, parent_id)) = stack.pop() {
            map.insert(node.node_id.clone(), parent_id.map(String::from));
            stack.extend(
                node.children
                    .iter()
                    .rev()
                    .map(|child| (child, Some(node.node_id.as_str()))),
            );
        }
        map
    }

    /// Build depth map: node_id -> depth (0 for roots).
    pub fn build_depth_map(&self) -> HashMap<String, u32> {
        let mut map = HashMap::new();
        let mut stack: Vec<(&Node, u32)> =
            self.structure.iter().rev().map(|root| (root, 0)).collect();
        while let Some((node, depth)) = stack.pop() {
            map.insert(node.node_id.clone(), depth);
            stack.extend(
                node.children
                    .iter()
                    .rev()
                    .map(|child| (child, depth.saturating_add(1))),
            );
        }
        map
    }

    /// Returns the maximum one-based tree depth, or zero for an empty document.
    pub fn max_depth(&self) -> u32 {
        let mut maximum = 0_u32;
        let mut stack: Vec<(&Node, u32)> =
            self.structure.iter().rev().map(|root| (root, 1)).collect();
        while let Some((node, depth)) = stack.pop() {
            maximum = maximum.max(depth);
            stack.extend(
                node.children
                    .iter()
                    .rev()
                    .map(|child| (child, depth.saturating_add(1))),
            );
        }
        maximum
    }

    /// Build children map: node_id -> list of child node_ids.
    pub fn build_children_map(&self) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        let mut stack: Vec<&Node> = self.structure.iter().rev().collect();
        while let Some(node) = stack.pop() {
            let child_ids: Vec<String> = node.children.iter().map(|c| c.node_id.clone()).collect();
            if !child_ids.is_empty() {
                map.insert(node.node_id.clone(), child_ids);
            }
            stack.extend(node.children.iter().rev());
        }
        map
    }

    /// Find a node by id.
    pub fn find_node(&self, node_id: &str) -> Option<&Node> {
        let mut stack: Vec<&Node> = self.structure.iter().rev().collect();
        while let Some(node) = stack.pop() {
            if node.node_id == node_id {
                return Some(node);
            }
            stack.extend(node.children.iter().rev());
        }
        None
    }

    /// Get path from root to a node (list of node_ids, root first).
    pub fn path_to_node(&self, node_id: &str) -> Vec<String> {
        let parent_map = self.build_parent_map();
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

    /// Assign sequential node IDs to all nodes.
    pub fn assign_node_ids(&mut self) {
        let mut counter = 0u32;
        let mut stack: Vec<&mut Node> = self.structure.iter_mut().rev().collect();
        while let Some(node) = stack.pop() {
            node.node_id = counter.to_string();
            counter = counter.saturating_add(1);
            stack.extend(node.children.iter_mut().rev());
        }
    }
}

/// Search result from FTS5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub node_id: NodeId,
    pub doc_id: String,
    pub doc_name: String,
    pub title: String,
    pub summary: String,
    pub text: String,
    pub source_type: String,
    pub source_path: String,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub score: f64,
    pub depth: u32,
    /// Breadcrumb path from root to this node.
    #[serde(default)]
    pub breadcrumb: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> Document {
        let mut doc = Document::new("test", "test.rs", SourceType::Code);
        let mut root = Node::new("0", "Root");
        let mut child1 = Node::new("1", "Child 1");
        child1.text = "some text".into();
        let child2 = Node::new("2", "Child 2");
        child1.children.push(Node::new("3", "Grandchild"));
        root.children.push(child1);
        root.children.push(child2);
        doc.structure.push(root);
        doc
    }

    #[test]
    fn test_flatten_nodes() {
        let doc = sample_doc();
        let flat = doc.flatten_nodes();
        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0].title, "Root");
        assert_eq!(flat[1].title, "Child 1");
        assert_eq!(flat[2].title, "Grandchild");
        assert_eq!(flat[3].title, "Child 2");
    }

    #[test]
    fn test_parent_map() {
        let doc = sample_doc();
        let pm = doc.build_parent_map();
        assert_eq!(pm["0"], None);
        assert_eq!(pm["1"], Some("0".into()));
        assert_eq!(pm["3"], Some("1".into()));
    }

    #[test]
    fn test_depth_map() {
        let doc = sample_doc();
        let dm = doc.build_depth_map();
        assert_eq!(dm["0"], 0);
        assert_eq!(dm["1"], 1);
        assert_eq!(dm["3"], 2);
    }

    #[test]
    fn test_find_node() {
        let doc = sample_doc();
        assert!(doc.find_node("3").is_some());
        assert_eq!(doc.find_node("3").unwrap().title, "Grandchild");
        assert!(doc.find_node("999").is_none());
    }

    #[test]
    fn test_path_to_node() {
        let doc = sample_doc();
        let path = doc.path_to_node("3");
        assert_eq!(path, vec!["0", "1", "3"]);
    }

    #[test]
    fn test_assign_node_ids() {
        let mut doc = Document::new("test", "test.rs", SourceType::Code);
        let mut root = Node::new("", "Root");
        root.children.push(Node::new("", "A"));
        root.children.push(Node::new("", "B"));
        doc.structure.push(root);
        doc.assign_node_ids();
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[0].children[0].node_id, "1");
        assert_eq!(doc.structure[0].children[1].node_id, "2");
    }

    #[test]
    fn validation_rejects_duplicate_ids_and_invalid_ranges() {
        let mut duplicate = sample_doc();
        duplicate.structure[0].children[1].node_id = "1".into();
        assert_eq!(
            duplicate.validate_structure(),
            Err(DocumentStructureError::DuplicateNodeId {
                node_id: "1".into()
            })
        );

        let mut invalid_range = sample_doc();
        invalid_range.structure[0].line_start = Some(4);
        invalid_range.structure[0].line_end = Some(3);
        assert_eq!(
            invalid_range.validate_structure(),
            Err(DocumentStructureError::InvalidLineRange {
                node_id: "0".into()
            })
        );
    }

    #[test]
    fn iterative_lookups_handle_a_deep_tree() {
        let mut node = Node::new("4095", "leaf");
        for depth in (0..4095).rev() {
            let mut parent = Node::new(depth.to_string(), "node");
            parent.children.push(node);
            node = parent;
        }
        let mut document = Document::new("deep", "Deep", SourceType::Markdown);
        document.structure.push(node);

        assert_eq!(document.flatten_nodes().len(), 4096);
        assert_eq!(
            document.find_node("4095").map(|node| node.title.as_str()),
            Some("leaf")
        );
        assert_eq!(document.max_depth(), 4096);
        assert_eq!(document.path_to_node("4095").len(), 4096);
    }

    #[test]
    fn test_source_type_from_extension() {
        assert_eq!(SourceType::from_extension("rs"), SourceType::Code);
        assert_eq!(SourceType::from_extension("md"), SourceType::Markdown);
        assert_eq!(SourceType::from_extension("html"), SourceType::Html);
        assert_eq!(SourceType::from_extension("json"), SourceType::Json);
        assert_eq!(SourceType::from_extension("xyz"), SourceType::Unknown);
    }
}
