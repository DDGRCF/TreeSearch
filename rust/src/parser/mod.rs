//! Extensible parser registry whose built-in formats are selected with Cargo features.

#[cfg(feature = "parser-config")]
pub mod config_file;
#[cfg(feature = "parser-html")]
pub mod html;
#[cfg(feature = "parser-markdown")]
pub mod markdown;
#[cfg(feature = "parser-plaintext")]
pub mod plaintext;
#[cfg(feature = "parser-code")]
pub mod treesitter;

use crate::document::{Document, SourceType};
use anyhow::Result;
use std::path::Path;

/// Trait for document parsers.
pub trait Parser: Send + Sync {
    /// File extensions this parser handles (without dot, lowercase).
    fn extensions(&self) -> &[&str];

    /// Source type for parsed documents.
    fn source_type(&self) -> SourceType;

    /// Parse file content into a Document.
    /// Implementations must call `doc.assign_node_ids()` before returning.
    fn parse(&self, path: &Path, content: &str) -> Result<Document>;
}

/// Registry of all parsers. Routes files to the appropriate parser by extension.
pub struct ParserRegistry {
    parsers: Vec<Box<dyn Parser>>,
}

impl ParserRegistry {
    /// Creates a registry containing every parser enabled at compile time.
    pub fn new() -> Self {
        let parsers: Vec<Box<dyn Parser>> = Vec::new();
        #[cfg(feature = "parser-markdown")]
        let parsers = {
            let mut parsers = parsers;
            parsers.push(Box::new(markdown::MarkdownParser) as Box<dyn Parser>);
            parsers
        };
        #[cfg(feature = "parser-html")]
        let parsers = {
            let mut parsers = parsers;
            parsers.push(Box::new(html::HtmlParser) as Box<dyn Parser>);
            parsers
        };
        #[cfg(feature = "parser-config")]
        let parsers = {
            let mut parsers = parsers;
            parsers.push(Box::new(config_file::JsonParser) as Box<dyn Parser>);
            parsers.push(Box::new(config_file::YamlParser) as Box<dyn Parser>);
            parsers.push(Box::new(config_file::TomlParser) as Box<dyn Parser>);
            parsers
        };
        #[cfg(feature = "parser-code")]
        let parsers = {
            let mut parsers = parsers;
            parsers.push(Box::new(treesitter::TreeSitterParser) as Box<dyn Parser>);
            parsers
        };
        // Plaintext remains last because it is the most generic fallback.
        #[cfg(feature = "parser-plaintext")]
        let parsers = {
            let mut parsers = parsers;
            parsers.push(Box::new(plaintext::PlainTextParser) as Box<dyn Parser>);
            parsers
        };
        Self { parsers }
    }

    /// Creates an empty registry for applications that supply only custom parsers.
    pub fn empty() -> Self {
        Self {
            parsers: Vec::new(),
        }
    }

    /// Registers one application-defined parser after the built-in parsers.
    pub fn register<P>(&mut self, parser: P)
    where
        P: Parser + 'static,
    {
        self.parsers.push(Box::new(parser));
    }

    /// Find the parser for a given file extension (case-insensitive, without dot).
    fn find_parser_by_ext(&self, ext: &str) -> Option<&dyn Parser> {
        let ext_lower = ext.to_lowercase();
        self.parsers
            .iter()
            .find(|p| p.extensions().iter().any(|e| *e == ext_lower))
            .map(|p| p.as_ref())
    }

    /// Finds a parser from an extension or a known extensionless filename.
    fn find_parser_for_path(&self, path: &Path) -> Option<&dyn Parser> {
        path.extension()
            .and_then(|extension| extension.to_str())
            .or_else(|| path.file_name().and_then(|name| name.to_str()))
            .and_then(|discriminator| self.find_parser_by_ext(discriminator))
    }

    /// Parse a file from disk. Returns `Ok(None)` if no parser handles the extension.
    /// Sets `source_path` to the canonical absolute path.
    pub fn parse_file(&self, path: &Path) -> Result<Option<Document>> {
        let parser = match self.find_parser_for_path(path) {
            Some(p) => p,
            None => return Ok(None),
        };

        let content = std::fs::read_to_string(path)?;
        let abs_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut doc = parser.parse(path, &content)?;
        doc.source_path = abs_path.to_string_lossy().to_string();
        Ok(Some(doc))
    }

    /// Parse file content that has already been read. Returns `Ok(None)` if no
    /// parser handles the extension.
    pub fn parse_content(&self, path: &Path, content: &str) -> Result<Option<Document>> {
        let parser = match self.find_parser_for_path(path) {
            Some(p) => p,
            None => return Ok(None),
        };

        let mut doc = parser.parse(path, content)?;
        doc.source_path = path.to_string_lossy().to_string();
        Ok(Some(doc))
    }

    /// Check whether a file extension is supported by any registered parser.
    pub fn supports(&self, path: &Path) -> bool {
        self.find_parser_for_path(path).is_some()
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "parser-config", feature = "parser-markdown"))]
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_registry_supports() {
        let reg = ParserRegistry::new();
        #[cfg(feature = "parser-markdown")]
        assert!(reg.supports(Path::new("readme.md")));
        #[cfg(feature = "parser-html")]
        assert!(reg.supports(Path::new("index.html")));
        #[cfg(feature = "parser-code")]
        assert!(reg.supports(Path::new("main.rs")));
        #[cfg(feature = "parser-code")]
        assert!(reg.supports(Path::new("Dockerfile")));
        #[cfg(feature = "parser-code")]
        assert!(reg.supports(Path::new("Makefile")));
        #[cfg(feature = "parser-config")]
        assert!(reg.supports(Path::new("config.json")));
        #[cfg(feature = "parser-config")]
        assert!(reg.supports(Path::new("data.yaml")));
        #[cfg(feature = "parser-config")]
        assert!(reg.supports(Path::new("data.yml")));
        #[cfg(feature = "parser-config")]
        assert!(reg.supports(Path::new("Cargo.toml")));
        #[cfg(feature = "parser-plaintext")]
        assert!(reg.supports(Path::new("notes.txt")));
        assert!(!reg.supports(Path::new("image.png")));
        assert!(!reg.supports(Path::new("noext")));
    }

    #[cfg(feature = "parser-markdown")]
    #[test]
    fn test_registry_parse_file_markdown() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        writeln!(tmp, "# Hello\n\nWorld").unwrap();
        let reg = ParserRegistry::new();
        let doc = reg.parse_file(tmp.path()).unwrap().unwrap();
        assert_eq!(doc.source_type, SourceType::Markdown);
        assert!(!doc.structure.is_empty());
    }

    #[cfg(feature = "parser-config")]
    #[test]
    fn test_registry_parse_file_json() {
        let mut tmp = NamedTempFile::with_suffix(".json").unwrap();
        write!(tmp, r#"{{"key": "value"}}"#).unwrap();
        let reg = ParserRegistry::new();
        let doc = reg.parse_file(tmp.path()).unwrap().unwrap();
        assert_eq!(doc.source_type, SourceType::Json);
    }

    #[cfg(feature = "parser-markdown")]
    #[test]
    fn test_registry_parse_content() {
        let reg = ParserRegistry::new();
        let doc = reg
            .parse_content(Path::new("test.md"), "# Hello\n\nWorld")
            .unwrap()
            .unwrap();
        assert_eq!(doc.source_type, SourceType::Markdown);
        assert!(!doc.structure.is_empty());
    }

    #[test]
    fn test_registry_unsupported_extension() {
        let tmp = NamedTempFile::with_suffix(".png").unwrap();
        let reg = ParserRegistry::new();
        assert!(reg.parse_file(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn test_registry_no_extension() {
        let reg = ParserRegistry::new();
        assert!(!reg.supports(Path::new("Makefile_no_ext")));
    }

    #[test]
    fn test_registry_case_insensitive_ext() {
        let reg = ParserRegistry::new();
        #[cfg(feature = "parser-markdown")]
        assert!(reg.supports(Path::new("README.MD")));
        #[cfg(feature = "parser-html")]
        assert!(reg.supports(Path::new("page.HTML")));
        #[cfg(feature = "parser-config")]
        assert!(reg.supports(Path::new("data.JSON")));
    }
}
