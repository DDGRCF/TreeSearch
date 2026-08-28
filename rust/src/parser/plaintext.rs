//! Parses plain text paragraphs with UTF-8-safe titles and exact line ranges.

use crate::document::{Document, Node, SourceType};
use anyhow::Result;
use std::path::Path;

/// Parser for plain-text files (`.txt`, `.text`, `.log`, `.csv`).
///
/// Splits content by double newlines (paragraphs). Each paragraph becomes a flat
/// node with the first line (truncated to 80 chars) as the title and the full
/// paragraph text as the body.
pub struct PlainTextParser;

/// Maximum character length for a paragraph title.
const MAX_TITLE_LEN: usize = 80;

impl super::Parser for PlainTextParser {
    fn extensions(&self) -> &[&str] {
        &["txt", "text", "log", "csv"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Text
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let doc_id = path.to_string_lossy().to_string();
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Text);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let paragraphs = split_paragraphs(content);

        for paragraph in paragraphs {
            let title = make_title(&paragraph.text);

            let mut node = Node::new("", &title);
            node.text = paragraph.text;
            node.line_start = Some(paragraph.line_start);
            node.line_end = Some(paragraph.line_end);

            doc.structure.push(node);
        }

        doc.assign_node_ids();
        Ok(doc)
    }
}

/// One normalized paragraph with exact one-based source line bounds.
struct Paragraph {
    text: String,
    line_start: u32,
    line_end: u32,
}

/// Splits content on blank lines while preserving exact source line bounds.
fn split_paragraphs(content: &str) -> Vec<Paragraph> {
    /// Flushes an accumulated non-empty paragraph into the result list.
    fn flush(
        lines: &mut Vec<String>,
        line_start: &mut Option<u32>,
        line_end: u32,
        result: &mut Vec<Paragraph>,
    ) {
        let Some(start) = line_start.take() else {
            return;
        };
        result.push(Paragraph {
            text: lines.join("\n").trim().to_string(),
            line_start: start,
            line_end,
        });
        lines.clear();
    }

    let mut result = Vec::new();
    let mut paragraph_lines = Vec::new();
    let mut paragraph_start = None;
    let mut previous_content_line = 0;
    for (index, raw_line) in content.lines().enumerate() {
        let line_number = u32::try_from(index.saturating_add(1)).unwrap_or(u32::MAX);
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            flush(
                &mut paragraph_lines,
                &mut paragraph_start,
                previous_content_line,
                &mut result,
            );
            continue;
        }
        paragraph_start.get_or_insert(line_number);
        previous_content_line = line_number;
        paragraph_lines.push(line.to_string());
    }
    flush(
        &mut paragraph_lines,
        &mut paragraph_start,
        previous_content_line,
        &mut result,
    );
    result
}

/// Create a title from the first line of a paragraph, truncated to `MAX_TITLE_LEN`.
fn make_title(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("");
    let trimmed = first_line.trim();
    if trimmed.chars().count() <= MAX_TITLE_LEN {
        trimmed.to_string()
    } else {
        format!(
            "{}...",
            trimmed.chars().take(MAX_TITLE_LEN).collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn parse(content: &str) -> Document {
        let parser = PlainTextParser;
        parser
            .parse(Path::new("notes.txt"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_empty() {
        let doc = parse("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let doc = parse("   \n\n  \n  ");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_single_paragraph() {
        let doc = parse("Hello world.\nThis is a test.");
        assert_eq!(doc.structure.len(), 1);
        assert_eq!(doc.structure[0].title, "Hello world.");
        assert!(doc.structure[0].text.contains("This is a test"));
    }

    #[test]
    fn test_multiple_paragraphs() {
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 3);
        assert_eq!(doc.structure[0].title, "First paragraph.");
        assert_eq!(doc.structure[1].title, "Second paragraph.");
        assert_eq!(doc.structure[2].title, "Third paragraph.");
    }

    #[test]
    fn test_title_truncation() {
        let long_line = "A".repeat(120);
        let doc = parse(&long_line);
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].title.len() <= MAX_TITLE_LEN + 3);
        assert!(doc.structure[0].title.ends_with("..."));
    }

    #[test]
    fn test_line_numbers() {
        let content = "Para one line1\nPara one line2\n\nPara two.";
        let doc = parse(content);
        assert_eq!(doc.structure[0].line_start, Some(1));
        assert_eq!(doc.structure[0].line_end, Some(2));
    }

    #[test]
    fn test_line_numbers_account_for_leading_and_repeated_blank_lines() {
        let content = "\n\nFirst\nline\n\n\n\nSecond\r\nthird\r\n";
        let doc = parse(content);
        assert_eq!(doc.structure[0].line_start, Some(3));
        assert_eq!(doc.structure[0].line_end, Some(4));
        assert_eq!(doc.structure[1].line_start, Some(8));
        assert_eq!(doc.structure[1].line_end, Some(9));
    }

    #[test]
    fn test_node_ids_assigned() {
        let content = "A\n\nB\n\nC";
        let doc = parse(content);
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[1].node_id, "1");
        assert_eq!(doc.structure[2].node_id, "2");
    }

    #[test]
    fn test_source_type() {
        let doc = parse("hello");
        assert_eq!(doc.source_type, SourceType::Text);
        assert_eq!(doc.doc_id, "notes.txt");
    }

    #[test]
    fn test_multiple_blank_lines() {
        let content = "First\n\n\n\nSecond";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 2);
    }

    #[test]
    fn test_flat_structure() {
        let content = "A\n\nB\n\nC";
        let doc = parse(content);
        for node in &doc.structure {
            assert!(node.children.is_empty());
        }
    }

    #[test]
    fn test_unicode_title_truncation() {
        // Each CJK character is 3 bytes in UTF-8.
        let long_cjk = "你".repeat(100);
        let doc = parse(&long_cjk);
        // Should truncate without panicking (char boundary safe).
        assert!(doc.structure[0].title.ends_with("..."));
        assert_eq!(doc.structure[0].title.chars().count(), MAX_TITLE_LEN + 3);
    }
}
