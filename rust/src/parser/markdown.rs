//! Parses Markdown headings, body text, front matter, and exact line ranges.

use crate::document::{Document, Node, SourceType};
use anyhow::Result;
use pulldown_cmark::{Event, HeadingLevel, Options, Parser as CmarkParser, Tag, TagEnd};
use std::path::Path;

/// Parser for Markdown (`.md`, `.mdx`, `.markdown`) files.
///
/// Builds a hierarchical tree from headings: h2 nests under h1, h3 under h2, etc.
/// Text between headings becomes the parent heading's body text.
/// Front-matter (YAML between `---` fences) is captured in the root node's summary.
/// Code blocks are stored verbatim as part of the enclosing node's text.
pub struct MarkdownParser;

impl super::Parser for MarkdownParser {
    fn extensions(&self) -> &[&str] {
        &["md", "mdx", "markdown"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Markdown
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let doc_id = path.to_string_lossy().to_string();
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Markdown);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        // --- Extract front-matter ---
        let (front_matter, body) = extract_front_matter(content);

        // --- Parse markdown body ---
        let opts =
            Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
        let parser = CmarkParser::new_ext(body, opts).into_offset_iter();
        let line_starts = line_start_offsets(body);
        let body_byte_offset = content.len().saturating_sub(body.len());
        let body_line_offset = content[..body_byte_offset]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count() as u32;

        // Stack of (heading_level, Node).
        let mut stack: Vec<(u8, Node)> = Vec::new();
        let mut roots: Vec<Node> = Vec::new();

        let mut in_heading = false;
        let mut saw_heading = false;
        let mut heading_title = String::new();
        let mut heading_line = None;
        let mut current_text = String::new();
        let mut current_line_start = None;
        let mut current_line_end = None;
        let mut in_code_block = false;
        let mut code_lang;

        for (event, range) in parser {
            let event_line_start = body_line_offset + line_number(&line_starts, range.start);
            let event_line_end = body_line_offset
                + line_number(&line_starts, range.end.saturating_sub(1).max(range.start));
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    flush_text(
                        &mut stack,
                        &mut roots,
                        &mut current_text,
                        &mut current_line_start,
                        &mut current_line_end,
                    );
                    in_heading = true;
                    saw_heading = true;
                    heading_title.clear();
                    heading_line = Some(event_line_start);

                    let level_num = heading_level_to_u8(level);
                    collapse_stack(&mut stack, &mut roots, level_num);
                }
                Event::End(TagEnd::Heading(level)) => {
                    in_heading = false;
                    let level_num = heading_level_to_u8(level);
                    let mut node = Node::new("", heading_title.trim());
                    node.line_start = heading_line;
                    node.line_end = Some(event_line_end);
                    stack.push((level_num, node));
                }
                Event::Start(Tag::CodeBlock(kind)) => {
                    touch_lines(
                        &mut current_line_start,
                        &mut current_line_end,
                        event_line_start,
                        event_line_end,
                    );
                    in_code_block = true;
                    code_lang = match kind {
                        pulldown_cmark::CodeBlockKind::Fenced(lang) => lang.to_string(),
                        pulldown_cmark::CodeBlockKind::Indented => String::new(),
                    };
                    if !code_lang.is_empty() {
                        current_text.push_str(&format!("\n```{}\n", code_lang));
                    } else {
                        current_text.push_str("\n```\n");
                    }
                }
                Event::End(TagEnd::CodeBlock) => {
                    touch_lines(
                        &mut current_line_start,
                        &mut current_line_end,
                        event_line_start,
                        event_line_end,
                    );
                    in_code_block = false;
                    current_text.push_str("```\n");
                }
                Event::Text(text) => {
                    if in_heading {
                        heading_title.push_str(&text);
                    } else {
                        touch_lines(
                            &mut current_line_start,
                            &mut current_line_end,
                            event_line_start,
                            event_line_end,
                        );
                        current_text.push_str(&text);
                    }
                }
                Event::Code(code) => {
                    if in_heading {
                        heading_title.push('`');
                        heading_title.push_str(&code);
                        heading_title.push('`');
                    } else {
                        touch_lines(
                            &mut current_line_start,
                            &mut current_line_end,
                            event_line_start,
                            event_line_end,
                        );
                        current_text.push('`');
                        current_text.push_str(&code);
                        current_text.push('`');
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    if in_heading {
                        heading_title.push(' ');
                    } else {
                        touch_lines(
                            &mut current_line_start,
                            &mut current_line_end,
                            event_line_start,
                            event_line_end,
                        );
                        current_text.push('\n');
                    }
                }
                Event::End(TagEnd::Paragraph) if !in_code_block => {
                    current_text.push_str("\n\n");
                }
                _ => {}
            }
        }

        // Flush remaining text.
        flush_text(
            &mut stack,
            &mut roots,
            &mut current_text,
            &mut current_line_start,
            &mut current_line_end,
        );
        // Collapse entire stack.
        collapse_stack(&mut stack, &mut roots, 0);

        if !saw_heading {
            if let Some(root) = roots.first_mut() {
                root.title = file_name.clone();
            }
        }

        // Attach front-matter as summary on the first root node.
        if let Some(fm) = front_matter {
            if let Some(first) = roots.first_mut() {
                first.summary = fm;
            } else {
                let mut meta = Node::new("", &file_name);
                meta.summary = fm;
                roots.push(meta);
            }
        }

        // If no headings were found, create a single root from the entire content.
        if roots.is_empty() && !body.trim().is_empty() {
            let mut node = Node::new("", &file_name);
            node.text = body.trim().to_string();
            node.line_start = Some(1);
            roots.push(node);
        }

        doc.structure = roots;
        doc.assign_node_ids();
        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pop all stack entries with level >= `target_level`, attaching them as children
/// of the entry below. If the stack is exhausted, push to roots.
fn collapse_stack(stack: &mut Vec<(u8, Node)>, roots: &mut Vec<Node>, target_level: u8) {
    while let Some(&(lvl, _)) = stack.last() {
        if lvl >= target_level {
            let (_, node) = stack.pop().unwrap();
            if let Some(parent) = stack.last_mut() {
                parent.1.children.push(node);
            } else {
                roots.push(node);
            }
        } else {
            break;
        }
    }
}

/// Flush accumulated body text into the top-of-stack node or into roots.
fn flush_text(
    stack: &mut [(u8, Node)],
    roots: &mut Vec<Node>,
    text: &mut String,
    line_start: &mut Option<u32>,
    line_end: &mut Option<u32>,
) {
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        text.clear();
        *line_start = None;
        *line_end = None;
        return;
    }
    if let Some(top) = stack.last_mut() {
        if top.1.text.is_empty() {
            top.1.text = trimmed;
        } else {
            top.1.text.push_str("\n\n");
            top.1.text.push_str(&trimmed);
        }
        top.1.line_start = top.1.line_start.or(*line_start);
        top.1.line_end = (*line_end).or(top.1.line_end);
    } else {
        // Text before any heading — create an anonymous root node.
        let mut node = Node::new("", "");
        node.text = trimmed;
        node.line_start = *line_start;
        node.line_end = *line_end;
        roots.push(node);
    }
    text.clear();
    *line_start = None;
    *line_end = None;
}

/// Expands the current direct-text source range with one Markdown event.
fn touch_lines(
    current_start: &mut Option<u32>,
    current_end: &mut Option<u32>,
    event_start: u32,
    event_end: u32,
) {
    current_start.get_or_insert(event_start);
    *current_end = Some(event_end.max(current_end.unwrap_or(event_end)));
}

/// Returns byte offsets for every one-based source line in a UTF-8 body.
fn line_start_offsets(content: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(
            content
                .bytes()
                .enumerate()
                .filter_map(|(index, byte)| (byte == b'\n').then_some(index + 1)),
        )
        .collect()
}

/// Maps a byte offset to a one-based source line using binary search.
fn line_number(line_starts: &[usize], offset: usize) -> u32 {
    u32::try_from(line_starts.partition_point(|start| *start <= offset)).unwrap_or(u32::MAX)
}

fn heading_level_to_u8(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// Extract YAML front-matter delimited by `---` at the start of the file.
/// Returns `(Some(front_matter_content), remaining_body)` or `(None, full_content)`.
fn extract_front_matter(content: &str) -> (Option<String>, &str) {
    let trimmed = content.trim_start();
    let Some(opening_end) = trimmed.find('\n') else {
        return (None, content);
    };
    if trimmed[..opening_end].trim() != "---" {
        return (None, content);
    }
    let rest = &trimmed[opening_end + 1..];
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        let raw_line = line.strip_suffix('\n').unwrap_or(line);
        if raw_line.trim() == "---" {
            let fm = rest[..offset].trim().to_string();
            let body = rest[offset + line.len()..].trim_start_matches(['\r', '\n']);
            return (Some(fm), body);
        }
        offset += line.len();
    }
    // No exact closing fence — treat entire content as body.
    (None, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn parse(content: &str) -> Document {
        let parser = MarkdownParser;
        parser
            .parse(Path::new("test.md"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_empty() {
        let doc = parse("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let doc = parse("   \n\n  ");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_single_heading() {
        let doc = parse("# Hello\n\nWorld paragraph.");
        assert_eq!(doc.structure.len(), 1);
        assert_eq!(doc.structure[0].title, "Hello");
        assert!(doc.structure[0].text.contains("World paragraph"));
    }

    #[test]
    fn test_nested_headings() {
        let content = "# Top\n\nIntro\n\n## Section A\n\nText A\n\n### Sub A1\n\nDeep\n\n## Section B\n\nText B";
        let doc = parse(content);

        // Should have 1 root (h1).
        assert_eq!(doc.structure.len(), 1);
        let root = &doc.structure[0];
        assert_eq!(root.title, "Top");
        assert!(root.text.contains("Intro"));

        // Two h2 children.
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].title, "Section A");
        assert_eq!(root.children[1].title, "Section B");

        // h3 nested under first h2.
        assert_eq!(root.children[0].children.len(), 1);
        assert_eq!(root.children[0].children[0].title, "Sub A1");
    }

    #[test]
    fn test_front_matter() {
        let content = "---\ntitle: My Doc\nauthor: Alice\n---\n\n# Hello\n\nBody text";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].summary.contains("title: My Doc"));
        assert!(doc.structure[0].summary.contains("author: Alice"));
    }

    #[test]
    fn test_code_block_in_text() {
        let content = "# Code Example\n\nSome text\n\n```rust\nfn main() {}\n```\n\nMore text";
        let doc = parse(content);
        assert_eq!(doc.structure[0].title, "Code Example");
        assert!(doc.structure[0].text.contains("fn main()"));
        assert!(doc.structure[0].text.contains("```rust"));
    }

    #[test]
    fn test_no_headings() {
        let content = "Just a plain paragraph.\n\nAnother paragraph.";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 1);
        assert_eq!(doc.structure[0].title, "test.md");
        assert!(doc.structure[0].text.contains("Just a plain paragraph"));
        assert_eq!(doc.structure[0].line_start, Some(1));
        assert_eq!(doc.structure[0].line_end, Some(3));
    }

    #[test]
    fn test_node_ids_assigned() {
        let content = "# A\n\n## B\n\n## C";
        let doc = parse(content);
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[0].children[0].node_id, "1");
        assert_eq!(doc.structure[0].children[1].node_id, "2");
    }

    #[test]
    fn test_doc_id_and_source_type() {
        let doc = parse("# Hi");
        assert_eq!(doc.doc_id, "test.md");
        assert_eq!(doc.source_type, SourceType::Markdown);
    }

    #[test]
    fn test_multiple_h1() {
        let content = "# First\n\nText1\n\n# Second\n\nText2";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 2);
        assert_eq!(doc.structure[0].title, "First");
        assert_eq!(doc.structure[1].title, "Second");
    }

    #[test]
    fn test_text_before_heading() {
        let content = "Some intro text.\n\n# Title\n\nBody";
        let doc = parse(content);
        // Should have 2 roots: anonymous text node + heading node.
        assert_eq!(doc.structure.len(), 2);
        assert!(doc.structure[0].text.contains("Some intro text"));
        assert_eq!(doc.structure[1].title, "Title");
    }

    #[test]
    fn test_inline_code_in_heading() {
        let content = "# The `foo` function\n\nDetails here.";
        let doc = parse(content);
        assert_eq!(doc.structure[0].title, "The `foo` function");
    }

    #[test]
    fn test_extract_front_matter_no_fence() {
        let (fm, body) = extract_front_matter("Hello world");
        assert!(fm.is_none());
        assert_eq!(body, "Hello world");
    }

    #[test]
    fn test_extract_front_matter_unclosed() {
        let (fm, body) = extract_front_matter("---\ntitle: x\nno close");
        assert!(fm.is_none());
        assert_eq!(body, "---\ntitle: x\nno close");
    }

    #[test]
    fn test_deep_nesting_h1_to_h4() {
        let content = "# H1\n\n## H2\n\n### H3\n\n#### H4\n\nDeep";
        let doc = parse(content);
        let h1 = &doc.structure[0];
        assert_eq!(h1.children.len(), 1);
        let h2 = &h1.children[0];
        assert_eq!(h2.children.len(), 1);
        let h3 = &h2.children[0];
        assert_eq!(h3.children.len(), 1);
        let h4 = &h3.children[0];
        assert_eq!(h4.title, "H4");
        assert!(h4.text.contains("Deep"));
    }

    #[test]
    fn test_heading_and_body_line_ranges_include_front_matter_offset() {
        let content = "---\ntitle: Test\n---\n\n# 第一章\n正文\n\n## 小节\n详情";
        let doc = parse(content);
        assert_eq!(doc.structure[0].line_start, Some(5));
        assert_eq!(doc.structure[0].line_end, Some(6));
        assert_eq!(doc.structure[0].children[0].line_start, Some(8));
        assert_eq!(doc.structure[0].children[0].line_end, Some(9));
    }

    #[test]
    fn test_front_matter_requires_an_exact_closing_fence() {
        let content = "---\ntitle: Test\n---not-a-fence\n# Heading";
        let (front_matter, body) = extract_front_matter(content);
        assert!(front_matter.is_none());
        assert_eq!(body, content);
    }
}
