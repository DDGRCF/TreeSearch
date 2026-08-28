# TreeSearch Rust Library and CLI

`rtreesearch` is a fast, structure-aware document search library and optional CLI built with Rust.
It indexes files into a local SQLite FTS5 database and searches by document
structure instead of chunking text into arbitrary fragments.

The default features preserve the complete `ts` executable. Embedding applications
can disable defaults and compile only the parsers they actually use.

## Library embedding

For an application that constructs authorized documents itself and only parses
Markdown/plain text, use the dependency-minimal profile:

```toml
treesearch = { package = "rtreesearch", version = "1.1.2", default-features = false, features = ["parser-markdown", "parser-plaintext"] }
```

This profile keeps the in-memory flat/tree ranking engine, document model,
Markdown hierarchy, plain-text hierarchy, scorer, and dependency-free CJK
bigram/character tokenizers. The host supplies its authorized candidate scores
to `engine::candidate_search::search_with_scores_strict`; TreeSearch performs
mode routing, traversal, path aggregation, and final ranking. It does not
compile SQLite/FTS5, the CLI, directory crawler, progress rendering, output
formatters, HTML/config/code parsers, or jieba.

Available opt-in features:

| Feature | Capability |
| --- | --- |
| `cli` (default) | Full backward-compatible `ts` binary and all built-ins |
| `directory-indexer` | `.gitignore`-aware parallel directory crawling |
| `progress` | Indexing progress bars |
| `output` | JSON/plain/TTY renderers |
| `sqlite-fts` | Optional rusqlite/libsqlite3-sys adapter linked to a host/system SQLite |
| `sqlite-bundled` | `sqlite-fts` plus a bundled SQLite build; used by the default CLI |
| `cjk-jieba` | Jieba segmentation and custom dictionaries |
| `parser-markdown` | Markdown heading hierarchy |
| `parser-plaintext` | Plain-text paragraph hierarchy |
| `parser-html` | Browser-grade HTML parsing |
| `parser-config` | JSON, YAML, and TOML parsing |
| `parser-code` | Source-code structure parsing |
| `parsers-all` | Every built-in parser |

`ParserRegistry::empty()` plus `ParserRegistry::register()` lets a host add
domain-specific parsers without enabling unrelated built-ins.

The default `cli` feature includes `sqlite-bundled`, so existing CLI installs
remain self-contained. Library consumers should make an explicit choice:

- omit both SQLite features for pure in-memory candidate reranking;
- enable `sqlite-fts` to share the host/system `sqlite3` link;
- enable `sqlite-bundled` only for a self-contained SQLite build.

The minimal Markdown/plain-text profile contains neither `rusqlite` nor
`libsqlite3-sys`. `rusqlite 0.37` and SQLx 0.9 both resolve
`libsqlite3-sys 0.35`, but disabling TreeSearch's SQLite adapter is still the
recommended choice when the host does not use local FTS5.

## Candidate-search API

Hosts that already perform authorization and candidate retrieval should use the
strict API:

```rust
use treesearch::config::{SearchMode, TreeSearchConfig};
use treesearch::engine::candidate_search::{
    ScoredNodeCandidate, search_with_candidates_strict,
};

let mut config = TreeSearchConfig::default();
config.search_mode = SearchMode::Auto;
let outcome = search_with_candidates_strict(
    "token validation",
    &documents,
    &[ScoredNodeCandidate::new("guide", "auth", 0.92)],
    &config,
)?;
```

Strict search rejects empty queries, duplicate document/node identities,
unknown candidates, non-finite/out-of-range scores, invalid source ranges, and
invalid frontier thresholds. Scores must be globally calibrated in `0.0..=1.0`.
The lenient API ignores invalid candidates and, when raw positive scores exceed
`1.0`, divides the accepted set by its global maximum without destroying
cross-document ordering.

All result ordering has explicit `(score, doc_id, node_id)` tie-breaks.
`top_k_docs` limits distinct returned documents and `max_nodes_per_doc` limits
nodes from each selected document. Tree paths report the actual retrieval
anchor and UTF-8-safe snippets. See [API.md](API.md) for the full contract.

Resource policies are active rather than documentary: `max_concurrency` builds
a request-local Rayon pool for directory parsing, `max_node_chars` bounds text
examined by scoring/FTS indexing, and `max_result_chars` bounds returned node
text on UTF-8 character boundaries. SQLite transfers at most 5,000 raw matches
into memory before bounded TreeSearch reranking.

## Install

**Homebrew (macOS / Linux)**

```bash
brew tap shibing624/tap
brew install treesearch
ts --help
```

**Cargo**

```bash
cargo install rtreesearch
ts --help
```

**Prebuilt binaries**

Download from GitHub Releases:

- Linux: `x86_64-unknown-linux-gnu`
- macOS Intel: `x86_64-apple-darwin`
- macOS Apple Silicon: `aarch64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`

Release page: <https://github.com/shibing624/TreeSearch/releases>

## Why TreeSearch

- No embeddings
- No vector database
- No chunk splitting
- SQLite FTS5 with persistent local indexes
- Structure-aware retrieval for Markdown, text, code, HTML, and more

## Quick Start

Search the current directory with the default auto mode:

```bash
ts "How does auth work?" .
```

Build the index explicitly:

```bash
ts index .
```

Inspect index stats:

```bash
ts stats .
```

## Wildcard Queries

`ts` supports a narrow set of wildcard shortcuts:

- `auth*`: prefix match
- `*auth*`: contains-style regex match
- other wildcard shapes currently fall back to regular query parsing

For explicit control:

- `ts --regex "o?auth" .` treats the query as a raw regex
- `ts search --regex "o?auth"` runs indexed search in regex mode
- `ts --fts-expression "auth*" .` passes a raw FTS5 expression
- `ts search --fts-expression "auth*"` runs indexed search with raw FTS5 syntax
- Invalid regex patterns raise an explicit error instead of silently returning no results

Examples:

```bash
ts "auth*" .
ts "*auth*" .
ts --regex "o?auth" .
ts --fts-expression "auth*" .
```

## Search Modes

`ts` supports three search modes:

- `auto`: default mode, automatically selects `flat` or `tree`
- `flat`: force FTS-style flat retrieval
- `tree`: force tree traversal retrieval

Examples:

```bash
ts "query" .               # auto (default)
ts "query" . --mode flat   # force flat
ts "query" . --mode tree   # force tree
```

In `auto` mode, TreeSearch uses the same three-layer decision logic as the
Python version:

1. Source type mapping: file types that benefit from tree search are marked explicitly.
2. Depth verification: only documents with real structure depth are treated as hierarchical.
3. Ratio threshold: if enough indexed documents benefit from tree mode, use `tree`; otherwise use `flat`.

## Commands

```text
ts [OPTIONS] [QUERY] [PATH]
ts search <QUERY> [PATH]
ts index [PATH]
ts stats [PATH]
```

Useful options:

- `--mode auto|flat|tree`
- `--format tty|json|plain`
- `--json`
- `--follow`
- `-n, --max-results <N>`

## Output Formats

- `tty`: colored terminal output
- `json`: machine-readable JSON output
- `plain`: plain text output

## Documentation

- Project homepage: <https://github.com/DDGRCF/TreeSearch>
- API docs: <https://docs.rs/treesearch>
- Embedding contract: [API.md](API.md)

## License

Apache-2.0
