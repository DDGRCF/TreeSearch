# Rust embedding API

## Layering

`engine::candidate_search` is the library-first path. The host owns access
control and retrieval; TreeSearch owns mode routing, bounded traversal, path
scoring, deterministic reranking, and transport-ready results. It performs no
I/O and has no SQLite dependency when default features are disabled.

`engine::fts` and `engine::search` are optional SQLite adapters. They convert
FTS5 BM25 matches into the same globally normalized `CandidateScoreMap` used by
the in-memory engine. Flat and tree modes therefore share limits, depth,
breadcrumb, tie-break, and result semantics.

## Structural invariants

Call `search_with_candidates_strict` or `search_with_scores_strict` at trust
boundaries. Strict calls require:

- a non-empty query;
- unique, non-empty document IDs;
- unique, non-empty node IDs inside every document;
- `line_end >= line_start` when both are present;
- candidate IDs that exist in the supplied documents;
- finite scores in `0.0..=1.0`;
- finite frontier thresholds in `0.0..=1.0`, with early-stop not below the
  minimum frontier score.

Owned `Node` values cannot contain pointer cycles. Lookup, flattening, depth,
parent/child maps, ID assignment, destruction, ancestor propagation, and regex
traversal are iterative so deep host-built trees do not recurse through the
search stack. Duplicate IDs are rejected because every ranking identity is
`(doc_id, node_id)`.

## Score calibration

Strict APIs treat scores as already calibrated, with larger values meaning more
relevant. Lenient APIs accept arbitrary finite positive values: if the maximum
is above `1.0`, all accepted values are divided by that one global maximum.
They are never normalized independently per document, because per-document
normalization would make every document's best node tie at `1.0`.

SQLite BM25 scores use the same global normalization. Ancestor propagation is
then applied deterministically and, if necessary, the whole propagated set is
renormalized once. Invalid decay is treated as zero; valid decay is clamped to
`0.0..=1.0`.

## Mode semantics

- `Flat` returns supplied candidates directly after stable ordering and limits.
- `Tree` uses candidates as anchors, traverses parent/child/sibling edges, and
  adds scored root-to-target paths.
- `Auto` selects `Tree` when at least 30% of supplied documents are Markdown,
  JSON, YAML, TOML, or HTML with a one-based depth of at least two; otherwise it
  selects `Flat`.

The final ranking uses descending score, then ascending document and node IDs.
`top_k_docs` and `max_nodes_per_doc` are independent limits. Zero for either
returns no node results.

`max_node_chars` bounds the direct body text used for IDF, overlap, phrase
scoring, and optional FTS indexing. `max_result_chars` bounds copied result text
and path snippets without splitting UTF-8. `max_concurrency` controls a
request-local directory-parser pool and is clamped to the number of files.

## Cargo feature recipes

Pure in-memory Markdown/plain-text embedding:

```toml
treesearch = { package = "rtreesearch", version = "1.1.2", default-features = false, features = ["parser-markdown", "parser-plaintext"] }
```

SQLite using a host/system library:

```toml
treesearch = { package = "rtreesearch", version = "1.1.2", default-features = false, features = ["sqlite-fts"] }
```

Self-contained SQLite:

```toml
treesearch = { package = "rtreesearch", version = "1.1.2", default-features = false, features = ["sqlite-bundled"] }
```

`sqlite-fts` requires a linkable system `sqlite3` with FTS5 enabled.
`sqlite-bundled` compiles SQLite and must not be enabled accidentally in a host
that already controls its native SQLite build.
