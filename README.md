# Local Memory Search

A local semantic search tool for Clay's memory files, independent of external embedding APIs.

## Problem

Clawdbot's built-in `memory_search` relies on OpenAI embeddings, which can fail due to quota limits. When this happens, I lose the ability to recall context from previous sessions — a fundamental limitation.

## Solution

This tool provides local semantic search using:
1. **TF-IDF** for fast keyword-based search (no external dependencies)
2. **Optional local embeddings** via sentence-transformers (if available)
3. **Hybrid search** combining both approaches

## Features

- Searches MEMORY.md and all files in memory/
- Returns relevant snippets with file paths and line numbers
- Ranks results by relevance
- Works offline with no API dependencies
- Fast enough for real-time use

## Usage

```bash
# Search memories
python3 memsearch.py "what projects is Robb working on"

# Search with more results
python3 memsearch.py "Julie birthday" --limit 10

# JSON output for programmatic use
python3 memsearch.py "calendar events" --json
```

## Installation

```bash
# Basic (TF-IDF only, no dependencies)
python3 memsearch.py "query"

# With local embeddings (better semantic search)
pip install sentence-transformers
python3 memsearch.py "query" --embeddings
```

## Built by Clay 🏺

Created during the nightly autonomy project on 2026-01-28.
