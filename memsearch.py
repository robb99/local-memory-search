#!/usr/bin/env python3
"""
Local Memory Search for Clawdbot
Semantic search over MEMORY.md and memory/*.md files without external APIs.

Usage:
    python3 memsearch.py "query string"
    python3 memsearch.py "query" --limit 10
    python3 memsearch.py "query" --json
    python3 memsearch.py "query" --embeddings  # if sentence-transformers installed
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from math import log, sqrt
from collections import defaultdict

# Default paths relative to clawd workspace
DEFAULT_WORKSPACE = os.environ.get("CLAWD_WORKSPACE", "/Users/robb/clawd")
MEMORY_FILE = "MEMORY.md"
MEMORY_DIR = "memory"
# Additional context files to search
CONTEXT_FILES = ["USER.md", "IDENTITY.md", "TOOLS.md"]

@dataclass
class SearchResult:
    path: str
    line_start: int
    line_end: int
    snippet: str
    score: float
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "snippet": self.snippet,
            "score": round(self.score, 4)
        }

class TFIDFSearcher:
    """TF-IDF based search with BM25 ranking."""
    
    def __init__(self):
        self.documents: List[Tuple[str, int, int, str]] = []  # (path, line_start, line_end, text)
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_terms: List[Dict[str, int]] = []
        self.avg_doc_len = 0
        self.k1 = 1.5  # BM25 parameter
        self.b = 0.75  # BM25 parameter
        
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        # Remove very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'can',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other',
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                      'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
                      'or', 'because', 'until', 'while', 'it', 'its', 'this',
                      'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
                      'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                      'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                      'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
                      'theirs', 'themselves', 'what', 'which', 'who', 'whom'}
        return [t for t in tokens if len(t) > 1 and t not in stop_words]
    
    def index_document(self, path: str, line_start: int, line_end: int, text: str):
        """Add a document chunk to the index."""
        tokens = self.tokenize(text)
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
            
        # Update document frequencies
        for term in term_freqs:
            self.doc_freqs[term] += 1
            
        self.documents.append((path, line_start, line_end, text))
        self.doc_terms.append(dict(term_freqs))
        
    def finalize_index(self):
        """Calculate average document length after indexing."""
        if self.doc_terms:
            total_len = sum(sum(tf.values()) for tf in self.doc_terms)
            self.avg_doc_len = total_len / len(self.doc_terms)
        
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for documents matching the query using BM25."""
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
            
        n_docs = len(self.documents)
        if n_docs == 0:
            return []
            
        scores = []
        
        for idx, (path, line_start, line_end, text) in enumerate(self.documents):
            doc_terms = self.doc_terms[idx]
            doc_len = sum(doc_terms.values())
            score = 0
            
            for term in query_tokens:
                if term not in doc_terms:
                    continue
                    
                tf = doc_terms[term]
                df = self.doc_freqs[term]
                
                # IDF with smoothing
                idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 term frequency component
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1)))
                
                score += idf * tf_component
                
            if score > 0:
                scores.append(SearchResult(
                    path=path,
                    line_start=line_start,
                    line_end=line_end,
                    snippet=text[:500] + ("..." if len(text) > 500 else ""),
                    score=score
                ))
                
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:limit]


class EmbeddingSearcher:
    """Semantic search using local sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            self.model = None
            self.available = False
            
        self.documents: List[Tuple[str, int, int, str]] = []
        self.embeddings = None
        
    def index_document(self, path: str, line_start: int, line_end: int, text: str):
        """Add a document chunk for embedding."""
        self.documents.append((path, line_start, line_end, text))
        
    def finalize_index(self):
        """Compute embeddings for all documents."""
        if not self.available or not self.documents:
            return
            
        texts = [doc[3] for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search using cosine similarity of embeddings."""
        if not self.available or self.embeddings is None:
            return []
            
        import numpy as np
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                path, line_start, line_end, text = self.documents[idx]
                results.append(SearchResult(
                    path=path,
                    line_start=line_start,
                    line_end=line_end,
                    snippet=text[:500] + ("..." if len(text) > 500 else ""),
                    score=float(similarities[idx])
                ))
                
        return results


class MemorySearcher:
    """Main search interface combining TF-IDF and optional embeddings."""
    
    def __init__(self, workspace: str = DEFAULT_WORKSPACE, use_embeddings: bool = False):
        self.workspace = Path(workspace)
        self.tfidf = TFIDFSearcher()
        self.embedding = EmbeddingSearcher() if use_embeddings else None
        self._index_files()
        
    def _chunk_text(self, text: str, path: str) -> List[Tuple[int, int, str]]:
        """Split text into searchable chunks by paragraph/section."""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        chunk_start = 1
        
        for i, line in enumerate(lines, 1):
            # Start new chunk on headers or after blank lines
            if line.startswith('#') or (not line.strip() and current_chunk and len(current_chunk) > 2):
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append((chunk_start, i - 1, chunk_text))
                current_chunk = [line] if line.strip() else []
                chunk_start = i
            else:
                current_chunk.append(line)
                
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append((chunk_start, len(lines), chunk_text))
                
        return chunks
        
    def _index_files(self):
        """Load and index all memory files."""
        files_to_index = []
        
        # MEMORY.md
        memory_file = self.workspace / MEMORY_FILE
        if memory_file.exists():
            files_to_index.append(memory_file)
            
        # memory/*.md
        memory_dir = self.workspace / MEMORY_DIR
        if memory_dir.exists():
            for f in memory_dir.glob("*.md"):
                files_to_index.append(f)
                
        # Additional context files
        for filename in CONTEXT_FILES:
            ctx_file = self.workspace / filename
            if ctx_file.exists():
                files_to_index.append(ctx_file)
                
        # Index each file
        for filepath in files_to_index:
            try:
                text = filepath.read_text(encoding='utf-8')
                rel_path = str(filepath.relative_to(self.workspace))
                
                for line_start, line_end, chunk in self._chunk_text(text, rel_path):
                    self.tfidf.index_document(rel_path, line_start, line_end, chunk)
                    if self.embedding:
                        self.embedding.index_document(rel_path, line_start, line_end, chunk)
                        
            except Exception as e:
                print(f"Warning: Could not index {filepath}: {e}", file=sys.stderr)
                
        # Finalize indices
        self.tfidf.finalize_index()
        if self.embedding:
            self.embedding.finalize_index()
            
    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search memory files, combining TF-IDF and embeddings if available."""
        tfidf_results = self.tfidf.search(query, limit * 2)
        
        if self.embedding and self.embedding.available:
            embedding_results = self.embedding.search(query, limit * 2)
            
            # Combine and dedupe results
            seen = set()
            combined = []
            
            # Interleave results, normalizing scores
            max_tfidf = max((r.score for r in tfidf_results), default=1)
            max_embed = max((r.score for r in embedding_results), default=1)
            
            for i in range(max(len(tfidf_results), len(embedding_results))):
                if i < len(embedding_results):
                    r = embedding_results[i]
                    key = (r.path, r.line_start)
                    if key not in seen:
                        seen.add(key)
                        # Boost embedding score slightly
                        r.score = (r.score / max_embed) * 1.2
                        combined.append(r)
                        
                if i < len(tfidf_results):
                    r = tfidf_results[i]
                    key = (r.path, r.line_start)
                    if key not in seen:
                        seen.add(key)
                        r.score = r.score / max_tfidf
                        combined.append(r)
                        
            combined.sort(key=lambda x: x.score, reverse=True)
            return combined[:limit]
            
        return tfidf_results[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Local semantic search for Clawdbot memory files"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Max results (default: 5)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--embeddings", "-e", action="store_true", 
                        help="Use local embeddings (requires sentence-transformers)")
    parser.add_argument("--workspace", "-w", default=DEFAULT_WORKSPACE,
                        help=f"Workspace path (default: {DEFAULT_WORKSPACE})")
    
    args = parser.parse_args()
    
    searcher = MemorySearcher(workspace=args.workspace, use_embeddings=args.embeddings)
    results = searcher.search(args.query, limit=args.limit)
    
    if args.json:
        output = {
            "query": args.query,
            "results": [r.to_dict() for r in results]
        }
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print(f"No results found for: {args.query}")
            return
            
        print(f"Found {len(results)} results for: {args.query}\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r.path}:{r.line_start}-{r.line_end} (score: {r.score:.4f})")
            print(f"    {r.snippet[:200]}{'...' if len(r.snippet) > 200 else ''}")
            print()


if __name__ == "__main__":
    main()
