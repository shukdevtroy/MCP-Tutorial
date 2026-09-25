"""
MCP Server: Personal Knowledge Base
------------------------------------
Exposes two tools over the Model Context Protocol (stdio transport):

  - search_notes(query, top_k) -> most relevant note snippets (TF-IDF search)
  - read_note(filename)        -> full content of one note

Run this as a subprocess of an MCP client (see kb_assistant_mcp.py) — it is
not meant to be run standalone in a terminal and left idle; it communicates
over stdin/stdout using the MCP protocol.

Setup
-----
pip install mcp scikit-learn

Notes live in NOTES_DIR (default ./notes, override with env var).
"""

import os
import glob
import json

from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NOTES_DIR = os.environ.get("NOTES_DIR", os.path.join(os.path.dirname(__file__), "notes"))

mcp = FastMCP("kb-notes")

_notes_cache = {}
_vectorizer = None
_doc_matrix = None
_filenames = []


def load_notes():
    global _notes_cache
    _notes_cache = {}
    os.makedirs(NOTES_DIR, exist_ok=True)
    paths = glob.glob(os.path.join(NOTES_DIR, "**", "*.md"), recursive=True)
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _notes_cache[os.path.relpath(path, NOTES_DIR)] = f.read()
        except Exception as e:
            print(f"[warn] could not read {path}: {e}")


def build_index():
    global _vectorizer, _doc_matrix, _filenames
    load_notes()
    _filenames = list(_notes_cache.keys())
    if not _filenames:
        _vectorizer = None
        _doc_matrix = None
        return
    corpus = [_notes_cache[f] for f in _filenames]
    _vectorizer = TfidfVectorizer(stop_words="english")
    _doc_matrix = _vectorizer.fit_transform(corpus)


build_index()


@mcp.tool()
def search_notes(query: str, top_k: int = 3) -> str:
    """Search the user's personal notes and return the most relevant snippets.

    Args:
        query: search text
        top_k: number of results to return (default 3)
    """
    if _vectorizer is None or _doc_matrix is None:
        return json.dumps([{"error": f"No notes indexed. Add .md files to {NOTES_DIR}."}])

    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _doc_matrix).flatten()
    ranked = sorted(zip(_filenames, scores), key=lambda x: x[1], reverse=True)

    results = []
    for filename, score in ranked[:top_k]:
        if score <= 0:
            continue
        content = _notes_cache[filename]
        snippet = content[:500] + ("..." if len(content) > 500 else "")
        results.append({
            "filename": filename,
            "relevance_score": round(float(score), 3),
            "snippet": snippet,
        })

    if not results:
        return json.dumps([{"info": "No relevant notes found for this query."}])
    return json.dumps(results)


@mcp.tool()
def read_note(filename: str) -> str:
    """Read the full content of one specific note by its exact filename.

    Args:
        filename: exact filename as returned by search_notes
    """
    content = _notes_cache.get(filename)
    if content is None:
        return json.dumps({"error": f"Note '{filename}' not found. Check the exact filename from search_notes."})
    return json.dumps({"filename": filename, "content": content})


@mcp.tool()
def reindex_notes() -> str:
    """Re-scan NOTES_DIR and rebuild the search index (call after adding/editing notes)."""
    build_index()
    return json.dumps({"status": "ok", "notes_indexed": len(_filenames)})


if __name__ == "__main__":
    mcp.run()
