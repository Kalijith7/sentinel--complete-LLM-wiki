"""
wiki_engine/embedder.py
=======================
Semantic embedding index for the wiki.

Problem it solves:
  Old:  _find_entries scores only on filename slug tokens
        → "reasoning breakthrough" never finds chain-of-thought-training.md
        → Answer is in the wiki but the engine never reads that file

  New:  Every article is embedded at ingest time using Ollama's
        nomic-embed-text model. At query time the question is embedded
        and cosine similarity finds the semantically closest articles —
        regardless of whether query words appear in filenames.

Storage:
  wiki/.meta/embeddings.json
  [{path, title, snippet, embedding: [float...]}, ...]
  ~6 KB per article (768-dim). 200 articles ≈ 1.2 MB.

Fallback:
  If Ollama is unreachable or nomic-embed-text isn't pulled, every
  method degrades silently and returns [] so keyword search takes over.
"""
from __future__ import annotations

import json
import logging
import math
import re
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

EMBED_MODEL   = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
STORE_RELPATH = ".meta/embeddings.json"
EMBED_TEXT_CHARS = 600   # title + body snippet fed to the embedding model


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python cosine similarity (no numpy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingIndex
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingIndex:
    """
    Persisted embedding index over wiki articles.

    Lifecycle:
      build()          – embed every article (called by WikiIndexer.rebuild)
      update(path)     – re-embed a single article (called after ingest)
      search(query)    – return top-k paths by semantic similarity
    """

    def __init__(self, ollama, wiki_root: Path):
        self.ollama    = ollama        # OllamaClient instance
        self.wiki_root = wiki_root
        self._store    = wiki_root / STORE_RELPATH
        self._index: list[dict[str, Any]] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    async def build(self) -> int:
        """
        (Re)embed all wiki articles from scratch.
        Returns number of articles embedded.
        """
        self._index = []
        count = 0
        for md in sorted(self.wiki_root.rglob("*.md")):
            if self._skip(md):
                continue
            ok = await self._embed_article(md)
            if ok:
                count += 1
        self._save()
        log.info(f"Embedding index built: {count} articles")
        return count

    async def update(self, path: Path) -> bool:
        """
        Update (or add) the embedding for a single article.
        Called immediately after an article is written during ingest
        so the index stays current without a full rebuild.
        """
        rel = str(path.relative_to(self.wiki_root))
        # Remove stale entry
        self._index = [e for e in self._index if e["path"] != rel]
        ok = await self._embed_article(path)
        if ok:
            self._save()
        return ok

    async def search(self, query: str, top_k: int = 6) -> list[Path]:
        """
        Return the top-k wiki articles most semantically similar to query.
        Falls back to [] if Ollama is unavailable.
        """
        if not self._index:
            return []

        try:
            q_emb = await self.ollama.embed(query)
        except Exception as e:
            log.debug(f"Embed query failed (fallback to keyword): {e}")
            return []

        if not q_emb:
            return []

        scored: list[tuple[float, str]] = []
        for entry in self._index:
            emb = entry.get("embedding")
            if emb:
                sim = cosine(q_emb, emb)
                scored.append((sim, entry["path"]))

        scored.sort(reverse=True)

        results: list[Path] = []
        for sim, rel_path in scored[:top_k]:
            p = self.wiki_root / rel_path
            if p.exists():
                results.append(p)
                log.debug(f"  semantic hit: {rel_path} (sim={sim:.3f})")

        return results

    def stats(self) -> dict[str, int]:
        return {"indexed_articles": len(self._index)}

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _embed_article(self, path: Path) -> bool:
        """Embed one article and add to in-memory index. Returns True on success."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            title, snippet = self._extract_text(content)
            text_to_embed  = f"{title}\n{snippet}"[:EMBED_TEXT_CHARS]

            emb = await self.ollama.embed(text_to_embed)
            if not emb:
                return False

            rel = str(path.relative_to(self.wiki_root))
            self._index.append({
                "path":      rel,
                "title":     title,
                "snippet":   snippet[:200],
                "embedding": emb,
            })
            return True

        except Exception as e:
            log.debug(f"Embedding failed for {path.name}: {e}")
            return False

    @staticmethod
    def _extract_text(content: str) -> tuple[str, str]:
        """
        Pull (title, body_snippet) from a wiki article for embedding.
        Strips YAML frontmatter and wiki syntax so the model sees
        clean natural-language text.
        """
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()

        # Title: first # heading
        m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = m.group(1).strip() if m else body[:60].strip()

        # Body snippet: strip headings, wikilinks, markdown noise
        snippet = re.sub(r"^#{1,6}\s+.+$",   "",  body, flags=re.MULTILINE)
        snippet = re.sub(r"\[\[([^\]]+)\]\]", r"\1", snippet)   # [[link]] → link
        snippet = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", snippet)  # bold/italic
        snippet = re.sub(r"`[^`]+`",          "",  snippet)
        snippet = re.sub(r"^-{3,}$",          "",  snippet, flags=re.MULTILINE)
        snippet = " ".join(snippet.split())    # normalise whitespace
        snippet = snippet[:500]

        return title, snippet

    def _load(self):
        if self._store.exists():
            try:
                self._index = json.loads(self._store.read_text())
                log.debug(f"Loaded {len(self._index)} embeddings from {self._store}")
            except Exception as e:
                log.warning(f"Could not load embedding store: {e}")
                self._index = []

    def _save(self):
        self._store.parent.mkdir(parents=True, exist_ok=True)
        self._store.write_text(json.dumps(self._index))

    @staticmethod
    def _skip(path: Path) -> bool:
        s = str(path)
        return ".meta" in s or "derived" in s
