"""
Wiki Indexer — builds summaries index and cross-link tables.
From diagram: "Indexing — Summaries, links"
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .embedder import EmbeddingIndex


class WikiIndexer:
    def __init__(self, wiki_root: Path, ollama=None):
        self.wiki_root = wiki_root
        # ollama may be None if embeddings are not needed
        self._ollama   = ollama

    async def rebuild(self) -> dict[str, Any]:
        """Rebuild all index structures"""
        summaries = self._build_summaries_index()
        tags      = self._build_tags_index()
        entities  = self._build_entity_index()

        result = {
            "articles":    summaries,
            "tags":        tags,
            "entities":    entities,
            "rebuilt_at":  datetime.now(timezone.utc).isoformat(),
        }

        meta = self.wiki_root / ".meta"
        meta.mkdir(exist_ok=True)
        (meta / "index.json").write_text(json.dumps(result, indent=2, default=str))

        # Write a human-readable catalog
        self._write_catalog(summaries, tags)

        # Rebuild embedding index so semantic search stays in sync
        if self._ollama is not None:
            try:
                embedder = EmbeddingIndex(self._ollama, self.wiki_root)
                n = await embedder.build()
                result["embeddings_rebuilt"] = n
            except Exception as e:
                log.warning(f"Embedding rebuild failed (non-fatal): {e}")
                result["embeddings_rebuilt"] = 0

        return result

    def _build_summaries_index(self) -> list[dict]:
        articles = []
        for md in sorted(self.wiki_root.rglob("*.md"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            if ".meta" in str(md) or "derived" in str(md):
                continue
            fm   = self._parse_fm(md)
            rel  = str(md.relative_to(self.wiki_root))
            articles.append({
                "path":         rel,
                "stem":         md.stem,
                "title":        fm.get("title", md.stem),
                "type":         fm.get("type", md.parent.name),
                "tags":         fm.get("tags", []),
                "confidence":   fm.get("confidence"),
                "created_at":   fm.get("created_at", ""),
                "last_updated": fm.get("last_updated", ""),
                "source_docs":  fm.get("source_docs", []),
            })
        return articles

    def _build_tags_index(self) -> dict[str, list[str]]:
        tag_map: dict[str, list[str]] = defaultdict(list)
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            fm = self._parse_fm(md)
            rel = str(md.relative_to(self.wiki_root))
            for tag in fm.get("tags", []):
                tag_map[str(tag)].append(rel)
        return dict(tag_map)

    def _build_entity_index(self) -> dict[str, list[str]]:
        """Map entity names → articles that mention them"""
        entity_map: dict[str, list[str]] = defaultdict(list)
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            try:
                content = md.read_text()
            except OSError:
                continue
            rel = str(md.relative_to(self.wiki_root))
            for link in re.findall(r"\[\[([^\]]+)\]\]", content):
                entity_map[link].append(rel)
        return dict(entity_map)

    def _write_catalog(self, articles: list[dict], tags: dict):
        lines = [
            "# Wiki Catalog",
            f"\n_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')}_",
            f"\n**{len(articles)} articles total**\n",
        ]

        # Group by category
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for a in articles:
            by_cat[a.get("type", "other")].append(a)

        for cat, items in sorted(by_cat.items()):
            lines.append(f"\n## {cat.title()} ({len(items)})")
            for item in items[:50]:
                title = item.get("title", item["stem"])
                conf  = item.get("confidence")
                conf_str = f" _{conf:.0%} conf_" if conf else ""
                lines.append(f"- [[{item['stem']}]] — {title}{conf_str}")

        if tags:
            lines.append("\n## Tags")
            for tag, paths in sorted(tags.items()):
                lines.append(f"- **#{tag}** ({len(paths)} articles)")

        (self.wiki_root / "_catalog.md").write_text("\n".join(lines))


    @staticmethod
    def _parse_fm(path: Path) -> dict:
        try:
            content = path.read_text()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    return yaml.safe_load(parts[1]) or {}
        except Exception:
            pass
        return {}
