"""
Backlink Resolver — builds and maintains the [[wikilink]] graph.
Every time articles are written, this rebuilds graph.json,
enabling the multi-hop query walk.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any


class BacklinkResolver:
    def __init__(self, wiki_root: Path):
        self.wiki_root = wiki_root

    async def rebuild(self) -> dict[str, Any]:
        """
        Scan all wiki .md files, extract [[backlinks]], build
        bidirectional graph, write to .meta/graph.json.
        Returns the graph dict.
        """
        graph: dict[str, Any] = {}

        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            rel = str(md.relative_to(self.wiki_root))
            try:
                content = md.read_text()
            except OSError:
                continue

            links = list(set(re.findall(r"\[\[([^\]]+)\]\]", content)))
            graph[rel] = {
                "stem":     md.stem,
                "links_to": links,
                "linked_by": [],
            }

        # Build reverse edges
        for node, data in graph.items():
            for link_name in data["links_to"]:
                slug = self._to_slug(link_name)
                for other_node, other_data in graph.items():
                    if other_data["stem"] == slug or \
                       self._to_slug(other_data["stem"]) == slug:
                        if node not in other_data["linked_by"]:
                            other_data["linked_by"].append(node)

        out = self.wiki_root / ".meta" / "graph.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(graph, indent=2))
        return graph

    def stats(self) -> dict[str, int]:
        gp = self.wiki_root / ".meta" / "graph.json"
        if not gp.exists():
            return {}
        graph = json.loads(gp.read_text())
        total_links = sum(len(v["links_to"]) for v in graph.values())
        return {
            "articles": len(graph),
            "total_links": total_links,
            "orphans": sum(1 for v in graph.values()
                           if not v["links_to"] and not v["linked_by"]),
        }

    @staticmethod
    def _to_slug(s: str) -> str:
        s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
        return re.sub(r"[\s_]+", "-", s)[:80]
