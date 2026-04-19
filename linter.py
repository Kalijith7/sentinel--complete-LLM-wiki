"""
Wiki Linter — Phase 4: Lint & Maintain.
From the diagram: "Health checks & data integrity"
  - Find inconsistencies
  - Impute missing data
  - Suggest new articles
  - Find connections
"""
from __future__ import annotations
import json
import re
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


class WikiLinter:
    def __init__(self, wiki_root: Path):
        self.wiki_root = wiki_root

    async def run_all(self) -> dict[str, Any]:
        """Run all lint checks. Returns a structured report."""
        orphans       = self._find_orphans()
        broken_links  = self._find_broken_links()
        stale         = self._find_stale_articles(days=30)
        low_conf      = self._find_low_confidence()
        missing_fm    = self._find_missing_frontmatter()
        suggested     = self._suggest_new_articles()

        report = {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "orphaned_articles":  orphans,
            "broken_links":       broken_links,
            "stale_articles":     stale,
            "low_confidence":     low_conf,
            "missing_frontmatter": missing_fm,
            "suggested_articles": suggested,
            "totals": {
                "orphans":      len(orphans),
                "broken_links": len(broken_links),
                "stale":        len(stale),
                "low_conf":     len(low_conf),
            },
        }

        self._write_report(report)
        return report

    # ──────────────────────────────────────────────────────────────────────────
    # Checks
    # ──────────────────────────────────────────────────────────────────────────

    def _find_orphans(self) -> list[str]:
        """Articles with no incoming OR outgoing links"""
        gp = self.wiki_root / ".meta" / "graph.json"
        if not gp.exists():
            return []
        graph = json.loads(gp.read_text())
        return [
            node for node, data in graph.items()
            if not data.get("links_to") and not data.get("linked_by")
            and "_index" not in node
        ]

    def _find_broken_links(self) -> list[dict]:
        """[[links]] that point to non-existent articles"""
        broken = []
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            try:
                content = md.read_text()
            except OSError:
                continue
            for link in re.findall(r"\[\[([^\]]+)\]\]", content):
                resolved = self._resolve(link)
                if not resolved:
                    broken.append({
                        "in_article": str(md.relative_to(self.wiki_root)),
                        "link":       link,
                    })
        return broken

    def _find_stale_articles(self, days: int = 30) -> list[str]:
        """Articles not updated in `days` days"""
        threshold = datetime.now(timezone.utc) - timedelta(days=days)
        stale = []
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            fm = self._parse_fm(md)
            last = fm.get("last_updated") or fm.get("created_at", "")
            if last:
                try:
                    dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
                    if dt < threshold:
                        stale.append(str(md.relative_to(self.wiki_root)))
                except (ValueError, TypeError):
                    pass
        return stale

    def _find_low_confidence(self, threshold: float = 0.5) -> list[dict]:
        """Articles with confidence below threshold"""
        low = []
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            fm = self._parse_fm(md)
            conf = fm.get("confidence")
            if conf is not None and float(conf) < threshold:
                low.append({
                    "article":    str(md.relative_to(self.wiki_root)),
                    "confidence": conf,
                })
        return sorted(low, key=lambda x: x["confidence"])

    def _find_missing_frontmatter(self) -> list[str]:
        """Articles missing YAML frontmatter entirely"""
        missing = []
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            try:
                content = md.read_text()
                if not content.startswith("---"):
                    missing.append(str(md.relative_to(self.wiki_root)))
            except OSError:
                pass
        return missing

    def _suggest_new_articles(self) -> list[str]:
        """
        Find [[links]] that are referenced but don't exist yet.
        These are article stubs that should be created.
        """
        all_stems = {md.stem for md in self.wiki_root.rglob("*.md")}
        mentioned: set[str] = set()
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md):
                continue
            try:
                content = md.read_text()
                for link in re.findall(r"\[\[([^\]]+)\]\]", content):
                    slug = re.sub(r"[\s_]+", "-",
                                  re.sub(r"[^\w\s-]", "", link.lower()).strip())
                    mentioned.add(slug)
            except OSError:
                pass
        return sorted(mentioned - all_stems)

    # ──────────────────────────────────────────────────────────────────────────
    # Report writing
    # ──────────────────────────────────────────────────────────────────────────

    def _write_report(self, report: dict):
        meta_dir = self.wiki_root / ".meta"
        meta_dir.mkdir(exist_ok=True)

        # JSON for machine use
        (meta_dir / "lint_report.json").write_text(json.dumps(report, indent=2))

        # Markdown for human reading
        lines = [
            "# Wiki Lint Report",
            f"\n_Generated: {report['timestamp']}_\n",
            "## Summary",
            f"- Orphaned articles: **{report['totals']['orphans']}**",
            f"- Broken links: **{report['totals']['broken_links']}**",
            f"- Stale articles (>30d): **{report['totals']['stale']}**",
            f"- Low confidence: **{report['totals']['low_conf']}**",
        ]

        if report["broken_links"]:
            lines += ["\n## Broken Links"]
            for b in report["broken_links"][:20]:
                lines.append(f"- `{b['in_article']}` links to **[[{b['link']}]]** (not found)")

        if report["orphaned_articles"]:
            lines += ["\n## Orphaned Articles (no links in or out)"]
            for o in report["orphaned_articles"][:20]:
                lines.append(f"- `{o}`")

        if report["suggested_articles"]:
            lines += ["\n## Suggested New Articles (mentioned but missing)"]
            for s in report["suggested_articles"][:20]:
                lines.append(f"- [[{s}]]")

        (self.wiki_root / ".meta" / "lint_report.md").write_text("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve(self, link: str) -> Path | None:
        slug = re.sub(r"[\s_]+", "-",
                      re.sub(r"[^\w\s-]", "", link.lower()).strip())[:60]
        for md in self.wiki_root.rglob(f"{slug}.md"):
            return md
        for md in self.wiki_root.rglob("*.md"):
            if slug[:20] in md.stem:
                return md
        return None

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
