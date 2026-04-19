"""
Output Renderer — Derived Outputs.
From diagram: Slides (Marp format), Charts (Matplotlib), Markdown .md articles.
All outputs feed back into the wiki as derived/ articles.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class WikiRenderer:
    def __init__(self, wiki_root: Path, outputs_dir: Path):
        self.wiki_root  = wiki_root
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────
    # Markdown report
    # ─────────────────────────────────────────────────

    def render_markdown_report(self, title: str = "Knowledge Base Report",
                                category_filter: str | None = None) -> Path:
        """
        Compile wiki articles into a single Markdown report.
        Filed back → wiki as a derived output.
        """
        sections: list[str] = []
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

        sections.append(f"# {title}\n\n_Generated: {ts}_\n\n---\n")

        # Master index
        idx = self.wiki_root / "_index.md"
        if idx.exists():
            sections.append(idx.read_text())
            sections.append("\n---\n")

        # Articles
        pattern = f"{category_filter}/*.md" if category_filter else "**/*.md"
        for md in sorted(self.wiki_root.glob(pattern)):
            if ".meta" in str(md) or "derived" in str(md) or "_" == md.name[0]:
                continue
            try:
                sections.append(md.read_text())
                sections.append("\n---\n")
            except OSError:
                pass

        content = "\n\n".join(sections)
        out = self.outputs_dir / f"report-{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
        out.write_text(content)
        return out

    # ─────────────────────────────────────────────────
    # Marp slides
    # ─────────────────────────────────────────────────

    def render_marp_slides(self, title: str = "Knowledge Base Overview",
                            max_slides: int = 20) -> Path:
        """
        Generate a Marp-format Markdown presentation from wiki summaries.
        Marp converts this to HTML/PDF slides.
        """
        slides: list[str] = []

        # Marp frontmatter
        slides.append(
            f"---\nmarp: true\ntheme: default\npaginate: true\n"
            f"title: \"{title}\"\n---\n\n"
            f"# {title}\n\n"
            f"*Auto-generated from knowledge base*  \n"
            f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d')}*\n"
        )

        # Index slide
        idx = self.wiki_root / "_index.md"
        if idx.exists():
            body = self._strip_frontmatter(idx.read_text())
            # Take first 400 chars
            slides.append(f"---\n\n## Index\n\n{body[:400]}\n")

        # One slide per summary (most recent first)
        summaries = sorted(
            (self.wiki_root / "summaries").glob("*.md"),
            key=lambda p: p.stat().st_mtime, reverse=True
        )[:max_slides - 2]

        for md in summaries:
            fm   = self._parse_fm(md)
            body = self._strip_frontmatter(md.read_text())

            # Extract one-liner and first paragraph
            lines   = [l for l in body.split("\n") if l.strip()]
            heading = next((l for l in lines if l.startswith("# ")), f"# {md.stem}")
            content_lines = [l for l in lines if not l.startswith("#")][:8]

            slides.append(
                f"---\n\n{heading}\n\n"
                + "\n".join(content_lines[:6])
                + f"\n\n_Source: `{', '.join(fm.get('source_docs',[md.stem]))}`_\n"
            )

        content = "\n".join(slides)
        out = self.outputs_dir / f"slides-{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
        out.write_text(content)
        return out

    # ─────────────────────────────────────────────────
    # Charts (matplotlib)
    # ─────────────────────────────────────────────────

    def render_charts(self) -> list[Path]:
        """
        Generate wiki analytics charts:
        - Article count by category
        - Confidence distribution
        - Activity timeline (articles added over time)
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from collections import Counter
            from datetime import date
        except ImportError:
            return []

        charts: list[Path] = []
        index_path = self.wiki_root / ".meta" / "index.json"
        if not index_path.exists():
            return []

        data = json.loads(index_path.read_text())
        articles = data.get("articles", [])

        # ── Chart 1: Articles by category ──
        cats = Counter(a.get("type", "other") for a in articles)
        if cats:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(list(cats.keys()), list(cats.values()),
                   color="#4C72B0", alpha=0.85)
            ax.set_title("Wiki Articles by Category", fontsize=13)
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            p = self.outputs_dir / "chart-categories.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            charts.append(p)

        # ── Chart 2: Confidence distribution ──
        confs = [float(a["confidence"]) for a in articles
                 if a.get("confidence") is not None]
        if confs:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(confs, bins=10, range=(0, 1),
                    color="#55A868", alpha=0.85, edgecolor="white")
            ax.set_title("Article Confidence Distribution", fontsize=13)
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Articles")
            ax.axvline(0.5, color="red", linestyle="--", alpha=0.7,
                       label="0.5 threshold")
            ax.legend()
            plt.tight_layout()
            p = self.outputs_dir / "chart-confidence.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            charts.append(p)

        # ── Chart 3: Activity timeline ──
        dates = []
        for a in articles:
            raw = a.get("created_at", "")
            if raw:
                try:
                    dates.append(datetime.fromisoformat(
                        str(raw).replace("Z", "+00:00")).date())
                except (ValueError, TypeError):
                    pass
        if len(dates) >= 3:
            from collections import Counter
            date_counts = Counter(dates)
            sorted_dates = sorted(date_counts)
            counts = [date_counts[d] for d in sorted_dates]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sorted_dates, counts, marker="o",
                    color="#C44E52", linewidth=1.5)
            ax.fill_between(sorted_dates, counts, alpha=0.2, color="#C44E52")
            ax.set_title("Wiki Growth Over Time", fontsize=13)
            ax.set_xlabel("Date")
            ax.set_ylabel("Articles Added")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            plt.xticks(rotation=30)
            plt.tight_layout()
            p = self.outputs_dir / "chart-timeline.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            charts.append(p)

        return charts

    # ─────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────

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

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content
