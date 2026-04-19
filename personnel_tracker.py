"""
wiki_engine/personnel_tracker.py
=================================
Cross-document personnel dossier builder for the Sentinel-LAC Intel Wiki.

Responsibility: maintain a persistent, growing dossier for every identified
PLA soldier across the full 12-month corpus.

Key operations:
  upsert()              Create or update a dossier when a new report names
                        a soldier.  Append-only — never overwrites existing sightings.

  save_image_asset()    Write raw image bytes to wiki/assets/ and return the
                        wiki-relative path.

  scan_all_mentions()   Walk every wiki .md file to find articles that mention
                        known personnel names.  Used to rebuild cross-references
                        after a batch ingest.

  rebuild_cross_refs()  Update each dossier's "Appears In" section based on
                        scan_all_mentions() results.

  get_dossier()         Return a structured dict of a person's full record for
                        the dashboard personnel panel.

Article format:  wiki/personnel/<slug>.md
Image location:  wiki/assets/<src-stem>-img-<N>.<ext>
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .intel_schema import RANK_PREFIX_RE

log = logging.getLogger(__name__)


class PersonnelTracker:
    """
    Builds and maintains personnel dossiers across the full report corpus.

    Args:
        wiki_root: Path to the wiki/ directory.
    """

    def __init__(self, wiki_root: Path) -> None:
        self.wiki_root      = wiki_root
        self.personnel_dir  = wiki_root / "personnel"
        self.assets_dir     = wiki_root / "assets"
        self.personnel_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def upsert(
        self,
        name:          str,
        rank:          str,
        unit:          str,
        source_doc:    str,
        img_ref:       str | None,
        caption:       str,
        context:       str,
        dtg:           str | None,
        coords:        list[dict[str, str]] | None,
        incident_type: str | None,
        asset_rel_path: str | None = None,
    ) -> Path:
        """
        Create or update a personnel dossier.

        If the dossier already exists (slug match), appends a new sighting
        to the Sighting History.  Never overwrites existing sightings.

        Args:
            name:           Verbatim soldier name as written in source.
            rank:           Verbatim rank.
            unit:           Verbatim unit designation.
            source_doc:     Source filename (e.g. "patrol-report-2024-031.docx").
            img_ref:        Image reference label (e.g. "img-0") or None.
            caption:        Verbatim caption text associated with image.
            context:        Surrounding paragraph text.
            dtg:            Verbatim DTG string or None.
            coords:         List of {system, value} coordinate dicts or None.
            incident_type:  Incident type string or None.
            asset_rel_path: Wiki-relative path to saved image, or None.

        Returns:
            Path to the dossier .md file.
        """
        slug  = self._slugify_name(name)
        path  = self.personnel_dir / f"{slug}.md"
        today = _today()

        # Build sighting entry
        coord_str = ""
        if coords:
            coord_str = ", ".join(f"`{c['value']}`" for c in coords if c.get("value"))

        sighting: dict[str, Any] = {
            "source":     source_doc,
            "dtg":        dtg or "unknown",
            "coords":     coord_str,
            "incident":   incident_type or "unknown",
            "caption":    caption,
            "context":    context[:400],
            "img_ref":    img_ref,
            "asset_path": asset_rel_path,
            "date":       today,
        }

        if path.exists():
            # Merge new sighting into existing dossier
            self._merge_sighting(path, name, rank, unit, source_doc, sighting)
        else:
            # Create new dossier
            self._write_new_dossier(path, name, rank, unit, source_doc, sighting)

        log.info(f"  Personnel dossier upserted: {name} ({slug})")
        return path

    def save_image_asset(
        self,
        img_bytes:   bytes,
        ext:         str,
        source_stem: str,
        img_ref:     str,
    ) -> str:
        """
        Save raw image bytes to wiki/assets/.

        Filename: <source_stem>-<img_ref><ext>
        Returns wiki-relative path string: assets/<filename>

        If an identical image already exists (SHA-256 match), returns
        the existing path without re-writing.
        """
        filename = f"{_sanitise(source_stem)}-{img_ref}{ext}"
        dest     = self.assets_dir / filename

        if not dest.exists():
            dest.write_bytes(img_bytes)
            log.info(f"  Saved image asset: assets/{filename}")
        else:
            # Verify no collision via hash
            existing_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
            new_hash      = hashlib.sha256(img_bytes).hexdigest()
            if existing_hash != new_hash:
                # Rename to avoid overwrite
                filename = f"{_sanitise(source_stem)}-{img_ref}-{new_hash[:6]}{ext}"
                dest     = self.assets_dir / filename
                dest.write_bytes(img_bytes)
                log.info(f"  Saved renamed image asset: assets/{filename}")

        return f"../assets/{filename}"

    def scan_all_mentions(self) -> dict[str, list[str]]:
        """
        Scan every wiki .md file for mentions of known personnel names.

        Returns:
            {person_slug: [list of relative article paths that mention them]}

        Used by rebuild_cross_refs() to update backlinks in dossiers.
        """
        # Build {slug: [known_names]} from existing dossiers
        person_names: dict[str, list[str]] = {}
        for dossier in self.personnel_dir.glob("*.md"):
            fm = _parse_fm(dossier)
            names: list[str] = fm.get("known_names", [])
            if isinstance(names, str):
                names = [names]
            person_names[dossier.stem] = [n.lower() for n in names if n]

        if not person_names:
            return {}

        # Scan all wiki articles (skip personnel/ and .meta/)
        mentions: dict[str, list[str]] = {slug: [] for slug in person_names}

        for md in self.wiki_root.rglob("*.md"):
            rel = str(md.relative_to(self.wiki_root))
            if "personnel" in rel or ".meta" in rel or md.name == "_index.md":
                continue
            try:
                content_low = md.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue

            for slug, names in person_names.items():
                for name in names:
                    if len(name) > 4 and name in content_low:
                        if rel not in mentions[slug]:
                            mentions[slug].append(rel)
                        break

        return {k: v for k, v in mentions.items() if v}

    def rebuild_cross_refs(self) -> None:
        """
        Update each dossier's "Appears In" section based on scan_all_mentions().

        Called once after every ingest batch.  The section is completely
        rebuilt from the current mention scan — previous content is replaced.
        """
        mentions = self.scan_all_mentions()

        for slug, article_paths in mentions.items():
            dossier = self.personnel_dir / f"{slug}.md"
            if not dossier.exists():
                continue

            content  = dossier.read_text(encoding="utf-8", errors="ignore")
            fm, body = _split_fm(content)

            backlinks = "\n".join(f"- [[{p.replace('.md', '')}]]" for p in article_paths)
            new_section = f"\n\n## Appears In\n{backlinks}\n"

            if "## Appears In" in body:
                # Replace existing section
                body = re.sub(
                    r"\n## Appears In\n.*?(?=\n## |\Z)",
                    new_section,
                    body,
                    flags=re.DOTALL,
                )
            else:
                # Append before Sources section, or at end
                if "## Sources" in body:
                    body = body.replace("## Sources", new_section + "\n## Sources", 1)
                else:
                    body += new_section

            fm["total_sightings"] = len(fm.get("sighting_sources", []))
            fm["updated"]         = _today()
            dossier.write_text(
                f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n{body}"
            )
            log.debug(f"  Rebuilt cross-refs for: {slug}")

    def get_dossier(self, name: str) -> dict[str, Any] | None:
        """
        Return full dossier as a structured dict for the dashboard.

        Returns None if no dossier exists for this name.
        """
        slug = self._slugify_name(name)
        path = self.personnel_dir / f"{slug}.md"
        if not path.exists():
            # Try fuzzy slug match
            candidates = list(self.personnel_dir.glob("*.md"))
            for c in candidates:
                fm = _parse_fm(c)
                known = [n.lower() for n in fm.get("known_names", [])]
                if name.lower() in known or slug in c.stem:
                    path = c
                    break
            else:
                return None

        content  = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = _split_fm(content)
        return {
            "path":    str(path.relative_to(self.wiki_root)),
            "fm":      fm,
            "body":    body,
            "content": content,
        }

    def list_all(self) -> list[dict[str, Any]]:
        """Return summary list of all personnel dossiers."""
        results: list[dict[str, Any]] = []
        for d in sorted(self.personnel_dir.glob("*.md")):
            fm = _parse_fm(d)
            results.append({
                "slug":            d.stem,
                "name":            fm.get("title", d.stem),
                "known_ranks":     fm.get("known_ranks", []),
                "known_units":     fm.get("known_units", []),
                "total_sightings": fm.get("total_sightings", 0),
                "images":          fm.get("images", []),
                "first_seen_dtg":  fm.get("first_seen_dtg", ""),
                "updated":         fm.get("updated", ""),
            })
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Private: dossier writing
    # ─────────────────────────────────────────────────────────────────────

    def _write_new_dossier(
        self,
        path:       Path,
        name:       str,
        rank:       str,
        unit:       str,
        source_doc: str,
        sighting:   dict[str, Any],
    ) -> None:
        """Write a brand-new personnel dossier file."""
        today   = _today()
        slug    = path.stem

        fm: dict[str, Any] = {
            "title":             f"Personnel Dossier — {name}",
            "type":              "personnel",
            "classification":    "RESTRICTED",
            "known_names":       [name],
            "known_ranks":       [rank] if rank else [],
            "known_units":       [unit] if unit else [],
            "sighting_sources":  [source_doc],
            "first_seen_dtg":    sighting["dtg"],
            "first_seen_source": source_doc,
            "total_sightings":   1,
            "images":            ([sighting["asset_path"]]
                                   if sighting.get("asset_path") else []),
            "created":           today,
            "updated":           today,
        }

        sighting_block = self._render_sighting(sighting, 1)

        body = f"""
# Personnel Dossier — {name}

> **TLDR:** PLA personnel identified as {rank or 'rank unknown'}, \
{unit or 'unit unknown'}. First seen {sighting['dtg']} in {source_doc}.

## Identity
| Field | Value |
|-------|-------|
| **Name (verbatim)** | `{name}` |
| **Known Ranks** | {rank or '—'} |
| **Known Units** | `{unit or '—'}` |
| **Total Sightings** | 1 |
| **First Seen** | `{sighting['dtg']}` |
| **First Source** | `{source_doc}` |

## Sighting History

{sighting_block}

## Known Associations
<!-- Updated by rebuild_cross_refs() -->

## Sources
- [{source_doc}]
"""
        path.write_text(
            f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n{body}"
        )

    def _merge_sighting(
        self,
        path:       Path,
        name:       str,
        rank:       str,
        unit:       str,
        source_doc: str,
        sighting:   dict[str, Any],
    ) -> None:
        """Append a new sighting to an existing dossier."""
        content  = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = _split_fm(content)
        today    = _today()

        # Update frontmatter
        known_names = fm.get("known_names", [])
        if name not in known_names:
            known_names.append(name)
        fm["known_names"] = known_names

        known_ranks = fm.get("known_ranks", [])
        if rank and rank not in known_ranks:
            known_ranks.append(rank)
        fm["known_ranks"] = known_ranks

        known_units = fm.get("known_units", [])
        if unit and unit not in known_units:
            known_units.append(unit)
        fm["known_units"] = known_units

        sources = fm.get("sighting_sources", [])
        if source_doc not in sources:
            sources.append(source_doc)
        fm["sighting_sources"] = sources

        images = fm.get("images", [])
        if sighting.get("asset_path") and sighting["asset_path"] not in images:
            images.append(sighting["asset_path"])
        fm["images"]          = images
        fm["total_sightings"] = len(sources)
        fm["updated"]         = today

        # Append sighting to history section
        sighting_num   = fm["total_sightings"]
        sighting_block = self._render_sighting(sighting, sighting_num)

        if "## Sighting History" in body:
            insert_marker = "## Sighting History"
            parts         = body.split(insert_marker, 1)
            body          = parts[0] + insert_marker + "\n\n" + sighting_block + parts[1]
        else:
            body += f"\n\n## Sighting History\n\n{sighting_block}"

        # Update identity table total sightings
        body = re.sub(
            r"\| \*\*Total Sightings\*\* \|.*\|",
            f"| **Total Sightings** | {sighting_num} |",
            body,
        )

        # Append to Sources section
        source_line = f"- [{source_doc}]"
        if "## Sources" in body and source_line not in body:
            body = body.replace("## Sources\n", f"## Sources\n{source_line}\n")

        path.write_text(
            f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n{body}"
        )

    @staticmethod
    def _render_sighting(sighting: dict[str, Any], n: int) -> str:
        """Render a single sighting block as Markdown."""
        img_block = ""
        if sighting.get("asset_path"):
            caption   = sighting.get("caption", "")
            img_block = f'\n![{caption}]({sighting["asset_path"]})\n'

        coord_str = sighting.get("coords") or "—"
        return f"""### Sighting {n} — {sighting['dtg']}
**Source:** `{sighting['source']}` \\
**Location:** {coord_str} \\
**Incident:** {sighting.get('incident', '—')} \\
**Caption (verbatim):** "{sighting.get('caption', '—')}"
{img_block}
> Context: *{sighting.get('context', '')[:200]}*

---
"""

    # ─────────────────────────────────────────────────────────────────────
    # Private: utilities
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _slugify_name(name: str) -> str:
        """Convert a person's name to a filesystem-safe slug."""
        s = re.sub(r"[^\w\s-]", "", name.lower())
        s = re.sub(r"[\s_]+", "-", s.strip())
        return f"pla-{s[:60]}"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sanitise(s: str) -> str:
    """Remove characters unsafe for filenames."""
    return re.sub(r"[^\w\-]", "-", s)[:60]


def _parse_fm(path: Path) -> dict[str, Any]:
    """Parse YAML frontmatter from a wiki article. Returns {} on failure."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
    except Exception:
        pass
    return {}


def _split_fm(content: str) -> tuple[dict[str, Any], str]:
    """Split frontmatter and body from article content."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
                return fm, parts[2]
            except yaml.YAMLError:
                pass
    return {}, content
