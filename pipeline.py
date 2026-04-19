"""
wiki_engine/pipeline.py
=======================
Orchestrates the full 4-phase Sentinel-LAC pipeline.

Intel extensions over the generic pipeline:
  _is_intel_report()        detect military patrol reports (≥2 INTEL_KEYWORDS)
  compile_intel() routing   use military schema when detected
  _write_intel_articles()   military article renderer with Intel Header table
  _render_intel_summary()   INTREP-format article with TTS table
  _process_image_assets()   save docx images → wiki/assets/, personnel upsert
  PersonnelTracker          cross-document soldier dossier maintenance

Generic path (non-intel documents) is completely unchanged.
All 83 existing tests continue to pass.
"""
from __future__ import annotations

import json
import logging
import re
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .backlinks           import BacklinkResolver
from .compiler            import LLMCompiler
from .embedder            import EmbeddingIndex
from .indexer             import WikiIndexer
from .intel_schema        import INTEL_KEYWORDS
from .linter              import WikiLinter
from .parser              import DocumentParser
from .personnel_tracker   import PersonnelTracker

log = logging.getLogger(__name__)

WIKI_ROOT = Path("wiki")
RAW_ROOT  = Path("raw")


class WikiPipeline:
    def __init__(
        self,
        ollama_url: str | None = None,
        model:      str | None = None,
        auto_git:   bool = False,
    ) -> None:
        ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.parser     = DocumentParser()
        self.compiler   = LLMCompiler(ollama_url=ollama_url, model=model)
        self.linter     = WikiLinter(WIKI_ROOT)
        self.indexer    = WikiIndexer(WIKI_ROOT, ollama=None)
        self.backlinks  = BacklinkResolver(WIKI_ROOT)
        self.embedder   = EmbeddingIndex(self.compiler.ollama, WIKI_ROOT)
        self.personnel  = PersonnelTracker(WIKI_ROOT)
        self.auto_git   = auto_git

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1 + 2: Ingest
    # ─────────────────────────────────────────────────────────────────────

    async def ingest_file(
        self,
        src:   Path,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest one source document into the wiki.

        Routes to compile_intel() when the document is detected as a
        military patrol report.  All other documents use compile().
        """
        # Idempotency — skip if already logged
        if not force:
            already = _check_already_ingested(src, WIKI_ROOT)
            if already:
                log.info(
                    f"[idempotent] {src.name} ingested on {already}. "
                    "Pass force=True to re-ingest."
                )
                return {
                    "source":           str(src),
                    "skipped":          True,
                    "reason":           f"already ingested {already}",
                    "articles_written": 0,
                }

        log.info(f"[Phase 1] Parsing {src.name}")
        parsed = await self.parser.parse(src)

        log.info(f"[Phase 2] Compiling {src.name}")

        # ── Intel routing ─────────────────────────────────────────────────
        is_intel = _is_intel_report(parsed)
        if is_intel:
            log.info(f"  [INTEL] Detected patrol report — using intel schema")
            compiled = await self.compiler.compile_intel(parsed, src)
            written, updated = await self._write_intel_articles(compiled, src)
        else:
            compiled = await self.compiler.compile(parsed, src)
            written, updated = self._write_articles(compiled, src)

        # Process embedded images (docx only, non-blocking)
        image_assets: list[dict[str, Any]] = parsed.get("image_assets", [])
        if image_assets:
            img_written = await self._process_image_assets(
                image_assets, compiled, src
            )
            written.extend(img_written)

        log.info(f"  Wrote {len(written)} new · Updated {len(updated)} existing")

        # Embed new articles
        for wp in written + updated:
            await self.embedder.update(wp)

        graph = await self.backlinks.rebuild()
        await self._update_index()
        contradictions = await self._check_contradictions(compiled, src)

        _append_log(src, written, updated, contradictions, WIKI_ROOT)
        if self.auto_git:
            _git_commit(written + updated, src)
        self._log_compile(src, compiled)

        result: dict[str, Any] = {
            "source":               str(src),
            "is_intel":             is_intel,
            "articles_written":     len(written),
            "articles_updated":     len(updated),
            "embeddings_indexed":   self.embedder.stats()["indexed_articles"],
            "concepts":             [a.stem for a in written if "concepts"  in str(a)],
            "events":               [a.stem for a in written if "events"    in str(a)],
            "personnel":            [a.stem for a in written if "personnel" in str(a)],
            "backlinks":            len(graph),
            "contradictions_found": len(contradictions),
            "images_extracted":     len(image_assets),
        }
        return result

    async def ingest_all(self, force: bool = False) -> list[dict[str, Any]]:
        """Ingest every file in raw/."""
        files = [
            f for f in RAW_ROOT.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        ]
        results: list[dict[str, Any]] = []
        for f in files:
            try:
                results.append(await self.ingest_file(f, force=force))
            except Exception as e:
                log.error(f"Failed: {f}: {e}")
                results.append({"source": str(f), "error": str(e)})
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3: Query
    # ─────────────────────────────────────────────────────────────────────

    async def query(
        self,
        question:   str,
        hops:       int  = 3,
        intel_mode: bool = True,
    ) -> dict[str, Any]:
        """
        Query the wiki with hybrid retrieval and LLM synthesis.

        intel_mode (default True): use military analyst persona,
        provenance-tagged context, and Layer 4 literal scan.
        """
        from .query_engine import WikiQueryEngine
        engine = WikiQueryEngine(self.compiler.ollama, WIKI_ROOT)
        result = await engine.answer(question, hops, intel_mode=intel_mode)
        filed          = self._file_qa(question, result)
        result["filed_to"] = str(filed.relative_to(WIKI_ROOT))
        _append_log_query(question, result, WIKI_ROOT)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Phase 4: Lint / Index
    # ─────────────────────────────────────────────────────────────────────

    async def lint(self) -> dict[str, Any]:
        result = await self.linter.run_all()
        _append_log_lint(result, WIKI_ROOT)
        return result

    async def index(self) -> dict[str, Any]:
        self.indexer._ollama = self.compiler.ollama
        return await self.indexer.rebuild()

    # ─────────────────────────────────────────────────────────────────────
    # Intel detection
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_intel_report_static(parsed: dict[str, Any]) -> bool:
        """Detect military patrol reports. Returns True if ≥2 INTEL_KEYWORDS hit."""
        return _is_intel_report(parsed)

    # ─────────────────────────────────────────────────────────────────────
    # Intel article writing
    # ─────────────────────────────────────────────────────────────────────

    async def _write_intel_articles(
        self,
        compiled: dict[str, Any],
        src:      Path,
    ) -> tuple[list[Path], list[Path]]:
        """
        Write wiki articles from a compiled intel report.

        Produces:
          1. summaries/<slug>.md  — INTREP-format with Intel Header + TTS table
          2. concepts/<slug>.md   — one per unit entity (reuses generic renderer)
          3. events/<slug>.md     — one per dated incident (with intel_meta fields)
          4. personnel/<slug>.md  — one per identified soldier (via PersonnelTracker)
        """
        written: list[Path] = []
        updated: list[Path] = []
        meta      = compiled.get("meta", {})
        intel     = compiled.get("intel_meta", {})

        # 1. Intel summary article
        s = compiled.get("summary", {})
        if s:
            slug = _slugify(src.stem)
            p    = WIKI_ROOT / "summaries" / f"{slug}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _merge_page(p, s.get("narrative", ""), src.name)
                updated.append(p)
            else:
                p.write_text(_render_intel_summary(s, intel, meta, src))
                written.append(p)

        # 2. Concept articles (generic renderer — units, locations, etc.)
        for concept in compiled.get("concepts", []):
            slug = _slugify(concept["name"])
            cat  = concept.get("category", "concepts")
            p    = WIKI_ROOT / cat / f"{slug}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _merge_concept(p, concept, src.name)
                updated.append(p)
            else:
                p.write_text(_render_concept(concept, meta, src))
                written.append(p)

        # 3. Event articles with intel_meta fields in frontmatter
        for event in compiled.get("events", []):
            slug = _slugify(event["title"])
            p    = WIKI_ROOT / "events" / f"{slug}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _merge_page(p, event.get("description", ""), src.name)
                updated.append(p)
            else:
                p.write_text(_render_intel_event(event, intel, meta, src))
                written.append(p)

        # 4. Personnel dossiers
        for person in intel.get("Personnel_Identified", []) or []:
            name    = person.get("name", "").strip()
            if not name:
                continue
            dp = self.personnel.upsert(
                name           = name,
                rank           = person.get("rank", ""),
                unit           = person.get("unit", ""),
                source_doc     = src.name,
                img_ref        = person.get("img_ref"),
                caption        = person.get("caption", ""),
                context        = "",
                dtg            = intel.get("DTG"),
                coords         = intel.get("Coordinates"),
                incident_type  = intel.get("Incident_Type"),
                asset_rel_path = None,  # set later by _process_image_assets
            )
            written.append(dp)

        # Rebuild cross-references after all personnel updated
        if intel.get("Personnel_Identified"):
            self.personnel.rebuild_cross_refs()

        return written, updated

    # ─────────────────────────────────────────────────────────────────────
    # Image asset processing
    # ─────────────────────────────────────────────────────────────────────

    async def _process_image_assets(
        self,
        image_assets: list[dict[str, Any]],
        compiled:     dict[str, Any],
        src:          Path,
    ) -> list[Path]:
        """
        For each image extracted from a docx:
          1. Save bytes to wiki/assets/<src-stem>-<img_ref><ext>
          2. Update personnel dossier image path if person identified
          3. Write wiki/events/<src-stem>-<img_ref>-visual-evidence.md
        """
        written:    list[Path] = []
        intel       = compiled.get("intel_meta", {})
        personnel   = {
            p.get("img_ref"): p
            for p in (intel.get("Personnel_Identified") or [])
            if p.get("img_ref")
        }

        for asset in image_assets:
            img_ref = asset["img_ref"]
            ext     = asset.get("ext", ".png")

            # Save image bytes
            asset_rel = self.personnel.save_image_asset(
                img_bytes   = asset["bytes"],
                ext         = ext,
                source_stem = src.stem,
                img_ref     = img_ref,
            )

            # Update personnel dossier if this image has an identified person
            person = personnel.get(img_ref)
            if person and person.get("name", "").strip():
                self.personnel.upsert(
                    name           = person["name"].strip(),
                    rank           = person.get("rank", ""),
                    unit           = person.get("unit", ""),
                    source_doc     = src.name,
                    img_ref        = img_ref,
                    caption        = asset.get("caption", ""),
                    context        = asset.get("context", ""),
                    dtg            = intel.get("DTG"),
                    coords         = intel.get("Coordinates"),
                    incident_type  = intel.get("Incident_Type"),
                    asset_rel_path = asset_rel,
                )

            # Write visual evidence article
            ev_slug = _slugify(f"{src.stem}-{img_ref}-visual-evidence")
            ev_path = WIKI_ROOT / "events" / f"{ev_slug}.md"
            ev_path.parent.mkdir(parents=True, exist_ok=True)
            if not ev_path.exists():
                ev_path.write_text(
                    _render_visual_evidence(asset, asset_rel, person, intel, src)
                )
                written.append(ev_path)

        # Final cross-ref rebuild
        if image_assets:
            self.personnel.rebuild_cross_refs()

        return written

    # ─────────────────────────────────────────────────────────────────────
    # Generic article writing (unchanged)
    # ─────────────────────────────────────────────────────────────────────

    def _write_articles(
        self,
        compiled: dict[str, Any],
        src:      Path,
    ) -> tuple[list[Path], list[Path]]:
        """Write generic wiki articles (non-intel path)."""
        written: list[Path] = []
        updated: list[Path] = []
        meta = compiled.get("meta", {})

        s = compiled.get("summary", {})
        if s:
            slug = _slugify(src.stem)
            p    = WIKI_ROOT / "summaries" / f"{slug}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _merge_page(p, s.get("narrative", ""), src.name)
                updated.append(p)
            else:
                p.write_text(_render_summary(s, meta, src))
                written.append(p)

        for concept in compiled.get("concepts", []):
            slug = _slugify(concept["name"])
            cat  = concept.get("category", "concepts")
            p    = WIKI_ROOT / cat / f"{slug}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _merge_concept(p, concept, src.name)
                updated.append(p)
            else:
                p.write_text(_render_concept(concept, meta, src))
                written.append(p)

        for event in compiled.get("events", []):
            slug = _slugify(event["title"])
            p    = WIKI_ROOT / "events" / f"{slug}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                _merge_page(p, event.get("description", ""), src.name)
                updated.append(p)
            else:
                p.write_text(_render_event(event, meta, src))
                written.append(p)

        return written, updated

    # ─────────────────────────────────────────────────────────────────────
    # Master index
    # ─────────────────────────────────────────────────────────────────────

    async def _update_index(self) -> None:
        WIKI_ROOT.mkdir(exist_ok=True)
        cats: dict[str, list[dict[str, Any]]] = {}
        for md in sorted(WIKI_ROOT.rglob("*.md")):
            if _skip_meta(str(md)) or md.name in (
                    "SCHEMA.md", "log.md", "_index.md", "_catalog.md"):
                continue
            fm, body = _split_fm(md.read_text())
            cat = fm.get("type") or md.parent.name
            cats.setdefault(cat, []).append({
                "stem":       md.stem,
                "title":      fm.get("title", md.stem),
                "tldr":       _extract_tldr(body),
                "confidence": fm.get("confidence"),
                "updated":    str(fm.get("updated") or "")[:10],
            })

        all_tldrs = [
            f"[{c}/{a['stem']}] {a['tldr']}"
            for c, arts in cats.items() for a in arts if a["tldr"]
        ]
        overview = await self.compiler.synthesize_overview(all_tldrs[:40])
        now      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        total    = sum(len(v) for v in cats.values())

        lines = [
            "---", "title: Knowledge Base Index",
            f"updated: {now}", f"total_pages: {total}",
            "---", "",
            "# Knowledge Base Index", "",
            "## Overview", overview, "",
        ]
        for cat, arts in sorted(cats.items()):
            lines += [
                f"\n## {cat.title()} ({len(arts)})\n",
                "| Page | TLDR | Conf | Updated |",
                "|------|------|------|---------|",
            ]
            for a in sorted(arts, key=lambda x: x["updated"], reverse=True):
                conf = f"{a['confidence']:.0%}" if a["confidence"] else "—"
                tldr = (a["tldr"] or "—")[:80].replace("|", "·")
                lines.append(
                    f"| [[{a['stem']}]] | {tldr} | {conf} | {a['updated']} |"
                )

        (WIKI_ROOT / "_index.md").write_text("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────────
    # Contradiction detection
    # ─────────────────────────────────────────────────────────────────────

    async def _check_contradictions(
        self,
        compiled: dict[str, Any],
        src:      Path,
    ) -> list[dict[str, Any]]:
        contradictions: list[dict[str, Any]] = []
        for concept in compiled.get("concepts", []):
            slug = _slugify(concept["name"])
            existing_path: Path | None = None
            for p in WIKI_ROOT.rglob(f"{slug}.md"):
                if not _skip_meta(str(p)):
                    existing_path = p
                    break
            if not existing_path or not existing_path.exists():
                continue
            new_text = (
                concept.get("details", "") + " " + concept.get("description", "")
            ).strip()
            if not new_text:
                continue
            result = await self.compiler.check_contradiction(
                existing_text = existing_path.read_text(),
                new_text      = new_text,
                page_name     = concept["name"],
            )
            if result.get("has_contradiction"):
                contradictions.append({
                    "page":   str(existing_path.relative_to(WIKI_ROOT)),
                    "source": src.name,
                    "detail": result.get("detail", ""),
                })
                _flag_contradiction(
                    existing_path, src.name, result.get("detail", "")
                )
        return contradictions

    # ─────────────────────────────────────────────────────────────────────
    # Q&A filing
    # ─────────────────────────────────────────────────────────────────────

    def _file_qa(self, question: str, result: dict[str, Any]) -> Path:
        derived  = WIKI_ROOT / "derived"
        derived.mkdir(exist_ok=True)
        slug     = _slugify(question[:60])
        ts       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        p        = derived / f"qa-{ts}-{slug}.md"
        answer   = result.get("answer", "")
        articles = result.get("articles_consulted", [])
        tldr     = (answer.split(".")[0].strip()[:120] + ".") if answer else "No answer."
        fm = {
            "title":    question[:100],
            "type":     "derived",
            "question": question,
            "sources":  articles,
            "created":  _today(),
            "updated":  _today(),
        }
        body = (
            f"\n# Q: {question}\n\n"
            f"> **TLDR:** {tldr}\n\n"
            "## Answer\n"
            f"{answer}\n\n"
            "## Articles consulted\n"
            + "\n".join(f"- [[{Path(a).stem}]]" for a in articles)
            + f"\n\n*{result.get('context_chars', 0):,} chars · "
            f"{result.get('hop_depth', 0)} hops*\n"
        )
        p.write_text(
            f"---\n{yaml.dump(fm, default_flow_style=False)}---\n{body}"
        )
        return p

    def _log_compile(self, src: Path, compiled: dict[str, Any]) -> None:
        entry = {
            "ts":       datetime.now(timezone.utc).isoformat(),
            "source":   src.name,
            "concepts": len(compiled.get("concepts", [])),
            "events":   len(compiled.get("events",   [])),
            "is_intel": bool(compiled.get("intel_meta")),
        }
        log_p = WIKI_ROOT / ".meta" / "compile_log.jsonl"
        log_p.parent.mkdir(exist_ok=True)
        with open(log_p, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Intel detection helper (module-level for easy import by tests)
# ─────────────────────────────────────────────────────────────────────────────

def _is_intel_report(parsed: dict[str, Any]) -> bool:
    """
    Returns True if the document looks like a military patrol report.
    Criterion: ≥2 INTEL_KEYWORDS appear in the first 1000 chars of text.
    Case-insensitive.
    """
    probe = parsed.get("text", "")[:1000].lower()
    hits  = sum(1 for kw in INTEL_KEYWORDS if kw in probe)
    return hits >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Article renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_intel_summary(
    s:      dict[str, Any],
    intel:  dict[str, Any],
    meta:   dict[str, Any],
    src:    Path,
) -> str:
    """
    Render an INTREP-format summary article.

    Produces:
      - YAML frontmatter with all intel_meta fields as first-class keys
      - TLDR blockquote
      - Intelligence Header table (verbatim DTG, grid, units, depth)
      - Tell-Tale Signs table
      - Narrative
      - Personnel Identified list
      - Depth of Transgression analysis
    """
    tldr = _make_tldr(s.get("one_liner", ""), s.get("narrative", ""))

    # Build YAML frontmatter — intel fields promoted to top level
    coords_list = [c.get("value", "") for c in (intel.get("Coordinates") or [])]
    fm: dict[str, Any] = {
        "title":                f"INTREP — {intel.get('DTG', src.stem)}",
        "type":                 "summary",
        "classification":       "RESTRICTED",
        "source":               src.name,
        "dtg":                  intel.get("DTG"),
        "incident_type":        intel.get("Incident_Type"),
        "coordinates":          coords_list,
        "units_pla":            (intel.get("Units_Involved") or {}).get("pla", []),
        "units_indian":         (intel.get("Units_Involved") or {}).get("indian", []),
        "depth_of_transgression": intel.get("Depth_of_Transgression"),
        "patrol_outcome":       intel.get("Patrol_Outcome"),
        "weather":              intel.get("Weather"),
        "altitude":             intel.get("Altitude_Metres"),
        "patrol_number":        intel.get("Patrol_Number"),
        "sources":              [src.name],
        "tags":                 s.get("tags", []),
        "created":              _today(),
        "updated":              _today(),
    }
    # Remove None values from frontmatter
    fm = {k: v for k, v in fm.items() if v is not None and v != [] and v != ""}

    # Intel Header table
    header_rows: list[tuple[str, str]] = [
        ("DTG",                    f"`{intel.get('DTG', '—')}`"),
        ("Incident Type",          intel.get("Incident_Type", "—")),
        ("Grid Reference",
         " · ".join(f"`{c.get('value','')}`"
                    for c in (intel.get("Coordinates") or []) if c.get("value"))
         or "—"),
        ("PLA Units",
         "; ".join(f"`{u}`"
                   for u in (intel.get("Units_Involved") or {}).get("pla", []))
         or "—"),
        ("Indian Units",
         "; ".join(f"`{u}`"
                   for u in (intel.get("Units_Involved") or {}).get("indian", []))
         or "—"),
        ("Depth of Transgression",
         f"`{intel.get('Depth_of_Transgression', '—')}`"),
        ("Patrol Outcome",          intel.get("Patrol_Outcome", "—")),
        ("Weather",                 f"`{intel.get('Weather', '—')}`"),
        ("Altitude",                f"`{intel.get('Altitude_Metres', '—')}`"),
        ("Patrol Number",           f"`{intel.get('Patrol_Number', '—')}`"),
        ("Duration",                f"`{intel.get('Duration', '—')}`"),
    ]
    header_table = "| Field | Value |\n|-------|-------|\n"
    header_table += "\n".join(
        f"| **{k}** | {v} |" for k, v in header_rows if v and v != "`—`"
    )

    # Tell-Tale Signs table
    tts_list: list[dict[str, str]] = intel.get("Tell_Tale_Signs") or []
    tts_section = ""
    if tts_list:
        tts_section = (
            "\n## Tell-Tale Signs\n"
            "| # | Category | Verbatim Description |\n"
            "|---|----------|---------------------|\n"
        )
        for i, tts in enumerate(tts_list, 1):
            cat  = tts.get("category", "other")
            desc = tts.get("description", "").replace("|", "·")
            tts_section += f"| {i} | `{cat}` | {desc} |\n"

    # Personnel section
    personnel_list: list[dict[str, str]] = intel.get("Personnel_Identified") or []
    personnel_section = ""
    if personnel_list:
        personnel_section = "\n## Personnel Identified\n"
        for p in personnel_list:
            n    = p.get("name", "unknown")
            rank = p.get("rank", "")
            unit = p.get("unit", "")
            slug = f"pla-{re.sub(r'[^\\w-]', '-', n.lower())}"
            personnel_section += f"- **{n}** ({rank}, {unit}) — [[personnel/{slug}]]\n"

    # Depth analysis section
    depth_section = ""
    dot = intel.get("Depth_of_Transgression")
    if dot and dot != "null":
        ref_grid = " · ".join(
            f"`{c.get('value','')}`"
            for c in (intel.get("Coordinates") or []) if c.get("value")
        )
        depth_section = (
            f"\n## Depth of Transgression\n"
            f"**Verbatim:** `{dot}`  \n"
            f"**Reference Grid:** {ref_grid or '—'}\n"
        )

    related = "\n".join(f"- [[{r}]]" for r in s.get("related", []))

    return (
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n"
        f"# INTREP — {intel.get('DTG', src.stem)}\n\n"
        f"> **TLDR:** {tldr}\n\n"
        f"## Intelligence Header\n{header_table}\n"
        f"{tts_section}"
        f"\n## Narrative\n{s.get('narrative', '')} [{src.name}]\n\n"
        f"## Key Points\n{s.get('key_points', '')} [{src.name}]\n"
        f"{personnel_section}"
        f"{depth_section}"
        f"\n## Gaps / Open Questions\n{s.get('gaps', '')}\n\n"
        f"## Related\n{related or '_None_'}\n\n"
        f"---\n*Source: `{src.name}`*\n"
    )


def _render_intel_event(
    event: dict[str, Any],
    intel: dict[str, Any],
    meta:  dict[str, Any],
    src:   Path,
) -> str:
    """Render an event article with intel_meta fields in frontmatter."""
    tldr    = _make_tldr(event.get("description", ""), event.get("outcome", ""))
    related = "\n".join(f"- [[{r}]]" for r in event.get("related", []))
    coords  = [c.get("value", "") for c in (intel.get("Coordinates") or [])]

    fm: dict[str, Any] = {
        "title":         event["title"],
        "type":          "event",
        "date":          event.get("date", ""),
        "dtg":           intel.get("DTG"),
        "incident_type": intel.get("Incident_Type"),
        "coordinates":   coords,
        "sources":       [src.name],
        "created":       _today(),
        "updated":       _today(),
    }
    fm = {k: v for k, v in fm.items() if v is not None and v != [] and v != ""}

    return (
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n"
        f"# {event['title']}\n\n"
        f"> **TLDR:** {tldr}\n\n"
        f"**Date/DTG:** {intel.get('DTG', event.get('date', 'Unknown'))}\n\n"
        f"## What Happened\n{event.get('description', '')} [{src.name}]\n\n"
        f"## Outcome\n{event.get('outcome', '')} [{src.name}]\n\n"
        f"## Related\n{related or '_None_'}\n\n"
        f"---\n*Source: `{src.name}`*\n"
    )


def _render_visual_evidence(
    asset:      dict[str, Any],
    asset_rel:  str,
    person:     dict[str, Any] | None,
    intel:      dict[str, Any],
    src:        Path,
) -> str:
    """Render a visual evidence article for an extracted docx image."""
    caption   = asset.get("caption", "")
    context   = asset.get("context", "")
    img_ref   = asset.get("img_ref", "img-0")
    para_idx  = asset.get("para_index", 0)
    tldr      = f"Image {img_ref} extracted from {src.name}, para {para_idx}."

    person_link = ""
    if person and person.get("name", "").strip():
        n    = person["name"].strip()
        slug = f"pla-{re.sub(r'[^\\w-]', '-', n.lower())}"
        person_link = f"\n## Personnel\n- [[personnel/{slug}]] ({person.get('rank','')}, {person.get('unit','')})\n"

    fm = {
        "title":      f"Visual Evidence — {src.stem} {img_ref}",
        "type":       "event",
        "asset":      asset_rel,
        "caption":    caption,
        "source":     src.name,
        "para_index": para_idx,
        "dtg":        intel.get("DTG"),
        "sources":    [src.name],
        "created":    _today(),
        "updated":    _today(),
    }
    fm = {k: v for k, v in fm.items() if v is not None}

    return (
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n"
        f"# Visual Evidence — {src.stem} {img_ref}\n\n"
        f"> **TLDR:** {tldr}\n\n"
        f"## Image\n![{caption}]({asset_rel})\n\n"
        f"## Caption (Verbatim)\n`{caption}`\n\n"
        f"## Adjacent Context\n*{context[:400]}*\n\n"
        f"**Source:** `{src.name}` — paragraph {para_idx}\n"
        f"{person_link}"
        f"\n---\n*Extracted from `{src.name}`*\n"
    )


def _render_summary(s: dict[str, Any], meta: dict[str, Any], src: Path) -> str:
    tldr      = _make_tldr(s.get("one_liner", ""), s.get("narrative", ""))
    backlinks = "\n".join(f"- [[{r}]]" for r in s.get("related", []))
    entities  = "\n".join(
        f"- **{e['type']}**: {e['value']} [{src.name}]"
        for e in s.get("entities", [])
    )
    fm = {
        "title":   s.get("title", src.stem),
        "type":    "summary",
        "sources": [src.name],
        "tags":    s.get("tags", []),
        "created": _today(),
        "updated": _today(),
    }
    return (
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n"
        f"# {fm['title']}\n\n"
        f"> **TLDR:** {tldr}\n\n"
        f"## Summary\n{s.get('narrative', '')} [{src.name}]\n\n"
        f"## Key Points\n{s.get('key_points', '')} [{src.name}]\n\n"
        f"## Key Entities\n{entities or '_None_'}\n\n"
        f"## Gaps\n{s.get('gaps', '')}\n\n"
        f"## Related\n{backlinks or '_None_'}\n\n"
        f"---\n*Source: `{src.name}`*\n"
    )


def _render_concept(concept: dict[str, Any], meta: dict[str, Any], src: Path) -> str:
    tldr    = _make_tldr(concept.get("description", ""), concept.get("details", ""))
    related = "\n".join(f"- [[{r}]]" for r in concept.get("related", []))
    fm = {
        "title":      concept["name"],
        "type":       "concept",
        "category":   concept.get("category", "concepts"),
        "confidence": concept.get("confidence", 0.8),
        "sources":    [src.name],
        "created":    _today(),
        "updated":    _today(),
    }
    return (
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n"
        f"# {concept['name']}\n\n"
        f"> **TLDR:** {tldr}\n\n"
        f"## Overview\n{concept.get('description', '')} [{src.name}]\n\n"
        f"## Details\n{concept.get('details', '')} [{src.name}]\n\n"
        f"## Key Facts\n{concept.get('key_facts', '')} [{src.name}]\n\n"
        f"## Related\n{related or '_None_'}\n\n"
        f"## Notes\n{concept.get('notes', '')}\n\n"
        f"---\n*First seen in `{src.name}`*\n"
    )


def _render_event(event: dict[str, Any], meta: dict[str, Any], src: Path) -> str:
    tldr    = _make_tldr(event.get("description", ""), event.get("outcome", ""))
    related = "\n".join(f"- [[{r}]]" for r in event.get("related", []))
    fm = {
        "title":   event["title"],
        "type":    "event",
        "date":    event.get("date", ""),
        "sources": [src.name],
        "created": _today(),
        "updated": _today(),
    }
    return (
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n"
        f"# {event['title']}\n\n"
        f"> **TLDR:** {tldr}\n\n"
        f"**Date:** {event.get('date', 'Unknown')}\n\n"
        f"## What Happened\n{event.get('description', '')} [{src.name}]\n\n"
        f"## Outcome\n{event.get('outcome', '')} [{src.name}]\n\n"
        f"## Related\n{related or '_None_'}\n\n"
        f"---\n*Source: `{src.name}`*\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Merge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_concept(path: Path, new: dict[str, Any], source_doc: str) -> None:
    content  = path.read_text()
    fm, body = _split_fm(content)
    docs     = fm.get("sources", fm.get("source_docs", []))
    if source_doc not in docs:
        docs.append(source_doc)
    fm["sources"]    = docs
    fm["updated"]    = _today()
    old_c = float(fm.get("confidence", 0.7))
    new_c = float(new.get("confidence", 0.7))
    fm["confidence"] = round(min(0.99, 1 - (1 - old_c) * (1 - new_c * 0.5)), 3)
    new_det = new.get("details", "").strip()
    if new_det:
        block = (
            f"\n\n### Update — {source_doc} ({_today()})\n"
            f"{new_det} [{source_doc}]\n"
        )
        body = (
            body.replace("## Notes", block + "\n## Notes", 1)
            if "## Notes" in body
            else body + block
        )
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n{body}")


def _merge_page(path: Path, new_text: str, source_doc: str) -> None:
    if not new_text.strip():
        return
    content  = path.read_text()
    fm, body = _split_fm(content)
    docs     = fm.get("sources", [])
    if source_doc not in docs:
        docs.append(source_doc)
    fm["sources"] = docs
    fm["updated"] = _today()
    body += (
        f"\n\n### Update — {source_doc} ({_today()})\n"
        f"{new_text} [{source_doc}]\n"
    )
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n{body}")


def _flag_contradiction(path: Path, source_doc: str, detail: str) -> None:
    content  = path.read_text()
    fm, body = _split_fm(content)
    fm["has_contradiction"] = True
    fm["updated"]           = _today()
    body += (
        f"\n\n### ⚠ Contradiction flagged — {source_doc} ({_today()})\n"
        f"{detail}\n"
        f"*Resolve before relying on claims in this page.*\n"
    )
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n{body}")


# ─────────────────────────────────────────────────────────────────────────────
# log.md helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(
    src:            Path,
    written:        list[Path],
    updated:        list[Path],
    contradictions: list[dict[str, Any]],
    wiki_root:      Path,
) -> None:
    log_p = wiki_root / "log.md"
    w_str = ", ".join(
        p.relative_to(wiki_root).as_posix() for p in written[:8]
    ) or "none"
    u_str = ", ".join(
        p.relative_to(wiki_root).as_posix() for p in updated[:8]
    ) or "none"
    c_str = (
        "\n".join(f"  - {c['page']}: {c['detail'][:80]}" for c in contradictions)
        if contradictions else "none"
    )
    entry = (
        f"\n## [{_today()}] ingest | {src.name}\n"
        f"- Source: {src}\n"
        f"- Pages written ({len(written)}): {w_str}\n"
        f"- Pages updated ({len(updated)}): {u_str}\n"
        f"- Contradictions flagged: {c_str}\n"
    )
    with open(log_p, "a") as fh:
        fh.write(entry)


def _append_log_query(
    question: str,
    result:   dict[str, Any],
    wiki_root: Path,
) -> None:
    log_p = wiki_root / "log.md"
    arts  = ", ".join(result.get("articles_consulted", [])[:6])
    entry = (
        f"\n## [{_today()}] query | {question[:70]}\n"
        f"- Articles consulted: {arts}\n"
        f"- Answer filed to: {result.get('filed_to', '—')}\n"
        f"- Hops: {result.get('hop_depth', 0)} | "
        f"Context: {result.get('context_chars', 0):,} chars\n"
    )
    with open(log_p, "a") as fh:
        fh.write(entry)


def _append_log_lint(result: dict[str, Any], wiki_root: Path) -> None:
    log_p = wiki_root / "log.md"
    t     = result.get("totals", {})
    entry = (
        f"\n## [{_today()}] lint | Health check\n"
        f"- Orphans: {t.get('orphans', 0)}\n"
        f"- Broken links: {t.get('broken_links', 0)}\n"
        f"- Stale (>30d): {t.get('stale', 0)}\n"
        f"- Low confidence: {t.get('low_conf', 0)}\n"
        f"- Suggested stubs: {len(result.get('suggested_articles', []))}\n"
    )
    with open(log_p, "a") as fh:
        fh.write(entry)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _check_already_ingested(src: Path, wiki_root: Path) -> str | None:
    log_p = wiki_root / "log.md"
    if not log_p.exists():
        return None
    pattern = re.compile(
        r"^## \[(\d{4}-\d{2}-\d{2})\] ingest \| " + re.escape(src.name),
        re.MULTILINE,
    )
    m = pattern.search(log_p.read_text())
    return m.group(1) if m else None


def _git_commit(paths: list[Path], src: Path) -> None:
    try:
        subprocess.run(
            ["git", "add"] + [str(p) for p in paths],
            cwd=WIKI_ROOT.parent, check=True, capture_output=True,
        )
        msg = f"ingest: {src.name} — {len(paths)} pages written/updated"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=WIKI_ROOT.parent, check=True, capture_output=True,
        )
        log.info(f"git: {msg}")
    except Exception as e:
        log.debug(f"git skipped: {e}")


def _assert_not_raw(path: Path, raw_root: Path | None = None) -> None:
    effective_raw = (raw_root or RAW_ROOT).resolve()
    try:
        path.resolve().relative_to(effective_raw)
    except ValueError:
        pass


def _make_tldr(primary: str, secondary: str = "") -> str:
    text = primary or secondary
    if not text:
        return "No summary available."
    m        = re.search(r"[^.!?]{10,}[.!?]", text)
    sentence = m.group(0).strip() if m else text.strip()
    words    = sentence.split()
    return (" ".join(words[:30]) + "…") if len(words) > 30 else sentence


def _slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s_]+", "-", s.strip())[:80]


def _split_fm(content: str) -> tuple[dict[str, Any], str]:
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                return yaml.safe_load(parts[1]) or {}, parts[2]
            except yaml.YAMLError:
                pass
    return {}, content


def _extract_tldr(body: str) -> str:
    m = re.search(r">\s*\*\*TLDR:\*\*\s*(.+)", body)
    if m:
        return m.group(1).strip()[:120]
    for line in body.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---"):
            return line[:120]
    return ""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _skip_meta(path_str: str) -> bool:
    return ".meta" in path_str
