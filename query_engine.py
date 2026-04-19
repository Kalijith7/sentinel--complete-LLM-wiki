"""
wiki_engine/query_engine.py
===========================
Wiki Query Engine — Phase 3 of the pipeline.

Four-layer hybrid retrieval, merged without duplicates:

  Layer 0 — Entity resolution
            Normalise query aliases before any search
            ("PLA 76th" → "PLA 76th Group Army" via thefuzz + UNIT_ALIASES)

  Layer 1 — _index.md always consulted (master catalogue)

  Layer 2 — Semantic (embedding cosine similarity)
            Finds related concepts even when query words don't appear in filenames.

  Layer 3 — Full-body keyword (TF count in article text, not just filenames)
            Catches exact technical terms the embedding model may under-weight.

  Layer 4 — Literal regex full-corpus scan  [INTEL EXTENSION]
            Triggered when query contains MGRS/DTG/LatLon patterns,
            or Layers 1-3 returned < 2 entries, or query has "quoted terms".
            Guarantees no grid reference or DTG is ever missed.

Context assembly uses provenance tagging:
  [src:events/fo-2024-031.md, para:3] The patrol encountered...

This allows intel_mode answers to cite paragraph-level sources.
"""
from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Any

from .compiler import OllamaClient, INTEL_ANSWER_SYSTEM
from .embedder import EmbeddingIndex
from .intel_schema import (
    UNIT_ALIASES,
    MGRS_RE,
    DTG_RE,
    LATLON_DMS_RE,
    LATLON_DD_RE,
    INTEL_KEYWORDS,
)

log = logging.getLogger(__name__)

GRID_RE = re.compile(r'[A-Z]{2}\s?\d{3}\s?\d{3}', re.IGNORECASE)

MAX_HOPS          = 3
MAX_ARTICLES      = 14
MAX_CONTEXT_CHARS = 24000
FUZZY_THRESHOLD   = 85   # minimum thefuzz ratio for alias matching


class WikiQueryEngine:
    """
    Multi-layer wiki retrieval with provenance-tagged context assembly.

    Public API:
        answer(question, hops)  — run retrieval + LLM synthesis
    """

    def __init__(self, ollama: OllamaClient, wiki_root: Path) -> None:
        self.ollama    = ollama
        self.wiki_root = wiki_root
        self.embedder  = EmbeddingIndex(ollama, wiki_root)
        self._graph    = self._load_graph()
        self._retrieval_detail: dict[str, list[str]] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    async def answer(
        self,
        question:   str,
        hops:       int  = MAX_HOPS,
        intel_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Run hybrid retrieval and synthesise an answer.

        Args:
            question:   The operator's query.
            hops:       Backlink traversal depth.
            intel_mode: Use military analyst persona + provenance context.

        Returns dict with:
            answer               — LLM synthesised response
            articles_consulted   — list of relative article paths
            hop_depth            — actual hops used
            context_chars        — context window size
            retrieval_detail     — which layer found which article
        """
        entries = await self._find_entries(question)
        log.info(f"Query '{question[:60]}': {len(entries)} entry points")

        all_articles = self._walk_graph(entries, hops)
        log.info(f"  After {hops}-hop walk: {len(all_articles)} articles")

        # Extremely Strict Provenance Prompting
        STRICT_INTEL_SYSTEM = (
            "You are an elite Military Intelligence Analyst. You are provided with a context window "
            "compiled from field patrol reports. Every paragraph is tagged with its provenance in the format "
            "[src:filename, para:N].\n\n"
            "RULES:\n"
            "1. You MUST answer the operator's question using ONLY the provided context.\n"
            "2. You MUST cross-reference evidence between different reports (e.g., matching equipment damage, injuries, or Tell-Tale Signs) to deduce connections and identify units.\n"
            "3. THINK STEP-BY-STEP: First, list relevant findings from individual reports. Second, actively search for matching items across reports. Third, state your logical deduction.\n"
            "4. If the context does not contain enough evidence to make a logical deduction, state 'Insufficient intelligence available.'\n"
            "5. You MUST cite your claims strictly using the provenance tags inline, e.g., 'The patrol spotted drone activity [src:report_5.md, para:2].'\n"
            "6. NEVER invent or hallucinate unit names, sector names, locations, coordinates, or patrol strengths.\n"
            "7. Write a highly readable, flowing, professional report. DO NOT copy-paste raw context blocks or headers. Integrate the facts into your own natural sentences.\n"
            "8. UNIT-ENTITY CORRELATION: When queried about a specific unit, explicitly correlate it with any associated Intercepts, Call Signs, and Radio Chatter found in the context."
        )

        # Use provenance-tagged context for intel queries
        if intel_mode:
            context = self._build_context_with_provenance(all_articles, question)
            system_prompt = STRICT_INTEL_SYSTEM
        else:
            context = self._build_context(all_articles, question)
            system_prompt = "You are a knowledgeable assistant. Cite article titles. Be precise."

        try:
            from .equipment_ref import lookup_pla_equipment
            found_serials = set(re.findall(r'\b(PLA-[A-Z]{2}-\d{3,4})\b', question + "\n" + context, re.I))
            if found_serials:
                eq_context = []
                for code in found_serials:
                    eq_info = lookup_pla_equipment(code.upper())
                    if eq_info:
                        eq_context.append(
                            f"[src:pla_equipment.json, para:0] {code.upper()} is a {eq_info.get('type', '')} ({eq_info.get('classification', '')}). "
                            f"Description: {eq_info.get('description', '')} "
                            f"Common uses: {', '.join(eq_info.get('common_uses', []))}."
                        )
                if eq_context:
                    context += "\n\n### [data/pla_equipment.json]\n" + "\n".join(eq_context)
        except ImportError:
            pass

        try:
            if any(w in question.lower() for w in ["trend", "depth", "spatiotemporal", "progress", "ingression"]):
                depth_data = []
                for p in all_articles:
                    try:
                        text = p.read_text()
                        if text.startswith("---"):
                            import yaml
                            fm = yaml.safe_load(text.split("---", 2)[1])
                            if fm and fm.get("patrol_dates") and fm.get("depth_m") is not None:
                                depth_data.append((fm["patrol_dates"][0], float(fm["depth_m"]), fm.get("sector", "Unknown")))
                    except Exception:
                        continue
                if depth_data:
                    depth_data.sort(key=lambda x: x[0])
                    trend_ctx = "\n\n### [Mathematical Aggregation: Spatiotemporal Depth Trend]\n"
                    for d, m, s in depth_data:
                        trend_ctx += f"- [src:mathematical_aggregation, para:0] {d[:10]} | Sector: {s} | Depth: {m}m\n"
                    if len(depth_data) > 1:
                        mid = len(depth_data) // 2
                        avg_early = sum(x[1] for x in depth_data[:mid]) / max(mid, 1)
                        avg_late = sum(x[1] for x in depth_data[mid:]) / max(len(depth_data) - mid, 1)
                        if avg_late > avg_early * 1.5:
                            trend_ctx += f"- [src:mathematical_aggregation, para:1] Trend Assessment: ESCALATING. Average depth increased significantly from {avg_early:.0f}m to {avg_late:.0f}m, indicating incursions are becoming deeper and more sophisticated over time.\n"
                        elif avg_late < avg_early * 0.5:
                            trend_ctx += f"- [src:mathematical_aggregation, para:1] Trend Assessment: DECREASING. Average depth decreased significantly from {avg_early:.0f}m to {avg_late:.0f}m.\n"
                        else:
                            trend_ctx += f"- [src:mathematical_aggregation, para:1] Trend Assessment: STABLE. Average depth changed from {avg_early:.0f}m to {avg_late:.0f}m.\n"
                    context += trend_ctx
        except Exception:
            pass

        user = (
            f"WIKI CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer based only on the wiki context. "
            "If the information is not present, say so clearly."
        )
        answer = await self.ollama.chat(system_prompt, user)

        return {
            "answer":             answer,
            "articles_consulted": [
                str(p.relative_to(self.wiki_root)) for p in all_articles
            ],
            "hop_depth":      hops,
            "context_chars":  len(context),
            "retrieval_detail": self._retrieval_detail,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Layer 0 — Entity resolution
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_entity_aliases(self, query: str) -> str:
        """
        Normalise known unit aliases in the query string.

        Uses UNIT_ALIASES exact map first, then thefuzz fuzzy matching
        at FUZZY_THRESHOLD% similarity for partial matches.

        Examples:
          "PLA 76th" → "PLA 76th Group Army"
          "wtc forces" → "PLA Western Theatre Command forces"
        """
        q_lower = query.lower()
        result  = query

        # Exact alias map (case-insensitive)
        for variant, canonical in UNIT_ALIASES.items():
            if variant in q_lower:
                # Replace case-insensitively but preserve surrounding text
                pattern = re.compile(re.escape(variant), re.IGNORECASE)
                result  = pattern.sub(canonical, result)
                log.debug(f"[EntityRes] '{variant}' → '{canonical}'")
                q_lower = result.lower()  # update for subsequent checks

        # Fuzzy matching fallback using thefuzz if available
        try:
            from thefuzz import process  # type: ignore
            canonical_names = list(set(UNIT_ALIASES.values()))

            # Extract meaningful tokens (≥5 chars) from query
            tokens = re.findall(r"\b\w{5,}\b", result)
            for token in tokens:
                match, score = process.extractOne(token, canonical_names)
                if score >= FUZZY_THRESHOLD and token.lower() not in match.lower():
                    log.debug(
                        f"[EntityRes] fuzzy '{token}' → '{match}' (score={score})"
                    )
                    result = result.replace(token, match)
        except ImportError:
            log.debug("[EntityRes] thefuzz not installed — skipping fuzzy matching")
        except Exception as e:
            log.debug(f"[EntityRes] fuzzy matching failed: {e}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Entry-point discovery (4 layers)
    # ─────────────────────────────────────────────────────────────────────

    async def _find_entries(self, question: str) -> list[Path]:
        """
        Discover entry-point articles using four layers.

        Returns deduplicated list of up to 12 article paths.
        """
        self._retrieval_detail = {
            "resolution": [],
            "always":     [],
            "semantic":   [],
            "keyword":    [],
            "literal":    [],
        }

        entries: list[Path] = []
        seen:    set[str]   = set()

        def add(p: Path, source: str) -> None:
            sp = str(p)
            # Exclude metadata files from context
            if p.name.lower() in ("schema.md", "log.md", "_catalog.md"):
                return
            if sp not in seen and p.exists():
                seen.add(sp)
                entries.append(p)
                self._retrieval_detail[source].append(p.stem)

        # Layer 0: Entity resolution
        resolved_question = self._resolve_entity_aliases(question)
        if resolved_question != question:
            self._retrieval_detail["resolution"].append(
                f"'{question[:40]}' → '{resolved_question[:40]}'"
            )

        # Layer 1: Master index always first
        idx = self.wiki_root / "_index.md"
        if idx.exists():
            add(idx, "always")

        # Layer 4 (Elevated): Literal regex scan
        # If a grid/MGRS is found, bypass semantic similarity and prioritize literal hits
        has_literal = self._query_has_literal_patterns(resolved_question)
        if has_literal:
            literal_hits = self._literal_scan(resolved_question, top_k=6)
            for p in literal_hits:
                add(p, "literal")
            
            if literal_hits:
                log.info("Grid/literal pattern found, bypassing semantic similarity.")
                return entries[:12]

        # Layer 2: Semantic embedding search
        try:
            semantic_hits = await self.embedder.search(resolved_question, top_k=6)
            for p in semantic_hits:
                add(p, "semantic")
        except Exception as e:
            log.debug(f"Semantic search failed: {e}")

        # Layer 3: Full-body keyword search
        keyword_hits = self._content_keyword_search(resolved_question, top_k=4)
        for p in keyword_hits:
            add(p, "keyword")

        # Literal regex scan fallback for quoted terms or low results
        needs_literal = (
            len(entries) < 2
            or bool(re.search(r'"[^"]+"', resolved_question))  # quoted terms
        )
        if needs_literal and not has_literal:
            literal_hits = self._literal_scan(resolved_question, top_k=6)
            for p in literal_hits:
                add(p, "literal")

        log.debug(f"  Retrieval: {self._retrieval_detail}")
        return entries[:12]

    # ─────────────────────────────────────────────────────────────────────
    # Layer 3 — Full-body keyword search
    # ─────────────────────────────────────────────────────────────────────

    def _content_keyword_search(
        self,
        question: str,
        top_k:    int = 4,
    ) -> list[Path]:
        """
        Count query-token occurrences in full article text (not just filenames).
        Returns top_k articles by term frequency.
        """
        stop = {
            "that", "this", "with", "from", "what", "when", "were",
            "have", "been", "will", "they", "their", "about", "which",
            "would", "could", "there", "into", "than", "then",
        }
        words = [
            w.lower()
            for w in re.findall(r"\b[\w-]{4,}\b", question)
            if w.lower() not in stop
        ]
        if not words:
            return []

        scored: list[tuple[int, Path]] = []
        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md) or "derived" in str(md) or md.name.lower() in ("schema.md", "log.md", "_catalog.md"):
                continue
            try:
                content = md.read_text(encoding="utf-8", errors="ignore").lower()
                score   = sum(content.count(w) for w in words)
                if score > 0:
                    scored.append((score, md))
            except OSError:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        log.debug(f"  keyword top-{top_k}: {[(s, p.stem) for s,p in scored[:top_k]]}")
        return [p for _, p in scored[:top_k]]

    # ─────────────────────────────────────────────────────────────────────
    # Layer 4 — Literal regex full-corpus scan
    # ─────────────────────────────────────────────────────────────────────

    def _query_has_literal_patterns(self, query: str) -> bool:
        """Return True if the query contains MGRS, DTG, or LatLon patterns."""
        return bool(
            MGRS_RE.search(query)
            or DTG_RE.search(query)
            or LATLON_DMS_RE.search(query)
            or LATLON_DD_RE.search(query)
            or GRID_RE.search(query)
        )

    def _literal_scan(
        self,
        query: str,
        top_k: int = 6,
    ) -> list[Path]:
        """
        Raw scan across ALL .md files in wiki_root for exact literal matches.

        Scoring:
          MGRS match in query AND in file  → +20 per match
          DTG  match in query AND in file  → +20 per match
          LatLon match in query AND in file → +15 per match
          "quoted term" in file (case-insensitive) → +10 per term
          Full query string in file        → +5

        Returns top_k by score.  O(n × m) where n=files, m=patterns.
        Acceptable for ≤300 file corpus.
        """
        # Extract literal patterns from query
        mgrs_patterns  = [m.group(0) for m in MGRS_RE.finditer(query)]
        dtg_patterns   = [m.group(0) for m in DTG_RE.finditer(query)]
        latlon_dms     = [m.group(0) for m in LATLON_DMS_RE.finditer(query)]
        latlon_dd      = [m.group(0) for m in LATLON_DD_RE.finditer(query)]
        grid_patterns  = [m.group(0) for m in GRID_RE.finditer(query)]
        quoted_terms   = re.findall(r'"([^"]+)"', query)

        all_literal_patterns = (
            mgrs_patterns + dtg_patterns + latlon_dms + latlon_dd + grid_patterns
        )

        if not all_literal_patterns and not quoted_terms:
            # No specific patterns — use full query as literal string
            all_literal_patterns = [query]

        scored: list[tuple[int, Path]] = []

        for md in self.wiki_root.rglob("*.md"):
            if ".meta" in str(md) or "derived" in str(md) or md.name.lower() in ("schema.md", "log.md", "_catalog.md"):
                continue
            try:
                content      = md.read_text(encoding="utf-8", errors="ignore")
                content_low  = content.lower()
                score        = 0

                for pat in mgrs_patterns:
                    if pat.lower().replace(" ", "") in content_low.replace(" ", ""):
                        score += 20

                for pat in dtg_patterns:
                    if pat.lower() in content_low:
                        score += 20

                for pat in latlon_dms + latlon_dd:
                    if pat.lower() in content_low:
                        score += 15

                for pat in grid_patterns:
                    if pat.lower().replace(" ", "") in content_low.replace(" ", ""):
                        score += 20

                for term in quoted_terms:
                    if term.lower() in content_low:
                        score += 10

                # Full query fallback
                if not all_literal_patterns and query.lower() in content_low:
                    score += 5

                if score > 0:
                    scored.append((score, md))

            except OSError:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in scored[:top_k]]
        if results:
            log.info(f"  Literal scan found: {[p.stem for p in results]}")
        return results

    # ─────────────────────────────────────────────────────────────────────
    # BFS backlink walk
    # ─────────────────────────────────────────────────────────────────────

    def _walk_graph(self, starts: list[Path], hops: int) -> list[Path]:
        """BFS traversal of [[backlinks]] up to `hops` depth."""
        visited  = {str(p) for p in starts}
        result   = list(starts)
        frontier = list(starts)

        for _ in range(hops):
            if len(result) >= MAX_ARTICLES:
                break
            next_frontier: list[Path] = []

            for article in frontier:
                try:
                    text = article.read_text()
                except OSError:
                    continue

                for link_name in re.findall(r"\[\[([^\]]+)\]\]", text):
                    target = self._resolve(link_name)
                    if target and str(target) not in visited:
                        visited.add(str(target))
                        next_frontier.append(target)
                        result.append(target)
                        if len(result) >= MAX_ARTICLES:
                            break
                if len(result) >= MAX_ARTICLES:
                    break

            frontier = next_frontier
            if not frontier:
                break

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Context assembly
    # ─────────────────────────────────────────────────────────────────────

    def _build_context(self, articles: list[Path]) -> str:
        """Assemble context window from article texts."""
        parts: list[str] = []
        total = 0
        for p in articles:
            try:
                text  = p.read_text()
                label = str(p.relative_to(self.wiki_root))
                chunk = f"\n\n### [{label}]\n{text[:3000]}"
                if total + len(chunk) > MAX_CONTEXT_CHARS:
                    break
                parts.append(chunk)
                total += len(chunk)
            except OSError:
                continue
        joined = "\n".join(parts)
        return joined[:MAX_CONTEXT_CHARS]

    def _build_context_with_provenance(self, articles: list[Path], question: str = "") -> str:
        """
        Assemble context with paragraph-level provenance tags.

        Each non-empty paragraph is prefixed with:
            [src:relative/path.md, para:N]

        This allows the LLM answer to cite paragraph-level sources,
        enabling operators to verify every claim in the source document.
        """
        # Tokenize query for semantic filtering
        stop_words = {"what", "is", "the", "a", "an", "where", "how", "who", "did", "are", "do", "does", "for", "with", "from", "that", "this"}
        q_tokens = [w.lower() for w in re.findall(r'\b[\w-]{3,}\b', question) if w.lower() not in stop_words]

        # Step 4: Unit-Entity Correlation
        unit_keywords = {"regiment", "pla", "battalion", "army", "brigade", "defense", "force", "unit"}
        if any(kw in q_tokens for kw in unit_keywords):
            q_tokens.extend(["intercept", "call", "sign", "radio", "chatter"])

        scored_articles = []

        for p in articles:
            # Skip master index files to prevent location/sector hallucinations in tactical alerts
            if p.name.lower() in ("_index.md", "_catalog.md"):
                continue

            try:
                text  = p.read_text()
                label = str(p.relative_to(self.wiki_root))

                # Retain YAML frontmatter so metadata is searchable and passed to the LLM
                body = text

                # Score article based on query tokens
                body_lower = body.lower()
                score = sum(body_lower.count(t) for t in q_tokens)
                
                scored_articles.append((score, label, body))
            except OSError:
                continue

        # Sort articles by score (highest first)
        scored_articles.sort(key=lambda x: x[0], reverse=True)

        parts: list[str] = []
        total = 0
        
        for score, label, body in scored_articles:
            paragraphs = body.split("\n\n")
            tagged_lines: list[str] = [f"\n\n### [{label}]"]

            for para_idx, para in enumerate(paragraphs):
                stripped = para.strip()
                if not stripped:
                    continue
                tagged_lines.append(
                    f"[src:{label}, para:{para_idx}] {stripped}"
                )

            chunk = "\n".join(tagged_lines)
            if total + len(chunk) > MAX_CONTEXT_CHARS:
                break
            parts.append(chunk)
            total += len(chunk)

        joined = "\n".join(parts)
        return joined[:MAX_CONTEXT_CHARS]

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _resolve(self, link: str) -> Path | None:
        """Find the wiki file for a [[backlink]] name."""
        slug = re.sub(r"[^\w\s-]", "", link.lower()).strip()
        slug = re.sub(r"[\s_]+", "-", slug)[:60]
        for md in self.wiki_root.rglob(f"{slug}.md"):
            return md
        for md in self.wiki_root.rglob("*.md"):
            if slug[:20] in md.stem:
                return md
        return None

    def _load_graph(self) -> dict[str, Any]:
        """Load the backlink graph from .meta/graph.json."""
        gp = self.wiki_root / ".meta" / "graph.json"
        if gp.exists():
            try:
                return json.loads(gp.read_text())
            except Exception:
                pass
        return {}
