"""
wiki_engine/compiler.py
=======================
LLM Compiler — Phase 2 of the pipeline.

Two paths:
  compile()       — generic knowledge-base documents
  compile_intel() — military patrol reports, enforces VERBATIM EXTRACTION RULE

All LLM calls go to local Ollama — zero external APIs, air-gap safe.
"""
from __future__ import annotations

import json
import re
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .intel_schema import (
    VERBATIM_FIELDS,
    INTEL_JSON_SCHEMA,
    MGRS_RE,
    DTG_RE,
    DEPTH_RE,
    TTS_CATEGORIES,
    INCIDENT_TYPES,
    PATROL_OUTCOMES,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

COMPILE_SYSTEM = """You are an expert knowledge curator. Your job is to read raw documents
and compile them into structured wiki articles for a knowledge base.

Rules:
- Extract ONLY what is actually in the document, never invent
- Use clear, concise language
- Identify meaningful concepts, entities, and events worth linking
- For concept names use canonical forms (e.g. full names, not abbreviations)
- Confidence scores reflect how clearly stated the information is (0.0-1.0)
- related[] arrays use lowercase-hyphenated slugs of article titles
- Keep descriptions factual and complete"""

COMPILE_PROMPT = """\
Read this document and return ONLY valid JSON matching this exact schema.
No preamble, no markdown fences, just the JSON object.

SCHEMA:
{{
  "meta": {{"title": "document title", "doc_type": "article|report|paper|notes|data|other"}},
  "summary": {{
    "title": "human-readable title",
    "one_liner": "one sentence most important idea",
    "narrative": "3-5 paragraph summary",
    "key_points": "- bullet\\n- bullet",
    "gaps": "unanswered questions",
    "tags": ["tag1"],
    "entities": [{{"type": "PERSON|ORG|LOCATION|CONCEPT|TECH|EVENT", "value": "name"}}],
    "related": ["slug-1"]
  }},
  "concepts": [{{
    "name": "Canonical Concept Name",
    "category": "concepts|units|locations|persons|technologies",
    "description": "2-3 sentences",
    "details": "deeper specifics",
    "key_facts": "- fact\\n- fact",
    "notes": "caveats",
    "related": ["slug-1"],
    "confidence": 0.9
  }}],
  "events": [{{
    "title": "Event Title",
    "date": "YYYY-MM-DD or partial",
    "type": "announcement|incident|publication|meeting|other",
    "description": "what happened",
    "outcome": "result",
    "related": ["slug-1"],
    "confidence": 0.85
  }}]
}}

Aim for 2-6 concepts and 0-4 events.

DOCUMENT (filename: {filename}):
{text}

TABLE DATA:
{tables}
"""

INDEX_SYSTEM = """You are a senior analyst writing a master knowledge base index.
Write in clear structured prose. Be comprehensive but concise.
Always cite which summaries or articles support each claim."""

INDEX_PROMPT = """\
Based on these recent summaries and events, write a master index article (_index.md).

Structure your response as clean Markdown with these sections:
## Overview
## Key Themes
## Recent Activity
## Important Concepts (list with [[wikilinks]])
## Open Questions

Recent summaries (most recent first):
{summaries}

Recent events:
{events}
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTEL PROMPTS — used by compile_intel()
# ─────────────────────────────────────────────────────────────────────────────

INTEL_COMPILE_SYSTEM = (
    "You are an Intelligence Staff Officer processing classified patrol reports.\n"
    "Extract structured intelligence and populate the standardised schema.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "VERBATIM EXTRACTION — MANDATORY HARD RULE\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Copy these field types CHARACTER-FOR-CHARACTER from the source.\n"
    "Zero tolerance for rounding, conversion, abbreviation, paraphrasing.\n\n"
    "  DTG         '041530ZAPR24'              NOT 'April 4th'\n"
    "  Coordinates '44RKP88327741'             NOT 'northeast sector'\n"
    "  Units       'PLA 76th Group Army, 3Bn'  NOT 'PLA troops'\n"
    "  Depth       '300 metres past Line X'    NOT 'approx 300m'\n"
    "  Serials     'Vehicle ZK-7734'           NOT 'a vehicle'\n\n"
    "If a field is absent: return null. Do NOT estimate or infer.\n"
    "Intelligence degradation = CRITICAL FAILURE.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "TELL-TALE SIGNS (TTS)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Extract every physical PLA activity indicator.\n"
    "Allowed categories: " + ", ".join(TTS_CATEGORIES) + "\n"
    "Descriptions must be VERBATIM from source.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "PERSONNEL\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "If a soldier is named in image captions or adjacent text, populate\n"
    "Personnel_Identified. Set img_ref to img-0, img-1 etc.\n\n"
    "Output: ONLY valid JSON. No preamble, no markdown fences.\n"
    "Incident_Type ONE OF: " + ", ".join(INCIDENT_TYPES) + "\n"
    "Patrol_Outcome ONE OF: " + ", ".join(PATROL_OUTCOMES)
)

# Template — placeholders replaced via .replace() not f-string
_INTEL_PROMPT_TEMPLATE = """\
Extract structured intelligence from the patrol report below.
Return ONLY valid JSON matching the schema exactly.
No preamble, no markdown fences, just the JSON.

SCHEMA:
__SCHEMA__

IMAGE ASSETS EXTRACTED FROM DOCUMENT:
__IMAGE_ASSETS__

DOCUMENT (filename: __FILENAME__):
__TEXT__

TABLE DATA:
__TABLES__
"""

INTEL_COMPILE_PROMPT: str = _INTEL_PROMPT_TEMPLATE.replace(
    "__SCHEMA__", INTEL_JSON_SCHEMA
)

# Intel answer system prompt
INTEL_ANSWER_SYSTEM = (
    "You are an Intelligence Staff Officer answering operational queries.\n\n"
    "Rules:\n"
    "1. Lead with the most operationally significant finding.\n"
    "2. Quote DTGs and grids VERBATIM: 'at 44RKP88327741 on 041530ZAPR24'.\n"
    "3. Cite provenance for every claim: [source: filename, para: N].\n"
    "4. If two sources conflict, state the conflict — never silently pick one.\n"
    "5. Structure: ASSESSMENT → EVIDENCE → CAVEATS → GAPS.\n"
    "6. THINK STEP-BY-STEP. Cross-reference evidence (e.g., matching equipment damage) to deduce unit attributions where logical. If entirely absent, say so.\n"
    "7. NEVER invent or hallucinate unit names, sector names, locations, coordinates, or patrol strengths.\n"
    "8. UNIT-ENTITY CORRELATION: When queried about a specific unit, explicitly correlate it with any associated Intercepts, Call Signs, and Radio Chatter found in the context.\n"
    "9. PERSONNEL & INTERCEPTS: 'Personnel Identified' and 'Intercepts' in the YAML frontmatter are high-value intelligence. When asked 'Who is...', you MUST scan these specific YAML keys across all context files before checking the body text.\n"
    "10. STRATEGIC SYNTHESIS: Recognize 'Infrastructure' as a keyword for 'Permanence'. If a report contains 'Fiber-optic', 'Permanent line', or 'Z-8 helicopter landing marks', you MUST conclude that the presence is shifting from temporary to permanent."
)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """Async HTTP client for local Ollama server. No external calls."""

    def __init__(
        self,
        url:   str | None = None,
        model: str | None = None,
    ) -> None:
        self.url   = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    async def chat(
        self,
        system:      str,
        user:        str,
        json_mode:   bool  = False,
        temperature: float = 0.05,
    ) -> str:
        """Send a chat completion request to Ollama."""
        payload: dict[str, Any] = {
            "model":    self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(f"{self.url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        """Get embedding vector using nomic-embed-text."""
        embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(
                f"{self.url}/api/embeddings",
                json={"model": embed_model, "prompt": text[:4096]},
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])

    async def models(self) -> list[str]:
        """Return available model names."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{self.url}/api/tags")
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# LLM Compiler
# ─────────────────────────────────────────────────────────────────────────────

class LLMCompiler:
    """
    Compiles parsed documents into structured wiki article data.

    compile()       — generic documents
    compile_intel() — military patrol reports (verbatim extraction enforced)
    """

    def __init__(self, ollama_url: str | None = None, model: str | None = None) -> None:
        self.ollama = OllamaClient(ollama_url, model)

    # ── Generic compile ───────────────────────────────────────────────────

    async def compile(
        self,
        parsed: dict[str, Any],
        src:    Path,
    ) -> dict[str, Any]:
        """Compile a generic document. Returns meta/summary/concepts/events."""
        text   = parsed.get("text", "")[:10000]
        tables = json.dumps(parsed.get("tables", [])[:5], indent=2)[:2000]

        prompt = (
            COMPILE_PROMPT
            .replace("{filename}", src.name)
            .replace("{text}",     text)
            .replace("{tables}",   tables)
        )

        log.info(f"  Compiling (generic): {src.name}")
        raw = await self.ollama.chat(
            system=COMPILE_SYSTEM,
            user=prompt,
            json_mode=True,
        )
        return self._safe_parse(raw, src)

    # ── Intel compile ─────────────────────────────────────────────────────

    async def compile_intel(
        self,
        parsed: dict[str, Any],
        src:    Path,
    ) -> dict[str, Any]:
        """
        Compile a military patrol report using the intel schema.

        Enforces VERBATIM EXTRACTION for DTG, coordinates, unit names,
        depth of transgression, and equipment serials.

        After LLM output, _validate_verbatim_fields() audits that critical
        values from the source appear in the compiled intel_meta.

        Returns dict with keys: meta, intel_meta, summary, concepts, events.
        """
        text   = parsed.get("text", "")[:10000]
        tables = json.dumps(parsed.get("tables", [])[:5], indent=2)[:2000]

        # Summarise image assets — caption + context only, no bytes
        image_assets: list[dict[str, Any]] = parsed.get("image_assets", [])
        assets_summary = "\n".join(
            f"[{a['img_ref']}] caption: {a['caption']!r}"
            f" | context: {a['context'][:200]!r}"
            for a in image_assets
        ) or "None"

        prompt = (
            INTEL_COMPILE_PROMPT
            .replace("__FILENAME__",    src.name)
            .replace("__TEXT__",        text)
            .replace("__TABLES__",      tables)
            .replace("__IMAGE_ASSETS__", assets_summary)
        )

        log.info(f"  Compiling (intel):   {src.name}")
        raw = await self.ollama.chat(
            system=INTEL_COMPILE_SYSTEM,
            user=prompt,
            json_mode=True,
            temperature=0.02,  # tighter for verbatim extraction
        )
        compiled = self._safe_parse(raw, src)

        # Audit verbatim fields — warns only, never modifies
        self._validate_verbatim_fields(
            original_text=parsed.get("text", ""),
            compiled=compiled,
        )
        return compiled

    # ── Index ─────────────────────────────────────────────────────────────

    async def synthesize_index(
        self,
        summaries: list[str],
        events:    list[str],
    ) -> str:
        """Build master _index.md from recent articles."""
        prompt = INDEX_PROMPT.format(
            summaries="\n\n---\n\n".join(summaries[:20]),
            events="\n\n---\n\n".join(events[:10]),
        )
        content = await self.ollama.chat(INDEX_SYSTEM, prompt)
        header = (
            f"---\ntitle: Knowledge Base Index\ntype: index\n"
            f"updated: {datetime.now(timezone.utc).isoformat()}\n---\n\n"
        )
        return header + content

    async def synthesize_overview(self, tldrs: list[str]) -> str:
        """Generate a 2-3 sentence knowledge base overview from TLDRs."""
        if not tldrs:
            return "Knowledge base is empty."
        prompt = (
            "Write a 2-3 sentence overview of what this knowledge base covers. "
            "Be specific, no padding.\n\n" + "\n".join(tldrs[:30])
        )
        return await self.ollama.chat(
            "You write concise, factual knowledge base descriptions.", prompt
        )

    # ── Q&A ───────────────────────────────────────────────────────────────

    async def answer_question(
        self,
        question:   str,
        context:    str,
        intel_mode: bool = False,
    ) -> str:
        """
        Answer a question from wiki context.

        Args:
            intel_mode: Use military analyst persona with provenance rules.
        """
        system = INTEL_ANSWER_SYSTEM if intel_mode else (
            "You are a knowledgeable assistant answering questions from a "
            "structured wiki knowledge base. Cite article titles when using "
            "their information. Be precise and factual."
        )
        user = (
            f"WIKI CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer based only on the wiki context. "
            "If the information is not present, say so clearly."
        )
        return await self.ollama.chat(system, user)

    # ── Contradiction detection ───────────────────────────────────────────

    async def check_contradiction(
        self,
        existing_text: str,
        new_text:      str,
        page_name:     str,
    ) -> dict[str, Any]:
        """
        Compare new evidence against existing wiki page.
        Returns {"has_contradiction": bool, "detail": str}.
        Conservative — only flags direct, clear factual conflicts.
        """
        prompt = (
            f"Compare these two texts about '{page_name}'.\n\n"
            f"EXISTING WIKI PAGE (excerpt):\n{existing_text[:1500]}\n\n"
            f"NEW EVIDENCE:\n{new_text[:800]}\n\n"
            "Do these texts contain a factual contradiction? "
            "(Different dates, quantities, or opposite claims — "
            "NOT just differences in detail level.)\n\n"
            "Return ONLY valid JSON:\n"
            '{"has_contradiction": true|false, "detail": "description or empty string"}'
        )
        raw = await self.ollama.chat(
            system=(
                "You are a fact-checker. Be conservative. "
                "Only flag clear, direct factual conflicts."
            ),
            user=prompt,
            json_mode=True,
        )
        return self._safe_parse(raw, type("_", (), {"name": page_name})())

    # ── Private helpers ───────────────────────────────────────────────────

    def _validate_verbatim_fields(
        self,
        original_text: str,
        compiled:      dict[str, Any],
    ) -> None:
        """
        Audit that MGRS, DTG, and depth values found in source text
        appear in the compiled intel_meta.  Logs warnings for mismatches.
        Never modifies compiled output — audit only.
        """
        if not original_text:
            return

        intel_meta   = compiled.get("intel_meta", {})
        compiled_str = json.dumps(intel_meta, ensure_ascii=False).lower()

        for m in MGRS_RE.finditer(original_text):
            val = m.group(0).replace(" ", "")
            if val.lower() not in compiled_str.replace(" ", ""):
                log.warning(
                    f"[VERBATIM AUDIT] MGRS '{val}' in source "
                    "not found in compiled intel_meta"
                )

        for m in DTG_RE.finditer(original_text):
            val = m.group(0)
            if val.lower() not in compiled_str:
                log.warning(
                    f"[VERBATIM AUDIT] DTG '{val}' in source "
                    "not found in compiled intel_meta"
                )

        if not intel_meta.get("Depth_of_Transgression"):
            if DEPTH_RE.search(original_text):
                log.warning(
                    "[VERBATIM AUDIT] Distance pattern in source "
                    "but Depth_of_Transgression is null"
                )

        log.debug("[VERBATIM AUDIT] Validation complete")

    def _safe_parse(self, raw: str, src: Any) -> dict[str, Any]:
        """Parse LLM JSON output, cleaning common model mistakes."""
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        clean = re.sub(r",(\s*[}\]])", r"\1", clean)

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            src_name = getattr(src, "name", str(src))
            log.warning(f"JSON parse failed for {src_name}: {e}")
            return {
                "meta":       {"title": src_name, "doc_type": "other"},
                "intel_meta": {},
                "summary": {
                    "title":      src_name,
                    "one_liner":  f"Document: {src_name}",
                    "narrative":  raw[:500],
                    "key_points": "",
                    "gaps":       "",
                    "tags":       [],
                    "entities":   [],
                    "related":    [],
                },
                "concepts": [],
                "events":   [],
            }
