"""
dashboard.py  ·  SENTINEL-LAC Intelligence Wiki  ·  Situation Room
===================================================================
streamlit run dashboard.py
streamlit run dashboard.py -- --wiki /path/to/wiki --model llama3.1:8b

Four operational tabs:
  1. TACTICAL MAP      — lat/lon grid plots, depth circles, sector labels
  2. PATTERN ANALYSIS  — TTS frequency, depth trends, sector heatmap, personnel
  3. LINK ANALYSIS     — network: officers · units · sectors · TTS · feeds pattern
  4. OPERATOR CHAT     — dual-mode: Wiki Lookup (instant) + Ollama Analysis
                         multi-file upload · animated ingest · result approval loop

Critical design principles:
  • run_async()    — thread-based, safe inside Streamlit's running event loop
  • DirectParser   — zero-dependency, works without Ollama
  • To-point query → to-point answer · Analyse query → full report
  • All approved answers committed back to wiki knowledge base
  • Link analysis drives pattern analysis (shared graph)
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import os
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:
    print("pip install streamlit"); sys.exit(1)

try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── Wiki engine ───────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

WIKI_ENGINE_OK = False
_import_error  = ""
try:
    # Stub httpx if not installed so wiki_engine can load for non-LLM paths.
    # Actual HTTP calls will fail at runtime — caught per-call.
    try:
        import httpx  # noqa: F401
    except ImportError:
        import types as _types, sys as _sys
        _httpx = _types.ModuleType("httpx")
        class _AsyncClient:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, *a, **kw): raise RuntimeError("httpx not installed")
            async def get(self, *a, **kw):  raise RuntimeError("httpx not installed")
        _httpx.AsyncClient = _AsyncClient
        _httpx.TimeoutException = RuntimeError
        _sys.modules["httpx"] = _httpx

    from wiki_engine.compiler      import OllamaClient
    from wiki_engine.pipeline      import WikiPipeline, _is_intel_report
    from wiki_engine.intel_schema  import MGRS_RE, DTG_RE, DEPTH_RE
    from wiki_engine.db import (get_stats as _db_stats,
        get_all_reports as _db_reports, get_tts_summary as _db_tts,
        get_all_personnel as _db_pers, get_all_issues as _db_issues,
        get_deepest_reports as _db_deep, get_report as _db_report,
        get_officer_reports as _db_off, get_tts_by_category as _db_tts_cat,
        get_all_tts as _db_all_tts, log_query as _db_log_q,
        commit_analysis as _db_commit)
    from wiki_engine.equipment_ref import lookup_pla_equipment
    WIKI_ENGINE_OK = True
except ImportError as _e:
    _import_error = str(_e)

# ── CLI ───────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--wiki",   default="wiki")
_ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
_ap.add_argument("--model",  default=os.environ.get("OLLAMA_MODEL", ""))
_cli, _ = _ap.parse_known_args()

WIKI_ROOT  = Path(_cli.wiki)
OLLAMA_URL = _cli.ollama
RAW_ROOT   = WIKI_ROOT.parent / "raw"

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# run_async — BUG FIX: safe inside Streamlit's already-running event loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_async(coro: Any, timeout: int = 3600) -> Any:
    def _run():
        lp = asyncio.new_event_loop()
        asyncio.set_event_loop(lp)
        try:
            return lp.run_until_complete(coro)
        finally:
            lp.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run).result(timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_COORDS: dict[str, list[float]] = {
    "kibithoo":        [28.158, 97.017],
    "kibithoo east":   [28.176, 97.031],
    "walong":          [28.167, 97.031],
    "dong":            [28.124, 97.045],
    "chaglagam":       [28.318, 96.604],
    "anjaw":           [28.310, 96.620],
    "tawang":          [27.586, 91.864],
    "bum la":          [27.723, 91.892],
    "nuranang chu":    [27.626, 91.843],
    "yangtse":         [27.765, 92.036],
    "zimithang":       [27.710, 91.730],
    "mechuka":         [28.597, 94.131],
    "tato":            [27.548, 94.222],
    "yargyap chu":     [28.520, 94.185],
    "namka chu":       [27.709, 91.703],
    "lungroka":        [28.200, 97.000],
}

TTS_CATS: dict[str, list[str]] = {
    "Boot Prints":      ["boot","foot","print","impression","step","person"],
    "Vehicle Tracks":   ["tyre","tire","vehicle","track","4x4","wheel","bicycle","tread"],
    "Drone Activity":   ["drone","uav","propeller","buzz","aerial","motor noise"],
    "Fire / Halt":      ["fire","ash","cooking","halt","camp","noodle","ration wrapper","fire pit"],
    "Camouflage":       ["camouflage","netting","bamboo","cut","vegetation","branch","scrap"],
    "Stores / Debris":  ["wrapper","packet","tin","bottle","canteen","cigarette","flask","energy bar"],
    "Markers":          ["marker","cairn","stone","pile","boundary","sign","rock pile","rock marker"],
    "Equipment":        ["flask lid","thermal","military marking","military canteen"],
}
TTS_COL: dict[str, str] = {
    "Boot Prints":     "#f97316",
    "Vehicle Tracks":  "#ef4444",
    "Drone Activity":  "#a855f7",
    "Fire / Halt":     "#f59e0b",
    "Camouflage":      "#22c55e",
    "Stores / Debris": "#3b82f6",
    "Markers":         "#06b6d4",
    "Equipment":       "#ec4899",
    "Other":           "#6b7280",
}
DEPTH_COL = {"Shallow": "#22c55e", "Moderate": "#f59e0b", "Deep": "#ef4444"}
NODE_COL   = {
    "report":  "#1d4ed8",
    "officer": "#d97706",
    "unit":    "#dc2626",
    "sector":  "#065f46",
    "tts":     "#7c3aed",
    "issue":   "#92400e",
}


def classify_tts(text: str) -> str:
    t = text.lower()
    for cat, kws in TTS_CATS.items():
        if any(k in t for k in kws):
            return cat
    return "Other"


def depth_cat(metres: float) -> str:
    if metres < 600:
        return "Shallow"
    if metres < 1500:
        return "Moderate"
    return "Deep"


def resolve_coords(text: str) -> list[float] | None:
    t = text.lower()
    for k, c in sorted(SECTOR_COORDS.items(), key=lambda x: -len(x[0])):
        if k in t:
            return c[:]
    return None


LATLON_PAIR_RE = re.compile(
    r'(?P<lat>[+-]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?))\s*[,/]\s*'
    r'(?P<lon>[+-]?(?:1[0-7]\d(?:\.\d+)?|[1-9]?\d(?:\.\d+)?|180(?:\.0+)?))',
    re.I,
)


def _norm_report_id(a: dict[str, Any]) -> str:
    return str(a.get("report_id") or a.get("_stem") or "UNK")


def _report_primary_date(a: dict[str, Any]) -> str:
    dates = a.get("patrol_dates") or []
    if isinstance(dates, list) and dates:
        return str(dates[0])[:10]
    return str(a.get("patrol_period", "") or "")[:20]


def _extract_latlon_points(text: str) -> list[list[float]]:
    pts: list[list[float]] = []
    for m in LATLON_PAIR_RE.finditer(text or ""):
        try:
            lat = float(m.group("lat"))
            lon = float(m.group("lon"))
        except Exception:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        pts.append([lat, lon])
    return pts


def _route_waypoints(route: str) -> list[list[float]]:
    if not route:
        return []
    segs = re.split(r'->|→|/|,|;|\bto\b| - | – ', route, flags=re.I)
    pts: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for seg in segs:
        c = resolve_coords(seg.strip())
        if c:
            key = (round(c[0], 5), round(c[1], 5))
            if key not in seen:
                pts.append(c)
                seen.add(key)
    return pts


def report_locations(a: dict[str, Any]) -> list[dict[str, Any]]:
    """Return map-plot locations discovered from explicit coords, body and route hints."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    # 1) Exact coordinates in frontmatter
    c = a.get("coords_latlon")
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            c = None
    if isinstance(c, list) and len(c) == 2:
        key = (round(float(c[0]), 5), round(float(c[1]), 5))
        seen.add(key)
        out.append({"lat": float(c[0]), "lon": float(c[1]), "source": "frontmatter"})

    # 2) Any lat/lon mention in report body
    text_blob = f"{a.get('_body', '')}\n{a.get('route', '')}\n{a.get('sector', '')}"
    for lat, lon in _extract_latlon_points(text_blob):
        key = (round(lat, 5), round(lon, 5))
        if key not in seen:
            out.append({"lat": lat, "lon": lon, "source": "latlon_mention"})
            seen.add(key)

    # 3) Route-derived waypoints
    for lat, lon in _route_waypoints(str(a.get("route", "") or "")):
        key = (round(lat, 5), round(lon, 5))
        if key not in seen:
            out.append({"lat": lat, "lon": lon, "source": "route"})
            seen.add(key)

    # 4) Sector fallback
    if not out:
        c2 = resolve_coords(str(a.get("sector", "") or ""))
        if c2:
            out.append({"lat": c2[0], "lon": c2[1], "source": "sector_fallback"})
    return out


def report_threat_score(a: dict[str, Any]) -> float:
    """Simple deterministic threat score for map sensitivity overlays."""
    dm = float(a.get("depth_m", 0) or 0)
    issues = int(a.get("issues_count", 0) or 0)
    tts_count = int(a.get("tts_count", 0) or 0)
    tts_cats = a.get("tts_cats") or []
    if isinstance(tts_cats, str):
        try:
            tts_cats = json.loads(tts_cats)
        except Exception:
            tts_cats = [tts_cats]
    drone = 1 if any("drone" in str(c).lower() for c in tts_cats) else 0
        
    high_threats = a.get("high_threat_indicators", [])
    if isinstance(high_threats, str):
        try:
            high_threats = json.loads(high_threats)
        except Exception:
            high_threats = [high_threats] if high_threats else []
    infra_boost = 3.0 if high_threats else 0.0
    
    return min((dm / 700.0) + (issues * 1.2) + (tts_count * 0.25) + (drone * 2.0) + infra_boost, 10.0)


def build_auto_brief(rpts: list[dict[str, Any]], secs: set[str], tts_map: dict[str, list[str]] | None = None) -> str:
    if not rpts:
        return "<div>No reports available.</div>"
    if tts_map is None:
        tts_map = {}
        for a in rpts:
            cats = a.get("tts_cats") or []
            if isinstance(cats, str):
                try:
                    cats = json.loads(cats)
                except Exception:
                    cats = [cats]
            for c in cats:
                tts_map.setdefault(str(c), []).append(_norm_report_id(a))

    max_d2 = max((float(a.get("depth_m", 0) or 0) for a in rpts), default=0)
    d_cnt = sum(1 for a in rpts if float(a.get("depth_m", 0) or 0) >= 1500)
    dr_cnt = len(tts_map.get("Drone Activity", []))
    infra_cnt = sum(1 for a in rpts if a.get("high_threat_indicators"))

    if max_d2 >= 2500 or d_cnt >= 2 or dr_cnt >= 2 or infra_cnt >= 1:
        tlvl, tclr = "HIGH", "#ef4444"
        tbasis = "deep ingression, permanent infrastructure buildup, and/or repeated drone ISR"
    elif max_d2 >= 1000:
        tlvl, tclr = "MODERATE-HIGH", "#f97316"
        tbasis = "ingression above 1 km in one or more sectors"
    elif max_d2 >= 400:
        tlvl, tclr = "MODERATE", "#f59e0b"
        tbasis = "repeat shallow-to-moderate TTS patterns"
    else:
        tlvl, tclr = "LOW", "#22c55e"
        tbasis = "surface-level TTS only"

    top_tts_b = sorted(tts_map.items(), key=lambda x: -len(x[1]))[:3]
    worst_a = max(rpts, key=lambda x: float(x.get("depth_m", 0) or 0), default={})
    hot_reports = sorted(rpts, key=report_threat_score, reverse=True)[:3]
    hot_line = " · ".join(
        f"{_norm_report_id(a)} ({report_threat_score(a):.1f})" for a in hot_reports if report_threat_score(a) >= 4.8
    ) or "No high-sensitivity clusters detected."

    return f"""
<div style='background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid {tclr};
border-radius:0 6px 6px 0;padding:16px 20px;line-height:2;font-size:13px'>
<div><strong style='color:{tclr}'>THREAT LEVEL: {tlvl}</strong> — {tbasis}</div>
<div style='color:#64748b'><strong style='color:#1e293b'>Corpus:</strong>
  {len(rpts)} reports · {len(secs)} sectors ·
  {sum(int(a.get('tts_count',0) or 0) for a in rpts)} TTS indicators</div>
<div style='color:#64748b'><strong style='color:#1e293b'>Dominant TTS:</strong>
  {' · '.join(f"{c} ({len(rs)})" for c, rs in top_tts_b) if top_tts_b else 'No TTS categories tagged yet'}</div>
<div style='color:#64748b'><strong style='color:#1e293b'>Deepest ingression:</strong>
  {worst_a.get('sector','—')} — Report
  <span style='color:#ef4444'>{worst_a.get('report_id','')}</span> —
  <code>{worst_a.get('depth_raw','—')}</code></div>
<div style='color:#64748b'><strong style='color:#1e293b'>High sensitivity reports:</strong> {hot_line}</div>
</div>"""


def pattern_search(rpts: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    terms = [t for t in re.findall(r"\b\w{2,}\b", q) if t not in {"the", "and", "for", "with", "from", "what", "which"}]
    if not terms:
        return []
    hits: list[dict[str, Any]] = []
    for a in rpts:
        rid = _norm_report_id(a)
        blob = " ".join([
            str(a.get("report_id", "")),
            str(a.get("sector", "")),
            str(a.get("unit", "")),
            str(a.get("route", "")),
            str(a.get("signed_by", "")),
            str(a.get("depth_raw", "")),
            str(a.get("tts_cats", "")),
            str(a.get("_body", "")),
        ]).lower()
        score = sum(blob.count(t) for t in terms)
        if score <= 0:
            continue
        snippet = ""
        for sent in re.split(r"[\n.!?]", str(a.get("_body", ""))):
            s = sent.strip()
            if len(s) < 18:
                continue
            if any(t in s.lower() for t in terms):
                snippet = s[:180]
                break
        hits.append({
            "Report": rid,
            "Sector": str(a.get("sector", ""))[:36],
            "Date": _report_primary_date(a),
            "Pattern Match": score,
            "Snippet": snippet or str(a.get("route", ""))[:180],
        })
    hits.sort(key=lambda x: -x["Pattern Match"])
    return hits[:25]


def verify_llm_output(raw_summary: str, fm_data: dict[str, Any]) -> tuple[bool, str]:
    """Validate LLM-generated executive summaries don't hallucinate serial numbers or coordinates."""
    # Check for hallucinated serial numbers
    found_serials = set(re.findall(r'\b(PLA-[A-Z]{2}-\d{3,4}|[A-Z0-9]{2}-\w+-\d+)\b', raw_summary, re.I))
    valid_serials = {str(s).upper() for s in fm_data.get("serial_numbers", [])}
    for serial in found_serials:
        if serial.upper() not in valid_serials:
            return False, f"Hallucinated serial number: {serial}"
            
    # Check for hallucinated grids
    found_grids = set(re.findall(r'\b([A-Z]{2}\s*\d{3}\s*\d{3})\b', raw_summary, re.I))
    valid_grids = {re.sub(r'\s+', '', str(g)).upper() for g in fm_data.get("grids", [])}
    for grid in found_grids:
        if re.sub(r'\s+', '', grid).upper() not in valid_grids:
            return False, f"Hallucinated grid coordinate: {grid}"
            
    return True, raw_summary.strip()


def auto_link_entities(text: str, fm_data: dict[str, Any]) -> str:
    """Format patrol narrative text with markdown emphasis on key entities (officer names, sectors, equipment codes)."""
    if not text:
        return ""
    res = text
    if fm_data.get("officer_name") and len(fm_data["officer_name"]) > 2:
        res = re.sub(rf'\b({re.escape(fm_data["officer_name"])})\b', r'**\1**', res, flags=re.I)
    if fm_data.get("sector") and len(fm_data["sector"]) > 2:
        res = re.sub(rf'\b({re.escape(fm_data["sector"])})\b', r'**\1**', res, flags=re.I)
    for serial in fm_data.get("serial_numbers", []):
        if serial:
            res = re.sub(rf'\b({re.escape(serial)})\b', r'`\1`', res, flags=re.I)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECT PARSER  —  zero-dep, regex-based patrol report parser
# ═══════════════════════════════════════════════════════════════════════════════

class DirectParser:
    _HDR      = re.compile(r'^[ \t]*(?:\*+\s*)?(?:###\s*|PATROLLING\s+)?Report\s*(?:[-–—:]\s*)?([A-Z0-9]{1,8})\b', re.M | re.I)
    _UNIT     = re.compile(r'\*?\*?Unit\*?\*?\s*(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _SECTOR   = re.compile(r'\*?\*?Sector\*?\*?\s*(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _REPNO    = re.compile(r'\*?\*?Report\s+No\.?(?:[^\n:\-;]{0,25})?(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _PERIOD   = re.compile(r'\*?\*?(?:Patrol\s+)?(?:Period|Dates?|Time)(?:[^\n:\-;]{0,25})?(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _PTYPE    = re.compile(r'\*?\*?(?:Patrol\s+)?(?:Type|Category|Nature)(?:[^\n:\-;]{0,25})?(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _ROUTE    = re.compile(r'\*?\*?Route(?:[^\n:\-;]{0,25})?(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _STRENGTH = re.compile(r'\*?\*?(?:Patrol\s+|Team\s+)?(?:Strength|Personnel|Composition|Members)(?:[^\n:\-;]{0,25})?(?:[:\-;]+)?\s*\n?\s*\*?\*?(?!(?:Weather|Depth|Route|Signed|Sector|Unit|Report|Issues|Wins|Tell)\b)([^\s][^\n]*)', re.I)
    _WEATHER  = re.compile(r'\*?\*?(?:Weather|Climate|Conditions)(?:[^\n:\-;]{0,25})?(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _SIGNED   = re.compile(r'\*?\*?Signed\*?\*?\s*(?:[:\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _DEPTH_H  = re.compile(r'\*?\*?(?:Assumed\s+)?(?:Depth|Ingression)(?:[^\n:\-;\*]{0,25})?(?:[:\*\-;]+)?\s*\n?\s*\*?\*?([^\s][^\n]*)', re.I)
    _GRID     = re.compile(r'Grid\s+(?:approx\s+)?([A-Z]{2}\s*\d{3}\s*\d{3})', re.I)
    _WINS     = re.compile(r'Wins identified.*?\n((?:\s*[-•].+\n?)+)', re.I | re.S)

    def read(self, path: Path) -> str:
        if path.suffix.lower() in (".docx", ".doc"):
            try:
                from docx import Document
                return "\n\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
            except ImportError:
                return "DOCX_ERROR: python-docx library is missing."
            except Exception as e:
                log.warning(f"docx failed {path.name}: {e}"); return f"DOCX_ERROR: {e}"
        return path.read_text(encoding="utf-8-sig", errors="ignore")

    def _strip(self, s: str) -> str:
        return re.sub(r'\*+', '', s).strip() if s else ""

    def _first(self, pat: re.Pattern, text: str) -> str:
        m = pat.search(text)
        return self._strip(m.group(1)) if m else ""

    def _extract_bullets(self, text: str, start_pat: str, stop_pat: str) -> list[str]:
        items, active = [], False
        for line in text.split("\n"):
            ll = re.sub(r'\*+', '', line).strip()
            if re.search(start_pat, ll, re.I):
                active = True; continue
            if active:
                item = re.sub(r'^[-•–\u2022\s]+', '', ll).strip()
                if item and len(item) > 6:
                    items.append(item)
                elif item and re.search(stop_pat, item, re.I):
                    break
        return items

    def _extract_keywords(self, text: str) -> dict[str, list[str]]:
        patterns = {
            "equipment": r'\b(PLA-[A-Z]{2}-\d{3,4}|thermal|imager|camera|sensor|equipment.*?(?:serial|model|code|id))\b',
            "weapons": r'\b(rifle|ak47|ak56|assault|carbine|sniper|rpg|launcher|ammunition|rounds?)\b',
            "comms": r'\b(radio|frequency|khz|mhz|transmission|signal|antenna|headset)\b',
            "vehicles": r'\b(truck|jeep|vehicle|suv|motorcycle|bicycle|transport)\b',
            "locations": r'\b(bunker|firebase|base|camp|fortification|outpost|position|post)\b',
            "activity_patterns": r'\b(night.*patrol|day.*patrol|sustained.*presence|repeated|recurring|daily|nightly)\b',
            "logistics": r'\b(ammunition|ration|supply|food|water|fuel|cache|store|depot)\b',
            "technology": r'\b(thermal|night.*vision|infrared|imager|sensor|optical|binocular|scope)\b',
            "behavior": r'\b(caching|fortif|entrench|hide|conceal|observe|survey|reconn)\b',
            "specific_identifiers": r'\b(call.*sign|unit.*code|team.*code|designation|alpha|bravo|charlie)\b',
            "infrastructure_permanent": r'\b(fiber-optic|fiber optic|permanent|infrastructure reinforcement|cable spool|line laying|concrete|bunker|road construction)\b',
            "surveillance_temporary": r'\b(temporary|tripod|relay|portable|removable|temporary signal|temporary op)\b'
        }
        results = {}
        for cat, pat in patterns.items():
            matches = [m.group(0).strip().lower() for m in re.finditer(pat, text, re.I | re.S)]
            results[cat] = sorted(list(set(matches)))
        return results

    def parse(self, text: str, source: str) -> list[dict[str, Any]]:
        """Parse the full text of a report into structured dictionary objects."""
        hdrs = list(self._HDR.finditer(text))
        seen: dict[str, str] = {}
        
        if not hdrs and len(text.strip()) > 20:
            # Fallback: treat the entire document as one report if no headers found
            rid = re.sub(r'[^A-Z0-9]', '', Path(source).stem.upper())[:8] or "REP1"
            seen[rid] = text.strip()
        else:
            for i, m in enumerate(hdrs):
                rid   = m.group(1).upper()
                start = m.start()
                end   = hdrs[i+1].start() if i+1 < len(hdrs) else len(text)
                sec   = text[start:end].strip()
                if rid not in seen or len(sec) > len(seen[rid]):
                    seen[rid] = sec

        reports: list[dict[str, Any]] = []
        for rid, sec in seen.items():
            if len(sec) < 20:
                continue

            strength_raw = self._first(self._STRENGTH, sec)
            depth_raw    = self._first(self._DEPTH_H,  sec)
            sector       = self._first(self._SECTOR,   sec)
            signed       = self._first(self._SIGNED,   sec)
            period       = self._first(self._PERIOD,   sec)

            # TTS — stop before Assumed depth / Wins / Location / Action
            tts = self._extract_bullets(
                sec,
                r'tell tale|tts.*identified',
                r'^(assumed|wins|location|action|signed|weather|issues)',
            )

            # Wins (PLA capabilities)
            wins = self._extract_bullets(
                sec,
                r'wins identified|pla capabilities|weapons?|communications?|equipment',
                r'^(locations|action|signed|issues|route)',
            )
            if not wins:
                cap_kws = ["weapon", "radio", "camera", "equipment", "rifle", "drone", "uav", "imager", "thermal", "sensor", "optics", "ammunition", "gear", "antenna"]
                for sent in re.split(r'[\n.!?]', sec):
                    s_clean = sent.strip()
                    if len(s_clean) > 10 and any(kw in s_clean.lower() for kw in cap_kws):
                        wins.append(s_clean)

            # Enhance wins with PLA Equipment Database lookups
            enhanced_wins = []
            for w in wins:
                serials = set(re.findall(r'\b(PLA-[A-Z]{2}-\d{3,4})\b', w, re.I))
                for s in serials:
                    eq_info = lookup_pla_equipment(s) if WIKI_ENGINE_OK else None
                    if eq_info and eq_info.get("type") not in w:
                        w += f" (Identified from DB as: {eq_info['type']})"
                enhanced_wins.append(w)
            wins = enhanced_wins

            # Issues
            issues = self._extract_bullets(
                sec,
                r'issues observed',
                r'^(action taken|signed|wins)',
            )

            # Actions taken
            actions = self._extract_bullets(
                sec,
                r'action taken',
                r'^(signed)',
            )

            # Keyword Extraction Integration
            keywords = self._extract_keywords(sec)

            grids   = self._GRID.findall(sec)
            coords  = resolve_coords(sector)
            s_nums  = [int(n) for n in re.findall(r'\b(\d+)\b', strength_raw)]
            s_total = sum(s_nums)

            # Depth in metres
            depth_m = _depth_metres(depth_raw)

            # Officer rank and name
            rank_m  = re.match(r'(Capt\.|Capt|Lt Col|Lt\.|Lt|Maj\.|Maj|Sub Maj\.|'
                               r'Sub Maj|Subedar|Nb Sub|Col|Brig)',
                               signed, re.I)
            rank    = rank_m.group(1).rstrip('.') if rank_m else ""
            officer = signed.replace(rank_m.group(0), "").strip() if rank_m else signed

            reports.append({
                "report_id":      rid,
                "source_doc":     source,
                "unit":           self._first(self._UNIT,    sec),
                "sector":         sector,
                "report_no":      self._first(self._REPNO,   sec),
                "patrol_period":  period,
                "patrol_dates":   _parse_dates(period),
                "patrol_type":    self._first(self._PTYPE,   sec),
                "route":          self._first(self._ROUTE,   sec),
                "strength_raw":   strength_raw,
                "strength_total": s_total,
                "weather":        self._first(self._WEATHER, sec),
                "depth_raw":      depth_raw,
                "depth_m":        depth_m,
                "depth_cat":      depth_cat(depth_m),
                "signed_by":      signed,
                "officer_name":   officer.strip(". "),
                "officer_rank":   rank,
                "tts_items":      tts,
                "tts_cats":       list({classify_tts(t) for t in tts}),
                "wins":           wins,
                "issues":         issues,
                "actions":        actions,
                "extracted_keywords": keywords,
                "grids":          grids,
                "coords":         coords,
                "text":           sec,
            })
        return reports

    def write_wiki(self, reports: list[dict], wiki_root: Path) -> list[Path]:
        """Format and write parsed reports as Markdown files with YAML frontmatter."""
        (wiki_root / "summaries").mkdir(parents=True, exist_ok=True)
        today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        written = []
        for r in reports:
            slug = re.sub(r"[^\w-]", "-",
                f"patrol-{r['report_id'].lower()}-"
                f"{Path(r['source_doc']).stem.lower()}")[:70]
            dest = wiki_root / "summaries" / f"{slug}.md"

            # Isolate severe infrastructure markers as explicit high-threat items
            high_threat_indicators = list(set(r["extracted_keywords"].get("infrastructure_permanent", [])))

            # Extract serial numbers to store as deterministic YAML Ground Truth
            extracted_serials = list(set(re.findall(r'\b(?:PLA-[A-Z]{2}-\d{3,4}|[A-Z0-9]{2}-\w+-\d+)\b', r['text'], re.I)))

            # YAML-safe frontmatter
            fm_data: dict[str, Any] = {
                "title":          f"Report {r['report_id']} — {r['sector'] or 'General'}",
                "type":           "summary",
                "source_hash":    r.get("source_hash"),
                "report_id":      r["report_id"],
                "source_doc":     r["source_doc"],
                "unit":           r["unit"],
                "sector":         r["sector"],
                "patrol_period":  r["patrol_period"],
                "patrol_type":    r["patrol_type"],
                "strength_raw":   r["strength_raw"],
                "strength_total": r["strength_total"],
                "weather":        r["weather"],
                "depth_raw":      r["depth_raw"],
                "depth_m":        r["depth_m"],
                "depth_cat":      r["depth_cat"],
                "signed_by":      r["signed_by"],
                "officer_name":   r["officer_name"],
                "officer_rank":   r["officer_rank"],
                "grids":          r["grids"],
                "coords_latlon":  r["coords"],
                "tts_count":      len(r["tts_items"]),
                "tts_cats":       r["tts_cats"],
                "wins_count":     len(r["wins"]),
                "issues_count":   len(r["issues"]),
                "patrol_dates":   r["patrol_dates"],
                "extracted_keywords": r["extracted_keywords"],
                "high_threat_indicators": high_threat_indicators,
                "serial_numbers": extracted_serials,
                "created":        today,
                "updated":        today,
            }
            fm_lines = ["---"]
            for k, v in fm_data.items():
                if v is None:
                    continue
                if isinstance(v, str):
                    v_esc = v.replace('"', '\\"')
                    fm_lines.append(f'{k}: "{v_esc}"')
                else:
                    fm_lines.append(f"{k}: {json.dumps(v)}")
            fm_lines.append("---")

            tts_md  = "\n".join(f"- {t}" for t in r["tts_items"]) if r["tts_items"] else "_None identified_"
            wins_md = "\n".join(f"- {w}" for w in r["wins"])      if r["wins"] else "_None assessed_"
            iss_md  = "\n".join(f"- {i}" for i in r["issues"])    if r["issues"] else "_None observed_"
            act_md  = "\n".join(f"- {a}" for a in r["actions"])   if r["actions"] else "_None reported_"
            grd_md  = ", ".join(f"`{g}`" for g in r["grids"])     if r["grids"] else "—"

            # Keywords Section
            kw_md = ""
            if any(r["extracted_keywords"].values()):
                kw_sections = []
                for cat, kws in r["extracted_keywords"].items():
                    if kws:
                        kw_sections.append(f"**{cat.replace('_', ' ').title()}**: {', '.join(kws)}")
                if kw_sections:
                    kw_md = "\n## Extracted Keywords\n" + "\n".join(f"- {s}" for s in kw_sections) + "\n"

            # --- Verification Layer & LLM Synthesis ---
            exec_summary = "Executive Summary not generated."
            if WIKI_ENGINE_OK:
                from wiki_engine.compiler import OllamaClient
                try:
                    prompt = (
                        f"Write a concise 2-sentence Executive Summary for this patrol report.\n"
                        f"Unit: {fm_data['unit']}\n"
                        f"Sector: {fm_data['sector']}\n"
                        f"Strength: {fm_data['strength_raw']}\n"
                        f"Depth: {fm_data['depth_raw']}\n"
                        f"Narrative: {r['text'][:1000]}\n\n"
                        f"Rule: Do NOT invent any numbers, serial numbers, or coordinates. Use ONLY the provided data."
                    )
                    model_to_use = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
                    llm_client = OllamaClient(OLLAMA_URL, model_to_use)
                    raw_summary = run_async(llm_client.chat(
                        system="You are a strict military intelligence summarizer.",
                        user=prompt,
                        temperature=0.1
                    ))
                    
                    is_valid, verified_text = verify_llm_output(raw_summary, fm_data)
                    if is_valid:
                        exec_summary = verified_text
                    else:
                        exec_summary = f"**Verification Error:** {verified_text} | **Fallback:** Patrol by {fm_data['unit']} in {fm_data['sector']}."
                except Exception as e:
                    log.error(f"LLM Summary failed: {e}")
                    exec_summary = f"**Fallback:** Patrol by {fm_data['unit']} in {fm_data['sector']}."

            processed_narrative = auto_link_entities(r['text'], fm_data)

            body = f"""
# Patrol Report {r['report_id']} — {r['sector']}

> **Executive Summary:** {exec_summary}

## Intelligence Header
| Field | Value |
|-------|-------|
| **Report ID** | `{r['report_id']}` |
| **Report No** | `{r['report_no']}` |
| **Unit** | {r['unit']} |
| **Sector** | {r['sector']} |
| **Period** | `{r['patrol_period']}` |
| **Type** | {r['patrol_type']} |
| **Route** | {r['route']} |
| **Strength** | `{r['strength_raw']}` · Total: **{r['strength_total']}** |
| **Weather** | {r['weather']} |
| **Depth** | `{r['depth_raw']}` · {r['depth_cat']} |
| **Grids** | {grd_md} |
| **Signed By** | {r['signed_by']} |

## Tell-Tale Signs of PLA Activity
{tts_md}

## PLA Capabilities Identified
{wins_md}

## Route
{r['route']}

## Issues Observed
{iss_md}

## Actions Taken
{act_md}
{kw_md}

## Full Narrative
{processed_narrative}

## Source Document
[{r['source_doc']}]
"""
            dest.write_text("\n".join(fm_lines) + "\n" + body.strip())
            log.info(f"Wrote wiki summary to {dest}")
            written.append(dest)
        return written


_DP = DirectParser()


def _depth_metres(s: str) -> float:
    if not s:
        return 0.0
    lt = re.search(r'<\s*(\d+(?:\.\d+)?)\s*(m|metres?|km)', s, re.I)
    if lt:
        v = float(lt.group(1))
        return v * 1000 if 'km' in lt.group(2).lower() else v
    nums = re.findall(r'(\d+(?:\.\d+)?)', s)
    if not nums:
        return 0.0
    v = float(nums[0])
    if 'km' in s.lower():
        v *= 1000
    return v


def _parse_dates(period: str) -> list[str]:
    mon = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
           "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
    out = []
    for m in re.finditer(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", period):
        d, ms, y = m.groups()
        mo = mon.get(ms[:3].lower())
        if mo:
            try:
                out.append(f"{y}-{mo}-{int(d):02d}")
            except ValueError:
                pass
    return out


def direct_ingest(src: Path, force: bool = False) -> dict[str, Any]:
    WIKI_ROOT.mkdir(exist_ok=True)
    (WIKI_ROOT / "log.md").touch()
    text    = _DP.read(src)
    if not text.strip():
        return {"error": f"Cannot read {src.name}"}
    if text.startswith("DOCX_ERROR:"):
        return {"error": text}
        
    # Step 1: Content-Based Deduplication
    text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    summaries_dir = WIKI_ROOT / "summaries"
    if not force and summaries_dir.exists():
        for md in summaries_dir.glob("*.md"):
            try:
                fm, _ = _fm(md.read_text(encoding="utf-8", errors="ignore"))
                if fm.get("source_hash") == text_hash:
                    return {
                        "error": f"Duplicate content match found for {src.name}", 
                        "source": str(src), "reports": [], "written": 0, "method": "direct_skipped"
                    }
            except Exception:
                pass

    reports = _DP.parse(text, src.name)
    for r in reports:
        r["source_hash"] = text_hash
    if not reports:
        return {"error": f"No patrol reports found in {src.name}"}
    written = _DP.write_wiki(reports, WIKI_ROOT)
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry   = (
        f"\n## [{today}] ingest | {src.name}\n"
        f"- Articles: {len(written)}\n"
        f"- Reports: {', '.join(r['report_id'] for r in reports)}\n"
        f"- Method: Direct\n"
    )
    with open(WIKI_ROOT / "log.md", "a") as f:
        f.write(entry)
    load_wiki.clear()
    return {
        "source":   str(src),
        "reports":  [r["report_id"] for r in reports],
        "written":  len(written),
        "method":   "direct",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=20, show_spinner=False)
def load_wiki(root_str: str) -> list[dict[str, Any]]:
    root = Path(root_str)
    arts: list[dict[str, Any]] = []
    if not root.exists():
        return arts
    for md in root.rglob("*.md"):
        rel = str(md.relative_to(root))
        if ".meta" in rel or md.name in ("SCHEMA.md", "log.md") or "derived" in rel:
            continue
        try:
            content  = md.read_text(encoding="utf-8", errors="ignore")
            fm, body = _fm(content)
            fm["_path"] = rel
            fm["_stem"] = md.stem
            fm["_body"] = body
            arts.append(fm)
        except OSError:
            pass
    return arts


def _wiki() -> list[dict[str, Any]]:
    return load_wiki(str(WIKI_ROOT))


def _fm(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter robustly. Tries yaml first, then regex."""
    if not content.startswith("---"):
        return {}, content
    pts = content.split("---", 2)
    if len(pts) < 3:
        return {}, content
    # Always try yaml (PyYAML is in requirements.txt)
    try:
        import yaml as _yaml  # local import avoids top-level flag dependency
        parsed = _yaml.safe_load(pts[1])
        if isinstance(parsed, dict):
            return parsed, pts[2]
    except Exception:
        pass
    # Regex fallback for edge cases
    fm: dict[str, Any] = {}
    for ln in pts[1].strip().split("\n"):
        m = re.match(r'^([\w_]+)\s*:\s*(.+)$', ln)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        elif v.startswith('[') or v.startswith('{'):
            try:
                v = json.loads(v)
            except Exception:
                pass
        elif re.fullmatch(r'-?\d+(\.\d+)?', v):
            try:
                v = float(v) if '.' in v else int(v)
            except ValueError:
                pass
        fm[k] = v
    return fm, pts[2]


# ═══════════════════════════════════════════════════════════════════════════════
# COMMIT APPROVED ANSWER TO WIKI
# ═══════════════════════════════════════════════════════════════════════════════

def _commit_to_wiki(pend: dict[str, Any]) -> None:
    """Write an approved analyst answer to wiki/derived/ as a Markdown article."""
    (WIKI_ROOT / "derived").mkdir(exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug  = re.sub(r"\W+", "-", pend["question"].lower())[:50]
    slug  = f"approved-{today}-{slug}.md"
    dest  = WIKI_ROOT / "derived" / slug

    fm_str = (
        f"---\ntitle: \"Approved Analysis: {pend['question'][:60]}\"\n"
        f"type: derived\nclassification: RESTRICTED\n"
        f"approved: \"{today}\"\n"
        f"sources: {json.dumps(pend.get('sources',[]))}\n---\n"
    )
    body = (
        f"\n# Approved Analysis\n\n"
        f"**Question:** {pend['question']}\n\n"
        f"**Answer:**\n\n{pend['answer']}\n\n"
        f"**Sources:** {', '.join(pend.get('sources',[])[:6])}\n\n"
        f"*Approved: {today}*\n"
    )
    dest.write_text(fm_str + body)

    # Append to log
    lp = WIKI_ROOT / "log.md"
    lp.touch()
    with open(lp, "a") as f:
        f.write(
            f"\n## [{today}] approved | {pend['question'][:60]}\n"
            f"- Committed: {slug}\n"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK GRAPH  —  shared between Link Analysis and Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30, show_spinner=False)
def build_graph(root_str: str) -> dict[str, Any]:
    """
    Build a networkx graph from all wiki articles.
    Nodes: reports, officers, units, sectors, TTS categories, issues.
    This graph drives both Link Analysis (visual) and Pattern Analysis (stats).
    """
    if not HAS_NX:
        return {"nodes": [], "edges": [], "hubs": [], "G_data": {}}

    arts = load_wiki(root_str)
    G    = nx.Graph()

    for a in arts:
        rid = a.get("report_id")
        if not rid:
            continue

        period = a.get("patrol_period", "")
        G.add_node(rid, label=f"{rid}", ntype="report",
                   period=period[:20],
                   sector=a.get("sector",""),
                   depth_cat=a.get("depth_cat",""),
                   unit=a.get("unit",""),
                   strength=a.get("strength_raw",""),
                   depth_raw=a.get("depth_raw",""),
                   signed=a.get("signed_by",""),
                   issues=int(a.get("issues_count",0) or 0))

        # Officer node
        ofr = (a.get("officer_name") or "").strip(". ")
        rnk = a.get("officer_rank","")
        if ofr and len(ofr) > 3:
            onid = f"OFF::{ofr}"
            G.add_node(onid, label=ofr, ntype="officer",
                       rank=rnk, signed_reports=[rid])
            G.add_edge(rid, onid, rel="signed_by")

        # Unit node — normalise
        unit = re.sub(r'\s*\(.*\)', '', a.get("unit","")).strip()
        unit = re.sub(r'\s+det.*$', '', unit, flags=re.I).strip()
        if unit:
            uid = f"UNIT::{unit[:35]}"
            G.add_node(uid, label=unit[:35], ntype="unit")
            G.add_edge(rid, uid, rel="unit")

        # Sector node
        sec = (a.get("sector","") or "")[:45]
        if sec:
            sid = f"SEC::{sec}"
            G.add_node(sid, label=sec, ntype="sector")
            G.add_edge(rid, sid, rel="sector")

        # TTS category nodes
        tts_cats = a.get("tts_cats")
        if isinstance(tts_cats, list):
            for cat in tts_cats:
                tnid = f"TTS::{cat}"
                G.add_node(tnid, label=cat, ntype="tts")
                G.add_edge(rid, tnid, rel="tts_observed")
        else:
            # Re-derive from body
            body = a.get("_body", "")
            m_t  = re.search(r'## Tell-Tale Signs.*?\n((?:- .+\n?)+)', body, re.S)
            if m_t:
                seen_c: set[str] = set()
                for item in re.findall(r'- (.+)', m_t.group(1)):
                    cat = classify_tts(item.strip("* "))
                    if cat not in seen_c:
                        tnid = f"TTS::{cat}"
                        G.add_node(tnid, label=cat, ntype="tts")
                        G.add_edge(rid, tnid, rel="tts_observed")
                        seen_c.add(cat)

        # Issue nodes (high-value intel flags)
        body = a.get("_body", "")
        mi   = re.search(r'## Issues Observed\n((?:- .+\n?)+)', body, re.S)
        if mi:
            for il in re.findall(r'- (.+)', mi.group(1)):
                il = il.strip("* ")
                if len(il) > 10:
                    ikey = hashlib.md5(il.encode()).hexdigest()[:8]
                    inid = f"ISS::{ikey}"
                    G.add_node(inid, label=il[:40], ntype="issue", full=il)
                    G.add_edge(rid, inid, rel="issue_raised")

    if G.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "hubs": [], "n_nodes": 0, "n_edges": 0}

    pos = nx.spring_layout(G, k=2.2, iterations=80, seed=42)

    nodes = [
        {
            "id":    n,
            "x":     pos[n][0],
            "y":     pos[n][1],
            "label": G.nodes[n].get("label", n),
            "ntype": G.nodes[n].get("ntype", "other"),
            "deg":   G.degree(n),
            "color": NODE_COL.get(G.nodes[n].get("ntype","other"), "#6b7280"),
            "size":  NODE_COL.get(G.nodes[n].get("ntype","other"), "#6b7280"),
            "meta":  {k: v for k, v in G.nodes[n].items()
                      if k not in ("label","ntype","color","size")},
        }
        for n in pos
    ]
    edges = [
        {"x0": pos[u][0], "y0": pos[u][1],
         "x1": pos[v][0], "y1": pos[v][1],
         "rel": G.edges[u, v].get("rel", "")}
        for u, v in G.edges() if u in pos and v in pos
    ]
    hubs = sorted(G.degree(), key=lambda x: -x[1])[:15]

    # Pattern data derived from graph (used by Tab 2)
    pattern: dict[str, Any] = {}

    # Officer → reports map
    off_map: dict[str, list[str]] = {}
    for u, v in G.edges():
        if G.edges[u,v].get("rel") == "signed_by":
            # u = report, v = officer (or vice versa)
            rep  = u if G.nodes[u].get("ntype") == "report"   else v
            ofr2 = v if G.nodes[v].get("ntype") == "officer"  else u
            lbl  = G.nodes.get(ofr2, {}).get("label", ofr2)
            off_map.setdefault(lbl, []).append(rep)
    pattern["officer_reports"] = off_map

    # TTS across reports
    tts_map: dict[str, list[str]] = {}
    for u, v in G.edges():
        if G.edges[u,v].get("rel") == "tts_observed":
            rep  = u if G.nodes[u].get("ntype") == "report" else v
            tts2 = v if G.nodes[v].get("ntype") == "tts"    else u
            cat  = G.nodes.get(tts2, {}).get("label", "")
            if cat:
                tts_map.setdefault(cat, []).append(rep)
    pattern["tts_reports"] = tts_map

    # Sector → reports
    sec_map: dict[str, list[str]] = {}
    for u, v in G.edges():
        if G.edges[u,v].get("rel") == "sector":
            rep = u if G.nodes[u].get("ntype") == "report" else v
            sec = v if G.nodes[v].get("ntype") == "sector" else u
            lbl = G.nodes.get(sec, {}).get("label", "")
            if lbl:
                sec_map.setdefault(lbl, []).append(rep)
    pattern["sector_reports"] = sec_map

    return {
        "nodes":   nodes,
        "edges":   edges,
        "hubs":    hubs,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "pattern": pattern,
    }


SYNONYMS = {
    "strength": ["troops", "personnel", "soldiers", "manpower", "size", "men", "team"],
    "depth": ["ingress", "transgress", "km", "metres", "meters", "deep", "shallow"],
    "tts": ["tell tale", "sign", "indicator", "boot", "vehicle", "drone", "camp", "fire", "debris"],
    "officer": ["commander", "capt", "major", "lt", "col", "signed"],
    "unit": ["regiment", "battalion", "forces", "army"],
    "weather": ["climate", "rain", "snow", "temperature", "overcast", "clear"],
}

def _expand_synonyms(words: list[str]) -> list[str]:
    expanded = set(words)
    for w in words:
        for key, syn_list in SYNONYMS.items():
            if w == key or w in syn_list:
                expanded.add(key)
                expanded.update(syn_list)
    return list(expanded)

def _fuzzy_match(word: str, text: str, threshold: int = 80) -> bool:
    try:
        from thefuzz import fuzz
    except ImportError:
        return word in text
    # Fast short-circuit exact match
    if word in text:
        return True
    # Split text and check token ratios for typos
    for t in text.split():
        if fuzz.ratio(word, t) > threshold:
            return True
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# WIKI LOOKUP  —  fast exact match, no LLM
# ═══════════════════════════════════════════════════════════════════════════════

def wiki_lookup(question: str) -> dict[str, Any]:
    """
    Instant verbatim extraction.
    Factual/to-point query → to-point answer.
    No LLM, no synthesis.
    """
    arts = _wiki()
    if not arts:
        return {
            "answer": (
                "⚠ **Wiki is empty.**\n\n"
                "Upload patrol report files and click **🧠 INGEST SELECTED FILES** "
                "in the chat tab to populate the intelligence database."
            ),
            "sources": [], "mode": "LOOKUP — EMPTY",
        }

    q    = question.lower()
    # 2-char min so report IDs like "K1","S3","S5" are not dropped
    words = [w for w in re.findall(r'\b[\w-]{2,}\b', q)
             if w not in {"a","an","the","is","it","in","on","at","to","of",
                          "and","or","for","not","with","from","that","this",
                          "what","when","were","have","been","will","they",
                          "their","about","which","were","show","list",
                          "give","tell","describe","during","all",
                          "found","where","who","how","many","any","are","do","does","did","has","had"}]

    target_rids = [w.upper() for w in words if re.match(r'^[a-z]\d+$', w)]
    
    expanded_words = _expand_synonyms(words)

    is_strength = any(w in q for w in
                      ["strength","personnel","soldiers","troops","how many",
                       "manpower","total","ranks","officers","members"])
    is_depth    = any(w in q for w in
                      ["depth","ingress","transgress","deep","shallow","km","metres","meters"])
    is_tts      = any(w in q for w in
                      ["tts","tell tale","tell-tale","sign","indicator","evidence","boot","vehicle","drone"])
    is_officer  = any(w in q for w in
                      ["signed","officer","capt","major","captain","lt","thapa",
                       "bhandari","katoch","menon","gurung","patil","balwinder"])
    is_unit     = any(w in q for w in
                      ["unit","regiment","rifles","gorkha","madras","maratha","punjab","jak"])
    is_sector   = any(w in q for w in
                      ["sector","kibithoo","walong","tawang","chaglagam","yangtse",
                       "mechuka","anjaw","bum la","zimithang"])
    is_weather  = any(w in q for w in
                      ["weather","temperature","cloud","snow","rain","clear","overcast","celsius"])
    is_grid_q   = any(w in q for w in ["grid","nk ","coord","mgrs","where tts"])
    is_period   = any(w in q for w in
                      ["period","patrol date","when was","march","april","may","date of"])
    is_route    = any(w in q for w in ["route","path","track","via"])
    is_serial   = any(w in q for w in ["serial", "equipment", "pla-zh-882", "id"])

    found_serials: set[str] = set([s.upper() for s in re.findall(r'\b(PLA-[A-Z]{2}-\d{3,4})\b', question, re.I)])

    field_hits:    list[str] = []
    sent_hits:     list[dict] = []
    sources_used:  list[str] = []

    art_scores = []
    for a in arts:
        rid     = a.get("report_id", a["_stem"])
        if target_rids and str(rid).upper() not in target_rids:
            continue
        content = a.get("_body", "")
        
        # Upgraded semantic/fuzzy scoring
        score = 0
        for w in expanded_words:
            if _fuzzy_match(w, content.lower()):
                score += 3 if (len(w) > 4 and ('-' in w or any(c.isdigit() for c in w))) else 1
                
        if score > 0:
            art_scores.append((score, a))

    if art_scores:
        art_scores.sort(key=lambda x: -x[0])
        max_score = art_scores[0][0]
        rare_in_query = any(len(w) > 4 and ('-' in w or any(c.isdigit() for c in w)) for w in words)
        if rare_in_query and max_score < 3:
            best_arts = []
        else:
            best_arts = [a for s, a in art_scores if s >= max_score * 0.7][:4]
    else:
        best_arts = []

    for a in best_arts:
        rid     = a.get("report_id", a["_stem"])
        content = a.get("_body", "")
        body    = a.get("_body", "")

        def _field(k: str, label: str, color: str = ""):
            val = a.get(k)
            tag = f" `[{color}]`" if color and val else ""
            display_val = str(val)[:80] if val else "Not recorded"
            field_hits.append(
                f"**[{rid}]** **{label}**: `{display_val}`{tag}  "
                f"*{(a.get('sector','') or '')[:30]}*"
            )

        if is_strength:
            _field("strength_raw",  "Strength")
            if a.get("strength_total"):
                field_hits.append(
                    f"**[{rid}]** **Total strength**: **{a['strength_total']}** personnel"
                )

        if is_depth:
            dc = a.get("depth_cat","")
            _field("depth_raw", "Depth of Transgression", dc)

        if is_officer:
            _field("signed_by",   "Signed By")
            _field("officer_rank","Officer Rank")

        if is_unit:
            _field("unit", "Unit")

        if is_sector:
            _field("sector",  "Sector")
            _field("patrol_period", "Period")

        if is_tts:
            m_t = re.search(r'## Tell-Tale Signs.*?\n((?:- .+\n?)+)', body, re.S)
            if m_t:
                items = re.findall(r'- (.+)', m_t.group(1))
                if items:
                    field_hits.append(
                        f"**[{rid}]** **TTS ({len(items)} items)**  *{a.get('sector','')[:30]}*:\n"
                        + "\n".join(f"  - {it.strip('* ')}" for it in items)
                    )

        if is_weather and a.get("weather"):
            field_hits.append(
                f"**[{rid}]** **Weather**: `{a['weather']}`  *{a.get('sector','')[:30]}*"
            )

        if is_grid_q:
            grids_val = a.get("grids")
            if isinstance(grids_val, list) and grids_val:
                field_hits.append(
                    f"**[{rid}]** **Grids**: {', '.join(grids_val)}  *{a.get('sector','')[:30]}*"
                )

        if is_period and a.get("patrol_period"):
            field_hits.append(
                f"**[{rid}]** **Patrol Period**: `{a['patrol_period']}`"
            )

        if is_route and a.get("route"):
            field_hits.append(
                f"**[{rid}]** **Route**: {a['route'][:80]}"
            )

        if is_serial:
            # Search for serial number pattern in the body, especially in 'wins' section
            m_wins = re.search(r'## (?:PLA Capabilities|Wins) Identified.*?\n((?:- .+\n?)+)', body, re.S | re.I)
            scan_text = m_wins.group(1) if m_wins else body

            # Check if other query keywords are present to ensure relevance
            non_serial_words = [w for w in words if w not in ['serial', 'number', 'equipment', 'id', 'pla-zh-882']]
            if not non_serial_words or any(w in content.lower() for w in non_serial_words):
                for serial_match in re.finditer(r'\b(PLA-[A-Z]{2}-\d{3,4})\b', scan_text, re.I):
                    serial_code = serial_match.group(0).upper()
                    found_serials.add(serial_code)
                    field_hits.append(
                        f"**[{rid}]** **Equipment Serial**: `{serial_code}`  "
                        f"*{a.get('sector','')[:30]}*"
                    )

        # Sentence scan
        for sent in re.split(r'[.!\n]', content):
            s = sent.strip()
            if len(s) < 15:
                continue
            n = sum(1 for w in expanded_words if _fuzzy_match(w, s.lower()))
            if n < 2:
                continue
            bonus = 12 if (is_strength and re.search(r'\b\d+\b', s)) else 0
            bonus += 8  if (is_depth    and DEPTH_RE.search(s))  else 0
            bonus += 15 if (is_serial and re.search(r'\b(PLA-[A-Z]{2}-\d{3,4})\b', s, re.I)) else 0
            sent_hits.append({"text": s[:200], "src": a["_path"], "score": n + bonus})
            if a["_path"] not in sources_used:
                sources_used.append(a["_path"])

    sent_hits.sort(key=lambda x: -x["score"])
    top = sent_hits[:5]

    if not field_hits and not top:
        return {
            "answer": (
                f"No match for: *\"{question}\"*\n\n"
                f"Wiki: {len(arts)} articles. Tokens: "
                f"`{'`, `'.join(words[:5])}`\n\n"
                "Ensure reports are ingested or broaden your query."
            ),
            "sources": [], "mode": "LOOKUP — NO MATCH",
        }

    # Build concise answer
    parts: list[str] = []
    parts.append("### ⚡ Instant Intelligence Retrieval")
    parts.append("> **Analysis Mode:** Lexical & Fuzzy Knowledge Base Scan\n")
    
    if found_serials and WIKI_ENGINE_OK:
        db_hits = []
        for code in found_serials:
            eq_info = lookup_pla_equipment(code)
            if eq_info:
                db_hits.append(
                    f"- **{code}** — {eq_info.get('type', 'Unknown Equipment')} "
                    f"(`{eq_info.get('classification', 'Unclassified')}`)\n"
                    f"  *Description:* {eq_info.get('description', '')}\n"
                    f"  *Common Uses:* {', '.join(eq_info.get('common_uses', []))}"
                )
        if db_hits:
            parts.append("#### 🛡️ PLA Equipment Reference Database")
            parts.extend(db_hits)
            parts.append("")

    if field_hits:
        parts.append("#### 📌 Structured Data Matches")
        for hit in field_hits[:10]:
            parts.append(f"- {hit}")
            
    if top and not field_hits:
        parts.append("#### 📄 Contextual Excerpts")
        for h in top:
            parts.append(f"> *\"{h['text']}\"*\n> **Provenance:** `{h['src']}`\n")
            
    return {
        "answer":  "\n\n".join(parts),
        "sources": sources_used[:6],
        "mode":    "WIKI LOOKUP",
    }


def _analyse_query(question: str) -> dict[str, Any]:
    """
    Generate a full structured analysis report from compiled wiki data.
    Synthesises across all reports, no LLM needed.
    """
    arts  = _wiki()
    rpts  = [a for a in arts if a.get("report_id")]
    graph = build_graph(str(WIKI_ROOT))
    pat   = graph.get("pattern", {})

    if not rpts:
        return wiki_lookup(question)

    q = question.lower()
    lines: list[str] = []

    # ── Determine analysis type ──────────────────────────────────────────────
    is_threat  = any(w in q for w in ["threat","assess","overall","situation","pla","pattern","analyze","analyse"])
    is_tts_q   = any(w in q for w in ["tts","tell tale","tell-tale","sign","indicator"])
    is_depth_q = any(w in q for w in ["depth","ingress","transgress","deep"])
    is_pers_q  = any(w in q for w in ["personnel","officer","signed","who","thapa","bhandari",
                                       "katoch","menon","gurung","patil"])
    is_sector_q= any(w in q for w in ["sector","where","which sector","area","axis"])
    is_drone_q = any(w in q for w in ["drone","uav","aerial","air"])
    is_equip_q = any(w in q for w in ["equipment","serial","capability","weapons","pla-"])

    # ── Always compute base metrics ──────────────────────────────────────────
    max_depth  = max((float(a.get("depth_m",0) or 0) for a in rpts), default=0)
    tts_all_items: list[str] = []
    tts_by_cat: dict[str, int] = {}
    for a in rpts:
        body = a.get("_body","")
        m_t  = re.search(r'## Tell-Tale Signs.*?\n((?:- .+\n?)+)', body, re.S)
        if m_t:
            for item in re.findall(r'- (.+)', m_t.group(1)):
                item = item.strip("* ")
                if item:
                    tts_all_items.append(item)
                    cat = classify_tts(item)
                    tts_by_cat[cat] = tts_by_cat.get(cat, 0) + 1

    drone_cnt = tts_by_cat.get("Drone Activity", 0)
    deep_cnt  = sum(1 for a in rpts if float(a.get("depth_m",0) or 0) >= 1500)
    infra_cnt = sum(1 for a in rpts if a.get("high_threat_indicators"))

    # ── Threat assessment ────────────────────────────────────────────────────
    if max_depth >= 2500 or deep_cnt >= 2 or drone_cnt >= 2 or infra_cnt >= 1:
        threat, tlvl_col = "🔴 HIGH", "#ef4444"
        t_reason = "permanent infrastructure buildup, multiple deep ingression events, and/or repeated drone surveillance"
    elif max_depth >= 1000:
        threat, tlvl_col = "🟠 MODERATE-HIGH", "#f97316"
        t_reason = "ingression exceeding 1 km threshold in one or more sectors"
    elif max_depth >= 400:
        threat, tlvl_col = "🟡 MODERATE", "#f59e0b"
        t_reason = "consistent shallow-to-moderate TTS with no persistent infrastructure"
    else:
        threat, tlvl_col = "🟢 LOW", "#22c55e"
        t_reason = "surface-level TTS, no significant ingression"

    # ── Build report ─────────────────────────────────────────────────────────
    if is_threat or (not is_tts_q and not is_depth_q and not is_pers_q
                     and not is_sector_q and not is_drone_q):
        lines.append(f"## THREAT ASSESSMENT: {threat}")
        lines.append(f"**Basis:** {t_reason}.")
        lines.append("")

    lines.append(f"**Corpus:** {len(rpts)} patrol reports across "
                 f"{len({a.get('sector','') for a in rpts})} sector(s) · "
                 f"Period: {min((a.get('patrol_dates',[['—']])[0] for a in rpts if a.get('patrol_dates')), default='—')} "
                 f"to {max((a.get('patrol_dates',[['—']])[-1] for a in rpts if a.get('patrol_dates')), default='—')}")
    lines.append("")

    if is_tts_q or is_threat:
        lines.append("### Tell-Tale Sign Analysis")
        top_tts = sorted(tts_by_cat.items(), key=lambda x: -x[1])
        for cat, cnt in top_tts:
            bar = "█" * min(cnt * 3, 15)
            lines.append(f"- **{cat}**: {cnt} occurrence(s)  `{bar}`")
        lines.append(f"\n**Total TTS logged:** {len(tts_all_items)} across all reports")
        lines.append("")

    if is_depth_q or is_threat:
        lines.append("### Ingression Depth by Report")
        for a in sorted(rpts, key=lambda x: float(x.get("depth_m",0) or 0), reverse=True):
            dm  = float(a.get("depth_m",0) or 0)
            cat = a.get("depth_cat","—")
            col = "🔴" if cat=="Deep" else ("🟡" if cat=="Moderate" else "🟢")
            lines.append(
                f"- **{a.get('report_id','')}** [{a.get('sector','')[:25]}]  "
                f"{col} `{a.get('depth_raw','—')}` ({dm:.0f} m)"
            )
        lines.append(f"\n**Deepest:** {max_depth:.0f} m — "
                     f"Report {max((rpts), key=lambda x: float(x.get('depth_m',0) or 0)).get('report_id','')}")
        lines.append("")

    if is_drone_q or (is_threat and drone_cnt):
        lines.append("### Drone / Aerial Activity")
        if drone_cnt:
            drone_rpts = [a.get("report_id","") for a in rpts
                          if any("drone" in i.lower() or "uav" in i.lower()
                                 for i in (a.get("tts_cats") or []))]
            lines.append(f"**{drone_cnt} drone TTS** recorded in reports: "
                         f"{', '.join(drone_rpts)}")
            lines.append("⚠ Enhanced aerial surveillance posture recommended.")
        else:
            lines.append("No drone TTS recorded in current corpus.")
        lines.append("")

    if is_pers_q or is_threat:
        lines.append("### Patrol Leadership")
        off_map = pat.get("officer_reports", {})
        if off_map:
            for ofr, rep_list in sorted(off_map.items(), key=lambda x: -len(x[1])):
                lines.append(f"- **{ofr}**: {', '.join(rep_list)}")
        else:
            for a in rpts:
                lines.append(f"- **{a.get('signed_by','—')}** — "
                             f"Report {a.get('report_id','')} · {a.get('sector','')[:25]}")
        lines.append("")

    if is_sector_q or is_threat:
        lines.append("### Sector Assessment")
        sec_map = pat.get("sector_reports", {})
        all_issues_count = sum(int(a.get("issues_count",0) or 0) for a in rpts)
        for sec, rep_list in sorted(sec_map.items(), key=lambda x: -len(x[1])):
            sec_rpts  = [a for a in rpts if a.get("sector","") == sec.replace("SEC::","")]
            max_d     = max((float(a.get("depth_m",0) or 0) for a in sec_rpts), default=0)
            cat       = depth_cat(max_d)
            issues    = sum(int(a.get("issues_count",0) or 0) for a in sec_rpts)
            lines.append(
                f"- **{sec.replace('SEC::','')}**  "
                f"{len(rep_list)} report(s) · Max depth {max_d:.0f} m [{cat}]"
                + (f" · ⚠ {issues} issues" if issues else "")
            )
        lines.append("")

    # Analyst flags
    all_issues_list: list[str] = []
    for a in rpts:
        body = a.get("_body","")
        mi   = re.search(r'## Issues Observed\n((?:- .+\n?)+)', body, re.S)
        if mi:
            for il in re.findall(r'- (.+)', mi.group(1)):
                il = il.strip("* ")
                if len(il) > 10:
                    all_issues_list.append(f"[{a.get('report_id','')}] {il}")

    if all_issues_list and (is_threat or is_sector_q):
        lines.append("### Analyst Flags / Issues Raised")
        for iss in all_issues_list[:6]:
            lines.append(f"- {iss}")
        lines.append("")

    # Equipment Profile
    if is_equip_q or is_threat:
        found_serials_analysis = set()
        for a in rpts:
            body = a.get("_body", "")
            for m in re.finditer(r'\b(PLA-[A-Z]{2}-\d{3,4})\b', body, re.I):
                found_serials_analysis.add(m.group(1).upper())
        
        if found_serials_analysis and WIKI_ENGINE_OK:
            equip_lines = ["### PLA Equipment Profile"]
            has_known = False
            for code in found_serials_analysis:
                eq_info = lookup_pla_equipment(code)
                if eq_info:
                    equip_lines.append(f"- **{code}** ({eq_info.get('classification', 'Unknown')}): {eq_info.get('type', '')}")
                    has_known = True
            if has_known:
                lines.extend(equip_lines)
                lines.append("")

    # Recommendations
    if is_threat:
        lines.append("### Recommended Actions")
        if drone_cnt >= 2:
            lines.append("- **Immediate**: Deploy counter-drone measures in Yangtse–Zimithang axis")
        if deep_cnt >= 1:
            lines.append("- **Priority**: Reinforce forward OPs in Chaglagam–Anjaw sector (deepest ingression 2.5–3.0 km)")
        lines.append("- Increase night patrol frequency in sectors with repeated TTS")
        lines.append("- Cross-reference TTS dates to identify PLA activity patterns")

    return {
        "answer":  "\n".join(lines),
        "sources": [a["_path"] for a in rpts],
        "mode":    "ANALYSIS REPORT",
    }


def is_analysis_query(q: str) -> bool:
    """Decide: to-point lookup vs full analysis."""
    q_low = q.lower()
    
    # Force Lookup trigger: specific Report ID AND quantitative keyword
    report_re = re.compile(r'\b[a-z]\d+\b', re.IGNORECASE)
    quant_keywords = ["strength", "count", "how many", "members"]
    if report_re.search(q_low) and any(kw in q_low for kw in quant_keywords):
        return False

    analysis_triggers = [
        "analys", "assess", "overall", "summarise", "summarize",
        "pattern", "trend", "compare", "across all", "threat level",
        "what is the situation", "intelligence brief", "tell me about",
        "synthesize", "synthesise", "alert"
    ]
    point_triggers = [
        "what was", "who signed", "which report", "what is the strength",
        "how many", "show me", "list", "when did", "when was", "what are the grids",
        "what type of patrol", "what unit", "what weather", "weather during",
        "depth of", "grid ref", "patrol period", "route of", "strength of",
        "who commanded", "signed by", "how many soldiers", "what happened in report",
        "serial number", "equipment serial", "what is the serial", 
        "strength", "officer count"
    ]
    if any(t in q_low for t in point_triggers):
        return False
    if any(t in q_low for t in analysis_triggers):
        return True
    # Default: long query = analysis, short = lookup
    return len(q.split()) > 8


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(question: str, answer: str, sources: list[str]) -> tuple[float, list[str]]:
    score, crits = 0.0, []
    if re.search(r'\b\d{1,5}\b', answer):
        score += 0.20
    else:
        crits.append("No specific numbers in answer")
    if   len(sources) >= 3: score += 0.20
    elif sources:           score += 0.10; crits.append(f"Only {len(sources)} source(s)")
    else:                   crits.append("No sources cited")
    terms = [w for w in re.findall(r'\b\w{5,}\b', question.lower())
             if w not in {"which","where","about","these","those","there"}]
    if terms:
        r = sum(1 for t in terms if t in answer.lower()) / len(terms)
        score += r * 0.40
        if r < 0.5: crits.append(f"Only {r:.0%} of query terms addressed")
    else:
        score += 0.40
    score += 0.20 if sources else 0.0
    return min(score, 1.0), crits


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama auto-detect
# ═══════════════════════════════════════════════════════════════════════════════

_PREF_MODELS = [
    "llama3.1:8b","llama3.1:70b","llama3:8b","llama3:70b",
    "mistral:7b","mistral:latest","qwen2.5:7b",
    "phi3:mini","llama2:13b","llama2:7b","llama3.2:3b",
]


def detect_model() -> tuple[str, list[str]]:
    if not WIKI_ENGINE_OK:
        return "", []
    try:
        models = run_async(OllamaClient(OLLAMA_URL, "").models(), timeout=4)
        if not models:
            return "", []
        for pref in _PREF_MODELS:
            for m in models:
                if pref.split(":")[0] in m:
                    return m, models
        return models[0], models
    except Exception:
        return "", []



# ═══════════════════════════════════════════════════════════════════════════════
# LLM STREAMING  —  real-time response queue
# ═══════════════════════════════════════════════════════════════════════════════

def stream_llm_queue(pipeline, question: str, hops: int = 2):
    """
    Stream LLM query results in real-time.
    Yields chunks as they arrive from the pipeline.
    """
    try:
        result = run_async(
            pipeline.query(question, hops=hops, intel_mode=True),
            timeout=3600,
        )
        answer = result.get("answer", "No answer generated.")
        # Yield in smaller chunks for streaming effect
        for chunk in answer.split("\n"):
            if chunk.strip():
                yield chunk + "\n"
    except asyncio.TimeoutError:
        yield "⚠ **Query timed out** (3600s). Try a simpler question.\n"
    except Exception as e:
        yield f"⚠ **LLM Error**: {str(e)[:200]}\n"


def get_pipeline(model: str) -> Any:
    if not WIKI_ENGINE_OK:
        return None
    if st.session_state.get("pipeline") is None:
        try:
            import wiki_engine.pipeline as _pm
            _pm.WIKI_ROOT = WIKI_ROOT
            _pm.RAW_ROOT  = RAW_ROOT
            pl = WikiPipeline(OLLAMA_URL, model)
            for attr in ("backlinks","linter","indexer","embedder"):
                getattr(pl, attr).wiki_root = WIKI_ROOT
            pl.embedder._store = WIKI_ROOT / ".meta" / "embeddings.json"
            st.session_state["pipeline"] = pl
        except Exception as e:
            st.error(f"Pipeline: {e}")
    return st.session_state.get("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SENTINEL-LAC",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS  —  Military dark situation room
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset + base ── */
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#ffffff;color:#1e293b;font-size:18px}
.stApp{background:#ffffff}
*{scrollbar-width:thin;scrollbar-color:#cbd5e1 #ffffff}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}

/* ── Classification banner ── */
.banner{
  background:linear-gradient(90deg,#78350f 0%,#d97706 40%,#f59e0b 50%,#d97706 60%,#78350f 100%);
  color:#000;text-align:center;font-family:'JetBrains Mono',monospace;
  font-weight:800;font-size:24px;letter-spacing:5px;padding:16px 0;border-radius:4px;
  margin-bottom:14px;box-shadow:0 4px 24px rgba(245,158,11,.25);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e2e8f0}
section[data-testid="stSidebar"] .stMetric{
  background:#f8fafc;border:1px solid #e2e8f0;border-radius:5px;padding:10px;margin-bottom:8px;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"]{
  color:#1e293b !important;font-family:'JetBrains Mono',monospace !important;font-size:32px !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{
  color:#64748b !important;font-size:16px !important;text-transform:uppercase;letter-spacing:1px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:#f8fafc;border-bottom:1px solid #e2e8f0;padding:0 8px}
.stTabs [data-baseweb="tab"]{
  background:transparent;color:#334155;padding:16px 24px;
  font-family:'JetBrains Mono',monospace;font-size:18px;letter-spacing:1.5px;
  border-radius:4px 4px 0 0;transition:all .15s;
}
.stTabs [data-baseweb="tab"]:hover{color:#1e293b;background:#f1f5f9}
.stTabs [aria-selected="true"]{
  color:#1d4ed8 !important;
  background:linear-gradient(180deg,transparent,rgba(59,130,246,.05)) !important;
  border-bottom:2px solid #3b82f6 !important;
}

/* ── Buttons ── */
.stButton>button{
  background:linear-gradient(135deg,#e0ecff,#dbeafe);border:1px solid #93c5fd;color:#0f172a;border-radius:6px;
  font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;letter-spacing:.5px;
  transition:all .15s;padding:12px 18px;
}
.stButton>button:hover{background:linear-gradient(135deg,#dbeafe,#bfdbfe);border-color:#3b82f6;box-shadow:0 0 12px rgba(59,130,246,.2)}
.stButton>button:active{background:#bfdbfe}
button[kind="primary"]{
  background:linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
  border-color:#2563eb !important;color:#ffffff !important;
}
button[kind="primary"]:hover{box-shadow:0 0 20px rgba(59,130,246,.3) !important}

/* ── Chat ── */
div[data-testid="stChatMessage"]{border-radius:0;background:transparent;padding:0}
div[data-testid="stChatMessage"] {
  animation: fade-in 0.4s ease-out forwards;
}
@keyframes fade-in {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p{
  color:#334155 !important;font-size:18px !important;line-height:1.8 !important;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code{
  color:#0f1e35 !important;background:#f1f5f9 !important;border-radius:3px;padding:2px 6px;font-size:16px !important;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong{color:#f59e0b !important}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2{color:#1e293b !important;font-size:24px !important}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3{color:#334155 !important;font-size:20px !important}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote{
  border-left:2px solid #cbd5e1;padding-left:10px;color:#64748b !important;
}

/* ── Mode badges ── */
.bdg{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:3px;
  font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;letter-spacing:.5px}
.bdg-lk{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:#2563eb}
.bdg-llm{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:#16a34a}
.bdg-ana{background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.3);color:#9333ea}
.bdg-err{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#dc2626}

/* ── Score ── */
.sc-hi{color:#16a34a;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700}
.sc-md{color:#d97706;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700}
.sc-lo{color:#dc2626;font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700}

/* ── Date separator ── */
.dsep{text-align:center;color:#1e3a5f;font-family:'JetBrains Mono',monospace;
  font-size:14px;letter-spacing:3px;padding:8px 0;margin:18px 0 10px;
  border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0}

/* ── Pending approval banner ── */
.approval-box{
  background:#f8fafc;border:1px solid #e2e8f0;border-left:3px solid #3b82f6;
  border-radius:0 6px 6px 0;padding:12px 16px;margin:6px 0;
}

/* ── Form / inputs ── */
.stTextArea textarea,.stTextInput input{
  background:#ffffff !important;border:1px solid #cbd5e1 !important;
  color:#1e293b !important;border-radius:4px !important;
  font-family:'JetBrains Mono',monospace !important;font-size:18px !important;
}
.stTextArea textarea:focus,.stTextInput input:focus{
  border-color:#3b82f6 !important;box-shadow:0 0 0 2px rgba(59,130,246,.12) !important;
}
.stSelectbox>div>div,.stMultiSelect>div>div{
  background:#ffffff !important;border:1px solid #cbd5e1 !important;color:#1e293b !important;
}
.stRadio label{color:#334155 !important;font-size:18px !important}
.stCheckbox label{color:#334155 !important;font-size:18px !important}
div[data-baseweb="radio"]>div{background:#ffffff !important}

/* ── Expanders ── */
div[data-testid="stExpander"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:5px}
details summary{color:#3b82f6 !important;font-family:'JetBrains Mono',monospace !important;font-size:16px !important}
label, .stMarkdown p, .stMarkdown li{font-size:18px !important}

/* ── Tables ── */
.stDataFrame{border:1px solid #cbd5e1;border-radius:5px}
.stDataFrame thead th{background:#f8fafc !important;color:#1d4ed8 !important;
  font-family:'JetBrains Mono',monospace !important;font-size:16px !important;
  border-bottom:1px solid #cbd5e1 !important}
.stDataFrame tbody tr:hover{background:#f1f5f9 !important}

/* ── File uploader ── */
[data-testid="stFileUploader"]{background:#f8fafc;border:2px dashed #cbd5e1;border-radius:8px;padding:8px}
[data-testid="stFileUploader"]:hover{border-color:#3b82f6}
[data-testid="stFileUploader"] label{color:#475569 !important;font-size:16px !important}

/* ── Select slider ── */
.stSlider label{color:#475569 !important;font-size:16px !important}

/* ── Progress ── */
.stProgress>div>div{background:linear-gradient(90deg,#1d4ed8,#3b82f6) !important}

/* ── Alerts ── */
.stAlert{border-radius:4px !important;font-size:16px !important}
[data-testid="stStatusContainer"]{background:#f8fafc !important;border:1px solid #e2e8f0 !important;border-radius:5px !important}

code{color:#15803d;background:#f1f5f9;padding:1px 5px;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:16px}
hr{border-color:#e2e8f0}
h1,h2,h3,h4{color:#1e293b}
h1{font-size:36px !important}
h2{font-size:30px !important}
h3{font-size:26px !important}
h4{font-size:22px !important}
p,li{color:#334155}
blockquote{border-left:2px solid #cbd5e1;padding-left:10px;color:#64748b}

.sec-hd{font-family:'JetBrains Mono',monospace;font-size:20px;color:#3b82f6;
  letter-spacing:2px;text-transform:uppercase;padding-bottom:8px;
  border-bottom:1px solid #e2e8f0;margin-bottom:14px}
  
.blueprint-box {
  background: #f1f5f9; border-left: 4px solid #1d4ed8; padding: 16px; border-radius: 4px;
  font-size: 18px; line-height: 1.6; color: #334155; margin-bottom: 20px;
}
.blueprint-box h4 { color: #0f172a; margin-top: 0; margin-bottom: 12px; font-size: 22px; }
.blueprint-box ul { margin-bottom: 0; padding-left: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="banner">⬡ SPEARHEAD SENTINEL INTELLIGENCE WIKI ⬡</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:right;margin:-8px 0 10px'>"
    "<a href='#help-center' style='font-family:JetBrains Mono,monospace;"
    "font-size:14px;color:#1d4ed8;text-decoration:none'>❓ HELP PAGE</a>"
    "</div>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init():
    hist_file = WIKI_ROOT / ".meta" / "chat_history.json"
    msgs = []
    if hist_file.exists():
        try:
            msgs = json.loads(hist_file.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    defs: dict[str, Any] = {
        "messages":        msgs,
        "pending":         [],    # answers awaiting approval
        "pipeline":        None,
        "model":           _cli.model or "",
        "ollama_status":   None,
        "avail_models":    [],
        "ingest_log":      [],
        "chat_prefill":    "",
        "chat_start_idx":  0,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
            
def _save_chat_history():
    hist_file = WIKI_ROOT / ".meta" / "chat_history.json"
    hist_file.parent.mkdir(exist_ok=True, parents=True)
    hist_file.write_text(json.dumps(st.session_state["messages"]), encoding="utf-8")

_init()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:16px;"
        "color:#3b82f6;letter-spacing:3px;font-weight:700;margin-bottom:16px'>"
        "⬡ SENTINEL-LAC</div>",
        unsafe_allow_html=True,
    )

    arts  = _wiki()
    rpts  = [a for a in arts if a.get("report_id")]
    secs  = {a.get("sector","") for a in rpts if a.get("sector")}

    c1, c2 = st.columns(2)
    c1.metric("Articles", len(arts))
    c1.metric("Sectors",  len(secs))
    c2.metric("Reports",  len(rpts))
    c2.metric("Pending",  len(st.session_state["pending"]))
    st.caption(
        "`Articles`: wiki pages in knowledge base · "
        "`Reports`: parsed patrol reports · "
        "`Sectors`: unique operational areas covered · "
        "`Pending`: analyst answers waiting approval/commit."
    )

    st.divider()

    # Ollama probe
    if st.session_state["ollama_status"] is None:
        with st.spinner("Probing Ollama…"):
            det, avail = detect_model()
            st.session_state["ollama_status"] = "online" if avail else "offline"
            st.session_state["avail_models"]  = avail
            if not st.session_state["model"] and det:
                st.session_state["model"] = det

    ollama_ok = st.session_state["ollama_status"] == "online"
    avail     = st.session_state.get("avail_models", [])

    if ollama_ok and avail:
        st.markdown(
            "<div style='font-size:14px;color:#22c55e;font-family:JetBrains Mono,"
            "monospace'>✓ Ollama online</div>", unsafe_allow_html=True,
        )
        cur = st.session_state.get("model", avail[0] if avail else "")
        idx = avail.index(cur) if cur in avail else 0
        sel = st.selectbox("Model", avail, index=idx,
                           label_visibility="collapsed")
        if sel != st.session_state.get("model"):
            st.session_state["model"]    = sel
            st.session_state["pipeline"] = None
    else:
        st.markdown(
            "<div style='font-size:14px;color:#f59e0b;font-family:JetBrains Mono,"
            "monospace'>⚠ Ollama offline</div>", unsafe_allow_html=True,
        )
    mi = st.text_input("Model", value=st.session_state.get("model", os.environ.get("OLLAMA_MODEL", "llama3.1:8b")),
                           label_visibility="collapsed")
    st.session_state["model"] = mi

    st.divider()

    if not WIKI_ENGINE_OK:
        st.error(f"Engine: {_import_error}")

    # Quick stats panel
    if rpts:
        max_d = max((float(a.get("depth_m",0) or 0) for a in rpts), default=0)
        st.markdown(
            f"<div style='font-size:14px;font-family:JetBrains Mono,monospace;"
            f"color:#475569;line-height:2'>"
            f"Max depth: <span style='color:#ef4444'>{max_d:.0f} m</span><br>"
            f"TTS total: <span style='color:#f59e0b'>"
            f"{sum(int(a.get('tts_count',0) or 0) for a in rpts)}</span><br>"
            f"With issues: <span style='color:#a855f7'>"
            f"{sum(1 for a in rpts if int(a.get('issues_count',0) or 0) > 0)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("⟳  Refresh All", use_container_width=True,
                 help="Reload wiki data, clear caches, re-probe Ollama"):
        load_wiki.clear()
        build_graph.clear()
        st.session_state["ollama_status"] = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab4, tab1, tab2, tab3, tab5 = st.tabs([
    "💬  OPERATOR CHAT",
    "🗺  TACTICAL MAP",
    "📊  PATTERN ANALYSIS",
    "🕸  LINK ANALYSIS",
    "❓  HELP",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  —  TACTICAL MAP
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="sec-hd">Tactical Incident Map — LAC Sector</div>',
                unsafe_allow_html=True)
    st.info(
        "How to read this map: each marker is a report-linked location (frontmatter coords, "
        "lat/lon mentions, route waypoints, or sector fallback). Click a marker for patrol name, "
        "date, route, strength, depth and issue hints. Red halo markers are auto-flagged high-sensitivity areas."
    )

    map_recs: list[dict[str, Any]] = []
    map_points: list[dict[str, Any]] = []
    for a in rpts:
        pts = report_locations(a)
        if not pts:
            continue
        threat_score = report_threat_score(a)
        map_recs.append({**a, "_pts": pts, "_threat_score": threat_score})
        for p in pts:
            map_points.append({**a, **p, "_threat_score": threat_score})

    c_map, c_stat = st.columns([3, 1])

    with c_map:
        if not HAS_FOLIUM:
            st.warning("Install `folium` and `streamlit-folium` for map.")
        else:
            if not map_points:
                lats, lons = [28.158], [97.017]
                st.info("No explicit coordinates found. Displaying default LAC overview map.")
            else:
                lats = [p["lat"] for p in map_points]
                lons = [p["lon"] for p in map_points]
            fmap = folium.Map(
                location=[sum(lats)/len(lats), sum(lons)/len(lons)],
                zoom_start=8, tiles="CartoDB positron",
            )
            mc = MarkerCluster(
                options={"maxClusterRadius":35,"disableClusteringAtZoom":11}
            ).add_to(fmap)

            for a in map_recs:
                dm      = float(a.get("depth_m",0) or 0)
                dc      = a.get("depth_cat","Shallow")
                clr     = DEPTH_COL.get(dc,"#6b7280")
                radius  = max(6, min(24, int(dm/180)))
                tts_cnt = int(a.get("tts_count",0) or 0)
                iss_cnt = int(a.get("issues_count",0) or 0)
                rid     = a.get("report_id",a["_stem"])
                pdate   = _report_primary_date(a)
                route   = (a.get("route","") or "Not recorded")[:90]
                officer = a.get("signed_by","—")
                issue_line = "Yes" if iss_cnt else "No"

                # Route polyline (if route-derived points are available)
                route_pts = [[p["lat"], p["lon"]] for p in a.get("_pts", []) if p.get("source") == "route"]
                if len(route_pts) >= 2:
                    folium.PolyLine(
                        locations=route_pts,
                        color="#2563eb",
                        weight=2,
                        opacity=0.7,
                        tooltip=f"{rid} route path",
                    ).add_to(fmap)

                for p in a.get("_pts", []):
                    popup = (
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                        f"background:#ffffff;color:#1e293b;padding:12px;min-width:300px;"
                        f"border-left:3px solid {clr}'>"
                        f"<div style='color:{clr};font-weight:700;font-size:13px;margin-bottom:8px'>■ {rid}</div>"
                        f"<table style='width:100%;border-collapse:collapse;line-height:1.9'>"
                        f"<tr><td style='color:#475569;width:90px'>Date</td><td>{pdate}</td></tr>"
                        f"<tr><td style='color:#475569'>Sector</td><td>{a.get('sector','')[:50]}</td></tr>"
                        f"<tr><td style='color:#475569'>Route</td><td>{route}</td></tr>"
                        f"<tr><td style='color:#475569'>Officer</td><td>{officer[:40]}</td></tr>"
                        f"<tr><td style='color:#475569'>Strength</td><td style='color:#1d4ed8;font-weight:700'>{a.get('strength_raw','')[:50]}</td></tr>"
                        f"<tr><td style='color:#475569'>Depth</td><td style='color:{clr};font-weight:700'>{a.get('depth_raw','')[:60]}</td></tr>"
                        f"<tr><td style='color:#475569'>TTS</td><td style='color:#f59e0b'>{tts_cnt} indicator(s)</td></tr>"
                        f"<tr><td style='color:#475569'>Issues</td><td style='color:{'#ef4444' if iss_cnt else '#16a34a'}'>{issue_line}{' (' + str(iss_cnt) + ')' if iss_cnt else ''}</td></tr>"
                        f"<tr><td style='color:#475569'>Point</td><td>{p.get('source','unknown').replace('_',' ')}</td></tr>"
                        f"</table></div>"
                    )
                    folium.CircleMarker(
                        location=[p["lat"], p["lon"]],
                        radius=radius,
                        color=clr,
                        fill=True,
                        fill_color=clr,
                        fill_opacity=0.78,
                        weight=1.5,
                        popup=folium.Popup(popup, max_width=360),
                        tooltip=f"[{rid}] {(a.get('sector','') or '')[:35]} — {dc} · {p.get('source','')}",
                    ).add_to(mc)

                # High-sensitivity auto threat overlay
                if a.get("_threat_score", 0) >= 4.8:
                    p0 = a["_pts"][0]
                    folium.Circle(
                        location=[p0["lat"], p0["lon"]],
                        radius=1800,
                        color="#dc2626",
                        fill=True,
                        fill_opacity=0.10,
                        weight=2,
                        tooltip=f"HIGH SENSITIVITY · {rid} · score {a['_threat_score']:.1f}",
                    ).add_to(fmap)

                # Report label
                p0 = a["_pts"][0]
                folium.Marker(
                    [p0["lat"]+0.04, p0["lon"]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-family:JetBrains Mono,monospace;'
                             f'font-size:9px;color:#334155;letter-spacing:1px">'
                             f'{rid}</div>',
                        icon_size=(30, 14),
                    ),
                ).add_to(fmap)

            folium.LayerControl().add_to(fmap)
            st_folium(fmap, height=490, use_container_width=True)
            st.caption(
                "🔴 Deep (≥1.5 km) · 🟡 Moderate (600 m–1.5 km) · "
                "🟢 Shallow (<600 m) · Red halo: high sensitivity by auto threat score · "
                "Size ∝ depth · Click marker for detail"
            )

    with c_stat:
        if map_recs and HAS_PLOTLY:
            depth_rows = sorted(
                [{"rid": a.get("report_id",a["_stem"]),
                  "dm":  float(a.get("depth_m",0) or 0),
                  "dc":  a.get("depth_cat","Shallow")}
                 for a in map_recs if float(a.get("depth_m",0) or 0) > 0],
                key=lambda x: -x["dm"],
            )
            if depth_rows:
                fig = go.Figure(go.Bar(
                    x=[r["rid"] for r in depth_rows],
                    y=[r["dm"]  for r in depth_rows],
                    marker_color=[DEPTH_COL[r["dc"]] for r in depth_rows],
                    text=[f"{r['dm']:.0f}" for r in depth_rows],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono", color="#94a3b8"),
                ))
                fig.update_layout(
                    title=dict(text="Depth (m)", font=dict(size=10,
                               color="#3b82f6", family="JetBrains Mono")),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(color="#334155"), height=210,
                    margin=dict(l=0,r=0,t=26,b=0), showlegend=False,
                    xaxis=dict(tickfont=dict(size=9, family="JetBrains Mono"),
                               gridcolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#e2e8f0",
                               title=dict(text="m", font=dict(size=8, color="#334155"))),
                )
                st.plotly_chart(fig, use_container_width=True)

        if rpts and HAS_PANDAS:
            tbl = [{"Report": a.get("report_id",""),
                    "Date":    _report_primary_date(a),
                    "Sector":  (a.get("sector","") or "")[:28],
                    "Depth":   a.get("depth_cat","—"),
                    "TTS":     int(a.get("tts_count",0) or 0),
                    "Issues":  int(a.get("issues_count",0) or 0),
                    "Threat":  f"{report_threat_score(a):.1f}"}
                   for a in sorted(rpts,
                       key=lambda x: float(x.get("depth_m",0) or 0),
                       reverse=True)]
            st.dataframe(pd.DataFrame(tbl), use_container_width=True,
                         hide_index=True, height=250)

    st.divider()
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
        "color:#3b82f6;letter-spacing:2px;margin-bottom:12px'>"
        "⬡ AUTOMATED INTELLIGENCE BRIEF</div>",
        unsafe_allow_html=True,
    )
    st.markdown(build_auto_brief(rpts, secs), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  —  PATTERN ANALYSIS  (feeds from Link Graph)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="sec-hd">Pattern Analysis — TTS · Depth · Personnel · Sectors</div>',
                unsafe_allow_html=True)
    st.info(
        "What to derive here: this tab turns raw patrol text into trend signals. "
        "Use TTS frequency to see repeating indicators, depth-over-time for escalation, "
        "heatmap for sector-vs-pattern concentration, and personnel table for leadership recurrence."
    )

    if not rpts:
        st.info("Ingest patrol reports to see pattern analysis.")
    elif not HAS_PLOTLY:
        st.warning("Install plotly.")
    else:
        graph_data = build_graph(str(WIKI_ROOT))
        pat        = graph_data.get("pattern", {})
        off_map    = pat.get("officer_reports", {})
        tts_map    = pat.get("tts_reports",  {})
        sec_map    = pat.get("sector_reports",{})

        # Pattern search panel
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#3b82f6;letter-spacing:1.5px;margin:6px 0 6px'>SEARCH PATTERN ACROSS REPORT DATABASE</div>",
            unsafe_allow_html=True,
        )
        pq1, pq2 = st.columns([3, 1])
        with pq1:
            p_query = st.text_input(
                "Pattern query",
                value="",
                placeholder="e.g. drone + night patrol, boot prints near walong, repeated camouflage",
                label_visibility="collapsed",
                key="pattern_query_input",
            )
        with pq2:
            run_p_search = st.button("Search Pattern", use_container_width=True, key="run_pattern_search")
        if run_p_search and p_query.strip():
            p_hits = pattern_search(rpts, p_query)
            if p_hits and HAS_PANDAS:
                st.success(f"{len(p_hits)} matching report(s) for pattern: {p_query}")
                st.dataframe(pd.DataFrame(p_hits), use_container_width=True, hide_index=True, height=220)
            else:
                st.warning("No pattern matches found. Try broader terms like sector/unit/TTS keywords.")

        # ── Row 1: TTS bar + Depth scatter ──────────────────────────────────
        r1a, r1b = st.columns(2)

        with r1a:
            st.caption("Interpretation: taller bars = frequently repeated signs across reports; persistent high bars usually indicate stable PLA operating habits.")
            if tts_map:
                cats_s = sorted(tts_map.items(), key=lambda x: -len(x[1]))
                fig_t  = go.Figure(go.Bar(
                    x=[c[0] for c in cats_s],
                    y=[len(c[1]) for c in cats_s],
                    marker_color=[TTS_COL.get(c[0],"#6b7280") for c in cats_s],
                    text=[len(c[1]) for c in cats_s],
                    textposition="outside",
                    textfont=dict(size=10, family="JetBrains Mono", color="#334155"),
                    customdata=[", ".join(c[1]) for c in cats_s],
                    hovertemplate="<b>%{x}</b><br>Reports: %{y}<br>%{customdata}<extra></extra>",
                ))
                fig_t.update_layout(
                    title=dict(text="TTS Category Frequency (from Link Graph)",
                               font=dict(size=10,color="#3b82f6",family="JetBrains Mono")),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(color="#334155"), height=280,
                    margin=dict(l=0,r=0,t=30,b=70), showlegend=False,
                    xaxis=dict(tickfont=dict(size=9,family="JetBrains Mono"),
                               tickangle=35, gridcolor="#e2e8f0"),
                    yaxis=dict(gridcolor="#e2e8f0"),
                )
                st.plotly_chart(fig_t, use_container_width=True)

        with r1b:
            st.caption("Interpretation: upward drift or repeated deep points across dates indicates increasing operational confidence or probing depth.")
            dep_pts = [(a.get("report_id",""),
                        float(a.get("depth_m",0) or 0),
                        (a.get("patrol_dates") or [""])[0][:7],
                        a.get("depth_cat","Shallow"))
                       for a in rpts if float(a.get("depth_m",0) or 0) > 0]
            if dep_pts:
                fig_d = go.Figure()
                for dc, clr in DEPTH_COL.items():
                    pts = [p for p in dep_pts if p[3] == dc]
                    if pts:
                        fig_d.add_trace(go.Scatter(
                            x=[p[2] for p in pts], y=[p[1] for p in pts],
                            mode="markers+text", name=dc,
                            text=[p[0] for p in pts],
                            textposition="top center",
                            textfont=dict(size=9, family="JetBrains Mono"),
                            marker=dict(size=16, color=clr, opacity=0.88,
                                        line=dict(width=2, color=clr)),
                        ))
                fig_d.update_layout(
                    title=dict(text="Ingression Depth Over Time",
                               font=dict(size=10,color="#3b82f6",family="JetBrains Mono")),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(color="#334155"), height=280,
                    margin=dict(l=0,r=0,t=30,b=70),
                    legend=dict(font=dict(size=9,family="JetBrains Mono"),
                                bgcolor="#ffffff",bordercolor="#e2e8f0"),
                    xaxis=dict(gridcolor="#e2e8f0",
                               tickfont=dict(size=9,family="JetBrains Mono")),
                    yaxis=dict(gridcolor="#e2e8f0",
                               title=dict(text="metres", font=dict(size=8, color="#334155"))),
                )
                st.plotly_chart(fig_d, use_container_width=True)

        # ── Row 2: Sector×TTS Heatmap + Personnel tracking ──────────────────
        r2a, r2b = st.columns([1.4, 1])

        with r2a:
            st.caption("Interpretation: hot cells identify where a specific TTS category repeatedly appears in a specific sector.")
            if tts_map and sec_map:
                all_cats  = list(TTS_CATS.keys()) + ["Other"]
                all_secs  = list(sec_map.keys())
                # Build matrix
                all_arts_body: dict[str, str] = {
                    a.get("report_id",""): a.get("_body","") for a in rpts
                }
                z = []
                for sec in all_secs:
                    sec_label = sec.replace("SEC::","")
                    row = []
                    sec_reps = sec_map.get(sec, [])
                    for cat in all_cats:
                        cat_reps = tts_map.get(cat, [])
                        overlap  = len(set(sec_reps) & set(cat_reps))
                        row.append(overlap)
                    z.append(row)
                if any(any(r) for r in z):
                    fig_h = go.Figure(go.Heatmap(
                        z=z,
                        x=all_cats,
                        y=[s.replace("SEC::","")[:30] for s in all_secs],
                        colorscale=[[0,"#f8fafc"],[0.25,"#e2e8f0"],
                                    [0.6,"#f59e0b"],[1,"#ef4444"]],
                        text=[[str(v) if v else "" for v in row] for row in z],
                        texttemplate="%{text}",
                        hovertemplate="Sector: %{y}<br>TTS: %{x}<br>Reports: %{z}<extra></extra>",
                    ))
                    fig_h.update_layout(
                        title=dict(text="Sector × TTS Heatmap (Link Graph Data)",
                                   font=dict(size=10,color="#3b82f6",family="JetBrains Mono")),
                        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                        font=dict(color="#334155"), height=300,
                        margin=dict(l=0,r=0,t=30,b=70),
                        xaxis=dict(tickfont=dict(size=8,family="JetBrains Mono"),
                                   tickangle=40),
                        yaxis=dict(tickfont=dict(size=9,family="JetBrains Mono")),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

        with r2b:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "color:#3b82f6;letter-spacing:1.5px;margin-bottom:10px'>"
                "PERSONNEL TRACKING</div>",
                unsafe_allow_html=True,
            )
            if off_map:
                st.caption("Interpretation: higher count means the same officer appears in more patrol reports (leadership recurrence).")
                pers_rows = []
                for ofr, rep_list in sorted(off_map.items(), key=lambda x: -len(x[1])):
                    a_data = next((a for a in rpts if
                                   (a.get("officer_name","") or "").strip(". ") == ofr), {})
                    pers_rows.append({
                        "Officer":   ofr[:28],
                        "Rank":      a_data.get("officer_rank","—"),
                        "Reports":   ", ".join(sorted(rep_list)),
                        "Count":     len(rep_list),
                    })
                if pers_rows and HAS_PANDAS:
                    st.dataframe(
                        pd.DataFrame(pers_rows).sort_values("Count", ascending=False),
                        use_container_width=True, hide_index=True, height=240,
                    )
            else:
                st.caption("No officer tracking data yet.")

        # ── Row 3: Strength summary + Wins analysis ──────────────────────────
        r3a, r3b = st.columns(2)

        with r3a:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "color:#3b82f6;letter-spacing:1.5px;margin-bottom:8px'>"
                "PATROL STRENGTH BY REPORT</div>",
                unsafe_allow_html=True,
            )
            if rpts and HAS_PANDAS:
                st.dataframe(
                    pd.DataFrame([{
                        "Report":   a.get("report_id",""),
                        "Sector":   (a.get("sector","") or "")[:28],
                        "Strength": a.get("strength_raw","—")[:45],
                        "Total":    a.get("strength_total","—"),
                        "Type":     (a.get("patrol_type","") or "")[:30],
                        "Date":     (a.get("patrol_dates") or ["—"])[0][:10],
                    } for a in sorted(rpts,
                                      key=lambda x: (x.get("patrol_dates") or [""])[0])]),
                    use_container_width=True, hide_index=True, height=220,
                )

        with r3b:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "color:#3b82f6;letter-spacing:1.5px;margin-bottom:8px'>"
                "PLA CAPABILITIES IDENTIFIED</div>",
                unsafe_allow_html=True,
            )
            wins_all: list[dict] = []
            for a in rpts:
                body  = a.get("_body","")
                m_win = re.search(r'## PLA Capabilities.*?\n((?:- .+\n?)+)', body, re.S)
                if m_win:
                    for w in re.findall(r'- (.+)', m_win.group(1)):
                        w = w.strip("* ")
                        if w and len(w) > 8:
                            wins_all.append({"Report": a.get("report_id",""),
                                             "Capability": w[:80]})
            if wins_all and HAS_PANDAS:
                st.dataframe(
                    pd.DataFrame(wins_all),
                    use_container_width=True, hide_index=True, height=220,
                )
            else:
                st.caption("No PLA capability assessments yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  —  LINK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="sec-hd">Link Analysis — Entity Network</div>',
                unsafe_allow_html=True)
    st.info(
        "How to read this network: nodes are entities (reports, officers, units, sectors, TTS, issues). "
        "Larger nodes have more links (higher centrality). Use `Focus on` + `Hops` to isolate one entity context."
    )

    if not HAS_NX or not HAS_PLOTLY:
        st.warning("Install `networkx` and `plotly`.")
    elif not arts:
        st.info("Ingest reports to build the link graph.")
    else:
        gd = build_graph(str(WIKI_ROOT))

        # Controls
        cc1, cc2, cc3 = st.columns([1.5, 1.5, 1])
        with cc1:
            ntypes_all = sorted({n["ntype"] for n in gd["nodes"]})
            sel_types  = st.multiselect(
                "Show node types",
                ntypes_all, default=ntypes_all,
                help="Filter which node types are visible",
            )
        with cc2:
            focus_opts = ["All"] + sorted({n["label"] for n in gd["nodes"] if n.get("label")})
            focus = st.selectbox(
                "Focus on",
                focus_opts,
                help=(
                    "Select an officer, unit or sector to centre the graph on it. "
                    "All connected reports and TTS will be highlighted. "
                    "Example: select 'Maj. Karan Thapa' to see all S5 patrol connections."
                ),
            )
        with cc3:
            depth_hop = st.slider("Hops", 1, 3, 1,
                                  help="Neighbour depth around focus node")

        cg, cl = st.columns([4, 1])

        with cg:
            if not gd["nodes"]:
                st.info("No linked articles yet.")
            else:
                # Filter + focus
                vis_nodes = [n for n in gd["nodes"] if n["ntype"] in sel_types]
                vis_ids   = {n["id"] for n in vis_nodes}

                if focus != "All":
                    focus_id = next(
                        (n["id"] for n in gd["nodes"] if n["label"] == focus), None
                    )
                    if focus_id and HAS_NX:
                        # Get N-hop neighbours
                        G_tmp = nx.Graph()
                        for e in gd["edges"]:
                            # Find node IDs from positions
                            n0 = next((n["id"] for n in gd["nodes"]
                                       if abs(n["x"]-e["x0"])<1e-9 and
                                          abs(n["y"]-e["y0"])<1e-9), None)
                            n1 = next((n["id"] for n in gd["nodes"]
                                       if abs(n["x"]-e["x1"])<1e-9 and
                                          abs(n["y"]-e["y1"])<1e-9), None)
                            if n0 and n1:
                                G_tmp.add_edge(n0, n1)
                        nbrs = nx.ego_graph(G_tmp, focus_id, radius=depth_hop).nodes()
                        vis_ids = vis_ids & set(nbrs)
                        vis_nodes = [n for n in vis_nodes if n["id"] in vis_ids]

                vis_edges = [
                    e for e in gd["edges"]
                    if any(abs(n["x"]-e["x0"])<1e-9 and abs(n["y"]-e["y0"])<1e-9
                           for n in vis_nodes)
                    and any(abs(n["x"]-e["x1"])<1e-9 and abs(n["y"]-e["y1"])<1e-9
                            for n in vis_nodes)
                ]

                edge_tr = [go.Scatter(
                    x=[e["x0"],e["x1"],None], y=[e["y0"],e["y1"],None],
                    mode="lines", hoverinfo="none", showlegend=False,
                    line=dict(width=0.8, color="#cbd5e1"),
                ) for e in vis_edges]

                TYPE_NAMES = {
                    "report":"Patrol Reports","officer":"Officers",
                    "unit":"Army Units","sector":"Sectors",
                    "tts":"TTS Categories","issue":"Issues Raised",
                }
                TYPE_SIZE_BASE = {
                    "report":18,"officer":16,"unit":14,
                    "sector":14,"tts":12,"issue":10,
                }
                by_t: dict[str, list[dict]] = {}
                for n in vis_nodes:
                    by_t.setdefault(n["ntype"], []).append(n)

                node_tr = [
                    go.Scatter(
                        x=[n["x"] for n in ns],
                        y=[n["y"] for n in ns],
                        mode="markers+text",
                        name=TYPE_NAMES.get(t, t),
                        marker=dict(
                            size=[TYPE_SIZE_BASE.get(t,12) + n["deg"]*2
                                  for n in ns],
                            color=NODE_COL.get(t,"#6b7280"),
                            opacity=0.88,
                            line=dict(width=1.2, color="#ffffff"),
                        ),
                        text=[n["label"] for n in ns],
                        textposition="top center",
                        textfont=dict(size=9, color="#334155",
                                      family="JetBrains Mono"),
                        customdata=[
                            f"Type: {t}<br>"
                            f"Connections: {n['deg']}<br>"
                            + "<br>".join(
                                f"{k}: {v}"
                                for k, v in (n.get("meta") or {}).items()
                                if v and k not in ("pos",)
                            )
                            for n in ns
                        ],
                        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
                    )
                    for t, ns in by_t.items()
                ]

                fig_g = go.Figure(data=edge_tr + node_tr)
                fig_g.update_layout(
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    font=dict(color="#334155"),
                    showlegend=True,
                    legend=dict(font=dict(size=9, family="JetBrains Mono"),
                                bgcolor="#ffffff", bordercolor="#e2e8f0"),
                    margin=dict(l=0,r=0,t=10,b=0), height=560,
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                )
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption(
                    "🔵 Reports · 🟡 Officers · 🔴 Units · 🟢 Sectors · "
                    "🟣 TTS Categories · 🟤 Issues — "
                    "Node size ∝ connections · Use Focus to inspect one entity"
                )

        with cl:
            st.metric("Nodes", gd["n_nodes"])
            st.metric("Links", gd["n_edges"])
            st.caption(
                "Tip: high-degree hubs usually represent recurring reports, sectors, or TTS categories "
                "that deserve deeper investigation."
            )
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                "color:#334155;letter-spacing:1px;margin:10px 0 6px'>"
                "TOP HUBS</div>",
                unsafe_allow_html=True,
            )
            hub_lookup = {n["id"]: n for n in gd["nodes"]}
            for nid, deg in gd["hubs"][:12]:
                nd  = hub_lookup.get(nid, {})
                clr = NODE_COL.get(nd.get("ntype","other"), "#6b7280")
                lbl = nd.get("label", nid)
                st.markdown(
                    f"<div style='font-size:10px;font-family:JetBrains Mono,"
                    f"monospace;margin-bottom:3px'>"
                    f"<span style='color:{clr}'>■</span> "
                    f"<span style='color:#334155'>{lbl[:22]}</span> "
                    f"<span style='color:#1e3a5f'>({deg})</span></div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  —  OPERATOR CHAT + INGEST
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.info(
        "Operator workflow: choose lookup or LLM analysis, ask/approve intelligence responses, "
        "and ingest new reports via the left pane. Left pane keeps quick context and prior chat prompts."
    )
    col_left, col_chat = st.columns([1.5, 4.5])

    with col_left:
        if st.button("➕ Start New Chat", use_container_width=True):
            st.session_state["chat_start_idx"] = len(st.session_state["messages"])
            st.session_state["chat_prefill"] = ""
            st.rerun()

        with st.expander("⬡ INTELLIGENCE INGEST", expanded=False):
            uploaded_files = st.file_uploader("Upload patrol reports (.txt, .docx)", accept_multiple_files=True)
            force_ingest = st.checkbox("Force Re-Ingest (Overwrite duplicates)", value=True, help="Check this to update reports if parser logic has changed.")
            if st.button("📥 Ingest All", use_container_width=True) and uploaded_files:
                with st.spinner("Ingesting reports..."):
                    RAW_ROOT.mkdir(parents=True, exist_ok=True)
                    docs_written = 0
                    for uf in uploaded_files:
                        dest = RAW_ROOT / uf.name
                        dest.write_bytes(uf.read())
                        res = direct_ingest(dest, force=force_ingest)
                        if res.get("error") and "DOCX_ERROR" in res.get("error", ""):
                            st.error(f"{uf.name} failed: {res['error']}")
                        docs_written += res.get("written", 0)
                    if docs_written > 0:
                        st.success(f"Ingested {docs_written} articles.")
                        load_wiki.clear()
                        build_graph.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("No valid reports extracted.")
            if WIKI_ENGINE_OK and ollama_ok:
                if st.button("🔄 Rebuild LLM Indexes", use_container_width=True):
                    with st.spinner("Rebuilding semantic & graph indexes..."):
                        pl = get_pipeline(st.session_state.get("model", ""))
                        if pl:
                            run_async(pl.embedder.build())
                            st.success("Semantic index rebuilt.")

        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#3b82f6;letter-spacing:1.5px;margin-bottom:8px;margin-top:12px'>LEFT PANE GUIDE</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "- `Sectors`: count of unique operational areas from ingested patrol reports\n"
            "- `Articles`: all wiki knowledge pages (reports + derived analysis notes)\n"
            "- `Reports`: patrol report entries with structured fields\n"
            "- `Pending`: answers waiting analyst approval before committing to wiki"
        )
        st.divider()
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#3b82f6;letter-spacing:1.5px;margin-bottom:8px'>PREVIOUS CHATS</div>",
            unsafe_allow_html=True,
        )
        all_msgs = st.session_state.get("messages", [])
        chat_pairs = []

        # Build conversation pairs (user query + assistant response)
        i = 0
        while i < len(all_msgs):
            m = all_msgs[i]
            if m.get("role") == "user":
                u_msg = m
                u_idx = i
                a_msg = None
                if i + 1 < len(all_msgs) and all_msgs[i+1].get("role") == "assistant":
                    a_msg = all_msgs[i+1]
                    i += 2
                else:
                    i += 1
                chat_pairs.append((u_idx, u_msg, a_msg))
            else:
                i += 1

        if not chat_pairs:
            st.caption("No previous chat prompts yet.")
        else:
            for idx, u_msg, a_msg in chat_pairs[-8:][::-1]:
                qtxt = str(u_msg.get("content", "")).strip().replace("\n", " ")
                q_lbl = (qtxt[:32] + "…") if len(qtxt) > 32 else qtxt
                
                if a_msg:
                    atxt = str(a_msg.get("content", "")).strip().replace("\n", " ")
                    a_lbl = (atxt[:28] + "…") if len(atxt) > 28 else atxt
                    mode = a_msg.get("mode", "")
                    
                    if "LOOKUP" in mode: mode_pfx = "⚡"
                    elif "ANALYSIS" in mode: mode_pfx = "📊"
                    elif "LLM" in mode: mode_pfx = "🧠"
                    else: mode_pfx = "💬"
                    
                    lbl = f"{mode_pfx} Q: {q_lbl} | A: {a_lbl}"
                    tooltip = f"Q: {qtxt}\n\nMode: {mode}\nA: {atxt[:200]}..."
                else:
                    lbl = f"Q: {q_lbl} | (No response)"
                    tooltip = f"Q: {qtxt}"
                    
                c_load, c_del = st.columns([5, 1])
                with c_load:
                    if st.button(lbl or "(empty)", key=f"hist_q_{idx}", use_container_width=True, help=tooltip):
                        st.session_state["chat_start_idx"] = idx
                        st.session_state["chat_prefill"] = ""
                        st.rerun()
                with c_del:
                    if st.button("✖", key=f"del_q_{idx}", use_container_width=True, help="Delete chat"):
                        if a_msg:
                            del st.session_state["messages"][idx:idx+2]
                            dc = 2
                        else:
                            del st.session_state["messages"][idx:idx+1]
                            dc = 1
                        _save_chat_history()
                        if st.session_state.get("chat_start_idx", 0) >= idx + dc:
                            st.session_state["chat_start_idx"] -= dc
                        elif st.session_state.get("chat_start_idx", 0) >= idx:
                            st.session_state["chat_start_idx"] = len(st.session_state["messages"])
                        st.rerun()

    # ── CHAT ──────────────────────────────────────────────────────────────────

    with col_chat:
        model_name = st.session_state.get("model","—")

        # Mode selector
        chat_mode = st.radio(
            "mode",
            [
                "⚡  Wiki Lookup  — instant exact match",
                f"🧠  Ollama Analysis  — {model_name}",
            ],
            horizontal=True,
            label_visibility="collapsed",
            help=(
                "Wiki Lookup: instant regex scan, no LLM, verbatim facts.\n"
                "Ollama Analysis: multi-report synthesis via local LLM."
            ),
        )
        use_llm = "Ollama" in chat_mode

        if use_llm:
            if ollama_ok:
                st.markdown(
                    f"<span class='bdg bdg-llm'>🧠 LLM · {model_name}</span>"
                    "<span style='font-size:10px;color:#334155;font-family:monospace;"
                    "margin-left:8px'>analyse + cross-reference + synthesise</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='bdg bdg-err'>⚠ Ollama offline — "
                    "falling back to Wiki Lookup</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span class='bdg bdg-lk'>⚡ WIKI LOOKUP · instant</span>"
                "<span style='font-size:10px;color:#334155;font-family:monospace;"
                "margin-left:8px'>verbatim extraction · no LLM</span>",
                unsafe_allow_html=True,
            )

        # ── Sample queries ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:9px;color:#1a2a45;font-family:JetBrains Mono,"
            "monospace;letter-spacing:2px;text-transform:uppercase;"
            "margin:10px 0 5px'>Sample Queries</div>",
            unsafe_allow_html=True,
        )
        sq1, sq2 = st.columns(2)
        _samples = [
        ("1. Fact Extract",  "What is the serial number of the equipment found in the Walong Sector?"),
        ("2. Multi-Hop",     "Based on the reports, is there a specific PLA unit responsible for the ingression in the Walong sector?"),
        ("3. Pattern",       "Are there any similarities in movement patterns or equipment by the PLA across Kibithoo and Chaglagam?"),
        ("4. Tac-Alert",     "Synthesize these findings into a 'Tactical Alert' for HQ and save it to the wiki."),
            ("Threat brief",   "Analyse PLA activity patterns and assess threat level"),
            ("Deep sectors",   "Which sector had the deepest PLA ingression?"),
        ("Drone reports",  "Which patrols reported drone activity?"),
        ("Maj Thapa",      "Show all reports involving Maj Karan Thapa"),
        ]
        _trig: str | None = None
        for i, (lbl, q) in enumerate(_samples):
            with (sq1 if i % 2 == 0 else sq2):
                if st.button(
                    lbl, key=f"sq{i}", use_container_width=True,
                    help=q,
                ):
                    _trig = q

        st.divider()

        # ── Pending approvals ──────────────────────────────────────────────────
        pending = st.session_state["pending"]
        if pending:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "color:#3b82f6;letter-spacing:1.5px;margin-bottom:8px'>"
                f"⬡ {len(pending)} ANSWER(S) PENDING APPROVAL</div>",
                unsafe_allow_html=True,
            )
            for pi, pend in enumerate(pending):
                with st.container():
                    st.markdown(
                        f"<div class='approval-box'>"
                        f"<div style='font-size:10px;color:#1d4ed8;font-family:JetBrains Mono,"
                        f"monospace;margin-bottom:6px'>Q: {pend['question']}</div>"
                        f"<div style='font-size:12px;color:#334155;max-height:120px;"
                        f"overflow-y:auto'>{pend['answer'][:400]}{'…' if len(pend['answer'])>400 else ''}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    pa1, pa2 = st.columns(2)
                    with pa1:
                        if st.button(
                            "✓ Approve & Commit to Wiki",
                            key=f"app{pi}",
                            use_container_width=True,
                            help="Approve this answer and write it to the wiki knowledge base",
                        ):
                            _commit_to_wiki(pend)
                            st.session_state["pending"].pop(pi)
                            st.success("Committed to wiki.")
                            load_wiki.clear()
                            st.rerun()
                    with pa2:
                        if st.button(
                            "✗ Discard",
                            key=f"dis{pi}",
                            use_container_width=True,
                            help="Discard this answer",
                        ):
                            st.session_state["pending"].pop(pi)
                            st.rerun()
            st.divider()

        # ── Conversation history ───────────────────────────────────────────────
        msgs    = st.session_state["messages"][st.session_state.get("chat_start_idx", 0):]
        prev_d  = None

        for msg in msgs:
            ts    = msg.get("ts","")
            d_str = ts[:10] if ts else ""
            if d_str and d_str != prev_d:
                st.markdown(
                    f"<div class='dsep'>── {d_str} ──</div>",
                    unsafe_allow_html=True,
                )
                prev_d = d_str

            if msg["role"] == "user":
                with st.chat_message("user", avatar="🎖"):
                    st.markdown(msg["content"])
                    if ts: st.caption(ts[11:16])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    mode_s = msg.get("mode","")
                    score  = msg.get("score",0.0)
                    crits  = msg.get("crits",[])
                    srcs   = msg.get("meta",{}).get("sources",[])

                    # Badge
                    if   "LOOKUP"   in mode_s: badge = f"<span class='bdg bdg-lk'>⚡ {mode_s}</span>"
                    elif "ANALYSIS" in mode_s: badge = f"<span class='bdg bdg-ana'>📊 {mode_s}</span>"
                    elif "LLM"      in mode_s: badge = f"<span class='bdg bdg-llm'>🧠 {mode_s}</span>"
                    else:                      badge = f"<span class='bdg bdg-err'>{mode_s}</span>"

                    sc_html = (f"<span class='sc-hi'>▲ {score:.0%}</span>" if score >= 0.7 else
                               f"<span class='sc-md'>◆ {score:.0%}</span>" if score >= 0.45 else
                               f"<span class='sc-lo'>▼ {score:.0%}</span>")

                    st.markdown(f"{badge} &nbsp; {sc_html}", unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    if ts: st.caption(ts[11:16])

                    if crits:
                        with st.expander(f"⚠ {len(crits)} critique(s)", expanded=False):
                            for c in crits:
                                st.markdown(
                                    f"<div style='font-size:11px;color:#fbbf24;"
                                    f"font-family:monospace'>• {c}</div>",
                                    unsafe_allow_html=True,
                                )
                    if srcs:
                        with st.expander(f"📄 {len(srcs)} source(s)", expanded=False):
                            for sp in srcs[:6]:
                                fp = WIKI_ROOT / sp if not sp.startswith("/") else Path(sp)
                                if fp.exists():
                                    fm_s, _ = _fm(
                                        fp.read_text(encoding="utf-8", errors="ignore"))
                                    st.markdown(
                                        f"`{Path(sp).stem}` — "
                                        f"{fm_s.get('report_id','')} "
                                        f"{(fm_s.get('sector','') or '')[:28]}"
                                    )
                                else:
                                    st.markdown(f"`{sp}`")

        if not msgs:
            st.markdown(
                "<div style='text-align:center;color:#0f1e35;font-family:"
                "JetBrains Mono,monospace;font-size:12px;margin:60px 0 40px'>"
                "No queries yet — ask anything about the patrol reports.</div>",
                unsafe_allow_html=True,
            )

        # ── Hops (LLM only) ───────────────────────────────────────────────────
        if use_llm and ollama_ok:
            hops = st.select_slider("Hops", [1,2,3,4,5], value=3,
                                    help="Graph traversal depth for LLM retrieval")
        else:
            hops = 2

        # ── Input form ─────────────────────────────────────────────────────────
        with st.form("chat_form", clear_on_submit=True):
            prefill_q = st.session_state.get("chat_prefill", "")
            q_in = st.text_area(
                "Q", height=78, label_visibility="collapsed",
                value=prefill_q,
                placeholder=(
                    "Type a question…\n"
                    "  Factual: 'What was the strength of K1 patrol?'\n"
                    "  Analysis: 'Analyse PLA patterns and assess threat level'"
                ),
            )
            fb1, fb2, fb3 = st.columns([3, 2, 1])
            with fb1:
                sub_btn = st.form_submit_button(
                    "⟶  SEND",
                    use_container_width=True,
                    help="Submit query. Short factual questions → instant answer. Analysis questions → full report.",
                )
            with fb2:
                exp_btn = st.form_submit_button(
                    "⬇  Export",
                    use_container_width=True,
                    help="Export full chat history as Markdown",
                )
            with fb3:
                clr_btn = st.form_submit_button(
                    "✕",
                    use_container_width=True,
                    help="Clear chat history",
                )

        if clr_btn:
            st.session_state["chat_start_idx"] = len(st.session_state["messages"])
            st.session_state["chat_prefill"] = ""
            st.rerun()

        if exp_btn and msgs:
            lines = [f"# SENTINEL-LAC Chat Export — "
                     f"{datetime.now().strftime('%Y-%m-%d %H:%M')}Z\n"]
            for m in msgs:
                lines.append(f"\n## [{m.get('ts','')[:16]}] {m['role'].upper()}")
                lines.append(m["content"])
                if m["role"] == "assistant":
                    lines.append(f"_Score: {m.get('score',0):.0%}_")
                    for c in m.get("crits",[]):
                        lines.append(f"- Critique: {c}")
            st.download_button(
                "⬇ Download .md",
                data="\n".join(lines),
                file_name=f"sentinel_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )

        # ── Process question ───────────────────────────────────────────────────
        question = _trig or (q_in.strip() if sub_btn and q_in.strip() else None)

        if question:
            st.session_state["chat_prefill"] = ""
            ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            st.session_state["messages"].append({
                "role": "user", "content": question, "ts": ts_now,
            })

            result: dict[str, Any] = {"sources":[], "mode":""}

            if use_llm and ollama_ok and WIKI_ENGINE_OK:
                pl = get_pipeline(model_name)
                if pl is None:
                    result = wiki_lookup(question) if not is_analysis_query(question) \
                             else _analyse_query(question)
                    result["mode"] = "ERROR → LOCAL FALLBACK"
                else:
                    import wiki_engine.pipeline as _pm
                    _pm.WIKI_ROOT = WIKI_ROOT
                    _pm.RAW_ROOT  = RAW_ROOT
                    for attr in ("backlinks","linter","indexer","embedder"):
                        getattr(pl, attr).wiki_root = WIKI_ROOT
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        try:
                            # Determine context upfront so sources are available
                            st.markdown("`⬡ Generating Real-Time Intelligence Synthesis...`")
                            
                            # Execute stream
                            if hasattr(pl, 'query_stream'):
                                ans = st.write_stream(stream_llm_queue(pl, question, hops))
                                srcs = []
                            else:
                                # Fallback if query_stream isn't implemented in compiler.py
                                res = run_async(pl.query(question, hops=hops, intel_mode=True), timeout=3600)
                                ans = res.get("answer", "No answer.")
                                srcs = res.get("articles_consulted", [])
                                st.write(ans)
                                
                            result = {
                                "answer":  ans,
                                "sources": srcs,
                                "mode":    f"LLM ANALYSIS ({model_name})",
                            }
                        except Exception as exc:
                            fb = (_analyse_query(question)
                                  if is_analysis_query(question)
                                  else wiki_lookup(question))
                            result = fb
                            result["mode"] = f"LLM FAILED → LOCAL ({exc!s:.40})"
            else:
                # Local analysis
                if is_analysis_query(question):
                    result = _analyse_query(question)
                    result["mode"] = "ANALYSIS REPORT"
                else:
                    result = wiki_lookup(question)
                    result["mode"] = "WIKI LOOKUP"

            ans     = result.get("answer","")
            srcs    = result.get("sources", result.get("articles_consulted",[]))
            sc, crs = evaluate(question, ans, srcs)

            st.session_state["messages"].append({
                "role":  "assistant",
                "content": ans,
                "ts":    ts_now,
                "mode":  result.get("mode",""),
                "meta":  {"sources": srcs},
                "score": sc,
                "crits": crs,
            })
            _save_chat_history()

            # Add to pending for approval (non-empty answers only)
            if ans and "EMPTY" not in result.get("mode","") and "NO MATCH" not in result.get("mode",""):
                st.session_state["pending"].append({
                    "question": question,
                    "answer":   ans,
                    "sources":  srcs,
                    "ts":       ts_now,
                })

            st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  —  HELP PAGE
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("<a id='help-center'></a>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hd">Help Page — Sentinel-LAC Usage Guide</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="blueprint-box">
            <h4>🏗️ System Blueprint & Intelligence Utility</h4>
            Sentinel-LAC is designed as a continuously evolving tactical knowledge graph. 
            Here is how it turns unstructured frontline reports into structured intelligence:
            <ul>
                <li><strong>Phase 1: Ingestion & Normalisation.</strong> Raw `.docx` and `.txt` files are parsed via regular expressions (Direct) or processed via local LLMs to extract exact entities, MGRS grids, DTG, and Tell-Tale Signs (TTS).</li>
                <li><strong>Phase 2: Entity Graphing.</strong> Extracted elements (Officers, Units, Sectors, TTS, Issues) are resolved and injected into a local <code>networkx</code> graph.</li>
                <li><strong>Phase 3: Multi-Layer Retrieval.</strong> When you ask a question, the system queries using a combination of Embedding Similarity, TF-IDF keyword counting, and hard regex fallback to build a semantic context window.</li>
                <li><strong>Phase 4: Provenance-Tagged RAG.</strong> The LLM synthesises the retrieved paragraphs, explicitly forcing citations back to the source file, preventing hallucination.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        "### What This Project Is\n"
        "SENTINEL-LAC is an intelligence workspace that converts patrol reports into a searchable wiki, "
        "visual map, pattern analytics, relationship graph, and analyst chat workflow."
    )
    st.markdown(
        "### LLM Wiki vs LLM Analysis\n"
        "- `LLM Wiki`: the structured local knowledge base generated from ingested patrol reports.\n"
        "- `Wiki Lookup`: instant exact/factual retrieval from that wiki (no synthesis).\n"
        "- `LLM Analysis`: cross-report synthesis using Ollama model when online."
    )
    st.markdown(
        "### How To Use (Recommended Flow)\n"
        "1. In `Operator Chat`, upload reports and run ingest.\n"
        "2. Open `Tactical Map` to inspect location markers and high-sensitivity halos.\n"
        "3. Use `Pattern Analysis` for trends and pattern search across report corpus.\n"
        "4. Use `Link Analysis` to inspect entity connectivity and hubs.\n"
        "5. Ask questions in chat, then approve strong answers to commit back into wiki."
    )
    st.markdown(
        "### Quick Interpretation Tips\n"
        "- Deep + repeated drone signals = higher threat posture.\n"
        "- Repeated TTS in one sector = likely recurring route/activity.\n"
        "- Same hubs appearing in link graph = high-priority entities."
    )
