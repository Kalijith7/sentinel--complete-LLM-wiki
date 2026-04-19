"""
wiki_engine/intel_schema.py
===========================
Single Source of Truth for all Sentinel-LAC intelligence schema definitions.

This module contains ONLY constants and compiled patterns.
No business logic. Every other module imports from here.

Design principle: if you change a schema field, change it here and
nowhere else.  All downstream modules use these names and values.
"""
from __future__ import annotations

import re
from typing import Final


# ─────────────────────────────────────────────────────────────────────────────
# A — VERBATIM FIELDS
#
# Fields whose values must be copied CHARACTER-FOR-CHARACTER from the
# source document.  The compiler prompt embeds an explicit hard rule for
# every name in this list.  _validate_verbatim_fields() in compiler.py
# uses this list to audit LLM output against the original text.
# ─────────────────────────────────────────────────────────────────────────────

VERBATIM_FIELDS: Final[list[str]] = [
    "DTG",                    # Date-Time Group, e.g. "041530ZAPR24"
    "Coordinates",            # MGRS / LatLon — never paraphrased
    "Units_Involved",         # Unit designations verbatim
    "Patrol_Number",          # Report serial / patrol number
    "Depth_of_Transgression", # Exact distance/grid string
    "Equipment_Serials",      # Vehicle IDs, equipment serials
    "Grid_Depth",             # Grid reference for depth measurement
    "Altitude_Metres",        # Altitude — verbatim, do not compute
    "Duration",               # Duration of incident verbatim
]


# ─────────────────────────────────────────────────────────────────────────────
# B — ENUMERATED DOMAIN VALUES
#
# Used both in the compiler prompt (instructing the LLM which values are
# legal) and in dashboard rendering (colour coding, filtering).
# ─────────────────────────────────────────────────────────────────────────────

TTS_CATEGORIES: Final[list[str]] = [
    "tire_tracks",
    "boot_prints",
    "campsite_remains",
    "fire_ash",
    "surveillance_camera_installed",
    "surveillance_camera_removed",
    "equipment_left_behind",
    "marker_placed",
    "marker_removed",
    "cut_vegetation",
    "disturbed_terrain",
    "supplies_cache",
    "comms_equipment",
    "observation_post_signs",
    "construction_activity",
    "vehicle_impressions",
    "rope_crossing_marks",
    "ammunition_casings",
    "other",
]

INCIDENT_TYPES: Final[list[str]] = [
    "Face-off",
    "Infrastructure_Buildup",
    "Airspace_Violation",
    "Patrol_Incursion",
    "Surveillance_Activity",
    "Marker_Placement",
    "Construction",
    "Obstruction",
    "Standoff",
    "Other",
]

PATROL_OUTCOMES: Final[list[str]] = [
    "Resolved_by_Dialogue",
    "PLA_Withdrew",
    "Indian_Withdrew",
    "Standoff_Continuing",
    "Escalated",
    "Physical_Altercation",
    "Unknown",
]

PLA_RANK_TERMS: Final[list[str]] = [
    "General", "Lieutenant General", "Major General", "Brigadier General",
    "Senior Colonel", "Colonel", "Lieutenant Colonel", "Major",
    "Captain", "First Lieutenant", "Second Lieutenant", "Lieutenant",
    "Staff Sergeant", "Sergeant", "Corporal", "Private First Class",
    "Private", "Soldier", "Officer", "NCO",
]

DOC_TYPES: Final[list[str]] = [
    "patrol_report", "sitrep", "intrep", "humint", "imint",
    "after_action_report", "incident_log", "other",
]


# ─────────────────────────────────────────────────────────────────────────────
# C — COMPILED REGEX PATTERNS
#
# Compiled once at import time, reused across compiler, query_engine,
# parser, and pipeline.  Avoids repeated compilation in hot paths.
# ─────────────────────────────────────────────────────────────────────────────

# Military Grid Reference System — 6, 8, or 10-digit precision
MGRS_RE: Final[re.Pattern[str]] = re.compile(
    r'\b\d{2}[A-Z]{3}\s?\d{4,10}\b'
)

# Date-Time Group — e.g. "041530ZAPR24"
DTG_RE: Final[re.Pattern[str]] = re.compile(
    r'\b\d{2}\d{4}[A-Z][A-Z]{3}\d{2}\b'
)

# Latitude/Longitude — DMS format  e.g. 28°30'45"N 79°15'20"E
LATLON_DMS_RE: Final[re.Pattern[str]] = re.compile(
    r'\d{1,3}[°\s]\d{1,2}[\'′\s]\d{0,2}(?:\.\d+)?[\"″\s]?[NSns]'
    r'[\s,]+'
    r'\d{1,3}[°\s]\d{1,2}[\'′\s]\d{0,2}(?:\.\d+)?[\"″\s]?[EWew]'
)

# Latitude/Longitude — decimal degrees e.g. "28.5123, 79.2541"
LATLON_DD_RE: Final[re.Pattern[str]] = re.compile(
    r'-?\d{1,3}\.\d{4,}[\s,]+-?\d{1,3}\.\d{4,}'
)

# PLA unit designations — e.g. "PLA 76th Group Army"
UNIT_RE: Final[re.Pattern[str]] = re.compile(
    r'(?:PLAA?F?|PLAA|PLAN|PLAAF|PLARF|PLA)\s+[\w\d/\s\-]{2,50}'
)

# Depth of transgression — numeric distance
DEPTH_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(?:m\b|metres?\b|km\b|kilometers?\b|meters?\b)',
    re.IGNORECASE,
)

# Personnel name detection near images (used by parser caption analysis)
RANK_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:' + '|'.join(re.escape(r) for r in PLA_RANK_TERMS) + r')\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# D — ENTITY RESOLUTION ALIAS MAP
#
# Maps known variant spellings / abbreviations → canonical unit name.
# Used by query_engine._resolve_entity_aliases() with thefuzz fallback.
# Extend this map as new variants are encountered in the corpus.
# ─────────────────────────────────────────────────────────────────────────────

UNIT_ALIASES: Final[dict[str, str]] = {
    # 76th Group Army variants
    "76th group army":              "PLA 76th Group Army",
    "76th ga":                      "PLA 76th Group Army",
    "pla 76th":                     "PLA 76th Group Army",
    "76 group army":                "PLA 76th Group Army",
    "76th combined arms army":      "PLA 76th Group Army",

    # 78th Group Army variants
    "78th group army":              "PLA 78th Group Army",
    "78th ga":                      "PLA 78th Group Army",
    "pla 78th":                     "PLA 78th Group Army",

    # 77th Group Army (Chengdu MR successor)
    "77th group army":              "PLA 77th Group Army",
    "pla 77th":                     "PLA 77th Group Army",

    # Military District variants
    "xinjiang md":                  "PLA Xinjiang Military District",
    "xjmd":                         "PLA Xinjiang Military District",
    "xinjiang military district":   "PLA Xinjiang Military District",
    "tibet md":                     "PLA Tibet Military District",
    "tmd":                          "PLA Tibet Military District",
    "tibet military district":      "PLA Tibet Military District",

    # Theatre Command variants
    "western theatre":              "PLA Western Theatre Command",
    "western theater":              "PLA Western Theatre Command",
    "wtc":                          "PLA Western Theatre Command",
    "west theatre":                 "PLA Western Theatre Command",
    "southern theatre":             "PLA Southern Theatre Command",
    "southern theater":             "PLA Southern Theatre Command",
    "stc":                          "PLA Southern Theatre Command",

    # Generic abbreviations
    "pla":                          "People's Liberation Army",
    "plaa":                         "PLA Army",
    "plan":                         "PLA Navy",
    "plaaf":                        "PLA Air Force",
    "plarf":                        "PLA Rocket Force",
    "ssf":                          "PLA Strategic Support Force",
}


# ─────────────────────────────────────────────────────────────────────────────
# E — INTEL KEYWORD TRIGGER LIST
#
# pipeline._is_intel_report() returns True when ≥2 of these keywords
# appear in the first 1000 characters of parsed document text.
# Case-insensitive matching.
# ─────────────────────────────────────────────────────────────────────────────

INTEL_KEYWORDS: Final[list[str]] = [
    "patrol",
    "face-off",
    "faceoff",
    "face off",
    "pla",
    "lac",
    "line of actual control",
    "transgression",
    "intrusion",
    "sitrep",
    "intrep",
    "mgrs",
    "dtg",
    "forward patrol base",
    "tell tale",
    "tell-tale",
    "tts",
    "depth of ingress",
    "depth of transgression",
    "border violation",
    "incursion",
    "standoff",
    "peoples liberation army",
    "people's liberation army",
    "chinese troops",
    "pla troops",
    "arunachal",
    "aksai chin",
    "depsang",
    "galwan",
    "tawang",
    "sector",
]


# ─────────────────────────────────────────────────────────────────────────────
# F — INTEL JSON SCHEMA TEMPLATE
#
# The exact JSON structure the compiler prompt asks the LLM to populate.
# Defined as a constant string here so the prompt stays clean.
# Uses double-braces {{ }} for literal braces (this string is NOT an
# f-string — it is used via str.replace() in the prompt builder).
# ─────────────────────────────────────────────────────────────────────────────

INTEL_JSON_SCHEMA: Final[str] = """{
  "meta": {
    "title": "document title or best guess",
    "doc_type": "patrol_report|sitrep|intrep|after_action_report|other"
  },
  "intel_meta": {
    "DTG": "VERBATIM Date-Time Group string, e.g. 041530ZAPR24, or null",
    "Incident_Type": "ONE OF: Face-off|Infrastructure_Buildup|Airspace_Violation|Patrol_Incursion|Surveillance_Activity|Marker_Placement|Construction|Obstruction|Standoff|Other",
    "Coordinates": [
      {"system": "MGRS|LatLon|LatLonDMS|GridRef", "value": "VERBATIM coordinate string"}
    ],
    "Units_Involved": {
      "indian": ["VERBATIM unit designation string"],
      "pla":    ["VERBATIM unit designation string"]
    },
    "Patrol_Number": "VERBATIM patrol or report serial, or null",
    "Depth_of_Transgression": "VERBATIM distance/area/grid string, e.g. 300 metres past Line X, or null",
    "Tell_Tale_Signs": [
      {
        "category": "ONE OF: tire_tracks|boot_prints|campsite_remains|fire_ash|surveillance_camera_installed|surveillance_camera_removed|equipment_left_behind|marker_placed|marker_removed|cut_vegetation|disturbed_terrain|supplies_cache|comms_equipment|observation_post_signs|construction_activity|vehicle_impressions|rope_crossing_marks|ammunition_casings|other",
        "description": "VERBATIM description from report"
      }
    ],
    "Weather": "VERBATIM weather description, or null",
    "Altitude_Metres": "VERBATIM altitude string, or null",
    "Patrol_Outcome": "ONE OF: Resolved_by_Dialogue|PLA_Withdrew|Indian_Withdrew|Standoff_Continuing|Escalated|Physical_Altercation|Unknown|null",
    "Duration": "VERBATIM duration string, or null",
    "Personnel_Identified": [
      {
        "name":        "VERBATIM soldier name as written",
        "rank":        "VERBATIM rank as written",
        "unit":        "VERBATIM unit as written",
        "img_ref":     "img-0|img-1|... matching extracted image, or null",
        "caption":     "VERBATIM caption or adjacent text identifying this person"
      }
    ]
  },
  "summary": {
    "title":      "human-readable report title",
    "one_liner":  "one sentence — most operationally significant finding",
    "narrative":  "3-5 paragraph narrative of the incident",
    "key_points": "- bullet point\\n- bullet point",
    "gaps":       "what information is missing or unclear",
    "tags":       ["tag1", "tag2"],
    "entities":   [{"type": "PERSON|ORG|LOCATION|CONCEPT|TECH|EVENT", "value": "name"}],
    "related":    ["slug-of-related-article"]
  },
  "concepts": [
    {
      "name":        "Canonical entity name",
      "category":    "concepts|units|locations|persons|technologies",
      "description": "what this entity is",
      "details":     "specifics from this document",
      "key_facts":   "- fact\\n- fact",
      "notes":       "caveats or open questions",
      "related":     ["slug-1"],
      "confidence":  0.9
    }
  ],
  "events": [
    {
      "title":       "Descriptive event title",
      "date":        "YYYY-MM-DD or DTG or partial",
      "type":        "Face-off|incident|patrol|other",
      "description": "what happened",
      "outcome":     "result or significance",
      "related":     ["slug-1"],
      "confidence":  0.9
    }
  ]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# G — DASHBOARD COLOUR MAPS
#
# Used by dashboard.py for consistent rendering.
# ─────────────────────────────────────────────────────────────────────────────

INCIDENT_COLOUR: Final[dict[str, str]] = {
    "Face-off":              "#ef4444",   # red
    "Infrastructure_Buildup":"#f97316",   # orange
    "Airspace_Violation":    "#a855f7",   # purple
    "Patrol_Incursion":      "#3b82f6",   # blue
    "Surveillance_Activity": "#eab308",   # yellow
    "Marker_Placement":      "#06b6d4",   # cyan
    "Construction":          "#f97316",   # orange
    "Obstruction":           "#ec4899",   # pink
    "Standoff":              "#f59e0b",   # amber
    "Other":                 "#6b7280",   # grey
}

NODE_COLOUR: Final[dict[str, str]] = {
    "personnel": "#f59e0b",   # amber
    "units":     "#ef4444",   # red
    "locations": "#3b82f6",   # blue
    "events":    "#eab308",   # yellow
    "summaries": "#22c55e",   # green
    "concepts":  "#8b5cf6",   # violet
    "derived":   "#06b6d4",   # cyan
    "other":     "#6b7280",   # grey
}

TTS_COLOUR: Final[dict[str, str]] = {
    "tire_tracks":                    "#f97316",
    "boot_prints":                    "#fb923c",
    "campsite_remains":               "#fbbf24",
    "fire_ash":                       "#f59e0b",
    "surveillance_camera_installed":  "#ef4444",
    "surveillance_camera_removed":    "#fca5a5",
    "equipment_left_behind":          "#a855f7",
    "marker_placed":                  "#3b82f6",
    "marker_removed":                 "#93c5fd",
    "cut_vegetation":                 "#22c55e",
    "disturbed_terrain":              "#84cc16",
    "supplies_cache":                 "#06b6d4",
    "comms_equipment":                "#8b5cf6",
    "observation_post_signs":         "#ec4899",
    "construction_activity":          "#f97316",
    "vehicle_impressions":            "#fb923c",
    "rope_crossing_marks":            "#fbbf24",
    "ammunition_casings":             "#ef4444",
    "other":                          "#6b7280",
}
