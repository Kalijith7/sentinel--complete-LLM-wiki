import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

EQUIPMENT_DB_PATH = Path("data/pla_equipment.json")
_EQUIPMENT_DB = None

def _load_db() -> dict:
    global _EQUIPMENT_DB
    if _EQUIPMENT_DB is None:
        if EQUIPMENT_DB_PATH.exists():
            try:
                _EQUIPMENT_DB = json.loads(EQUIPMENT_DB_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                log.error(f"Failed to load PLA equipment DB: {e}")
                _EQUIPMENT_DB = {}
        else:
            _EQUIPMENT_DB = {}
    return _EQUIPMENT_DB

def lookup_pla_equipment(code: str) -> dict | None:
    """Looks up a PLA equipment code in the static reference database."""
    db = _load_db()
    return db.get(code.upper())