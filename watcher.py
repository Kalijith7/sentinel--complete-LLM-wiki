"""
File Watcher — watches raw/ directory and auto-triggers wiki compilation.
Implements the "raw/ directory — Source documents staging" box from the diagram.
Run with: python -m wiki_engine.watcher
"""
from __future__ import annotations
import asyncio
import logging
import sys
import os
from pathlib import Path

log = logging.getLogger(__name__)

WATCH_DIR  = Path("raw")
WIKI_ROOT  = Path("wiki")
LOCK_DIR   = Path(".processing_locks")

SUPPORTED = {
    ".pdf", ".docx", ".doc", ".md", ".markdown",
    ".txt", ".csv", ".jpg", ".jpeg", ".png", ".tiff",
}


async def watch(ollama_url: str | None = None,
                model: str | None = None):
    """
    Watch raw/ and compile new files to wiki/.
    Uses watchfiles for efficient filesystem events.
    """
    try:
        from watchfiles import awatch
    except ImportError:
        log.error("watchfiles not installed. Run: pip install watchfiles")
        sys.exit(1)

    from .pipeline import WikiPipeline

    WATCH_DIR.mkdir(exist_ok=True)
    WIKI_ROOT.mkdir(exist_ok=True)
    LOCK_DIR.mkdir(exist_ok=True)

    ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    pipeline = WikiPipeline(ollama_url=ollama_url, model=model)
    log.info(f"Watching {WATCH_DIR.absolute()} for new documents...")
    log.info(f"Ollama: {ollama_url}  Model: {model}")

    async for changes in awatch(str(WATCH_DIR)):
        for change_type, path_str in changes:
            path = Path(path_str)

            # Only process file additions/modifications
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED:
                log.debug(f"Skipping unsupported type: {path.suffix}")
                continue

            lock = LOCK_DIR / f"{path.name}.lock"
            if lock.exists():
                log.debug(f"Already processing: {path.name}")
                continue

            # Set lock, process, release
            lock.touch()
            try:
                log.info(f"New file detected: {path.name}")
                result = await pipeline.ingest_file(path)
                log.info(f"  ✓ {result['articles_written']} articles written")
                log.info(f"  ✓ {result['backlinks']} backlinks in graph")
            except Exception as e:
                log.error(f"  ✗ Failed to ingest {path.name}: {e}", exc_info=True)
            finally:
                lock.unlink(missing_ok=True)


def run_watcher(ollama_url: str | None = None,
                model: str | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(watch(ollama_url, model))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Watch raw/ and auto-compile to wiki/")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--model",      default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    args = p.parse_args()
    run_watcher(args.ollama_url, args.model)
