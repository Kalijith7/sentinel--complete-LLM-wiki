#!/usr/bin/env python3
"""
llm-wiki CLI
============
The main command-line interface for the LLM Knowledge Base system.
Implements the full workflow shown in the diagrams:

  llm-wiki ingest <file>         # Phase 1+2: ingest file → compile → wiki
  llm-wiki ingest-all            # Phase 1+2: process everything in raw/
  llm-wiki watch                 # Auto-watch raw/ for new files
  llm-wiki query "<question>"    # Phase 3: ask the wiki a question
  llm-wiki lint                  # Phase 4: health checks
  llm-wiki index                 # Phase 4: rebuild summaries index
  llm-wiki render --format md    # Output: markdown report
  llm-wiki render --format marp  # Output: Marp slides
  llm-wiki render --format charts# Output: matplotlib charts
  llm-wiki status                # Show wiki stats
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from wiki_engine.pipeline    import WikiPipeline
from wiki_engine.renderer    import WikiRenderer
from wiki_engine.backlinks   import BacklinkResolver

WIKI_ROOT   = Path("wiki")
RAW_ROOT    = Path("raw")
OUTPUTS_DIR = Path("outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    path = Path(args.file)
    if not path.exists():
        print(f"✗ File not found: {path}")
        sys.exit(1)
    pipeline = _make_pipeline(args)
    result   = asyncio.run(pipeline.ingest_file(path, force=getattr(args,"force",False)))
    _print_json(result)


def cmd_ingest_all(args):
    pipeline = _make_pipeline(args)
    results  = asyncio.run(pipeline.ingest_all(force=getattr(args,"force",False)))
    ok    = [r for r in results if "error" not in r]
    fails = [r for r in results if "error" in r]
    print(f"\n✓ Ingested {len(ok)} files")
    if fails:
        print(f"✗ Failed: {len(fails)}")
        for f in fails:
            print(f"  {f['source']}: {f['error']}")
    total_articles = sum(r.get("articles_written", 0) for r in ok)
    print(f"✓ Total articles written: {total_articles}")


def cmd_watch(args):
    from wiki_engine.watcher import run_watcher
    print(f"Watching {RAW_ROOT.absolute()}")
    print(f"Model: {args.model}  Ollama: {args.ollama_url}")
    print("Drop files into raw/ to auto-compile. Ctrl+C to stop.\n")
    run_watcher(args.ollama_url, args.model)


def cmd_query(args):
    pipeline = _make_pipeline(args)
    result   = asyncio.run(pipeline.query(args.question, hops=args.hops))
    print("\n" + "="*60)
    print(result["answer"])
    print("="*60)
    print(f"\nArticles consulted ({len(result['articles_consulted'])}):")
    for a in result["articles_consulted"]:
        print(f"  • {a}")
    print(f"Context: {result['context_chars']:,} chars  |  Hops: {result['hop_depth']}")


def cmd_lint(args):
    pipeline = _make_pipeline(args)
    result   = asyncio.run(pipeline.lint())
    t = result["totals"]
    print(f"\n{'─'*40}")
    print(f" Wiki Lint Report")
    print(f"{'─'*40}")
    print(f"  Orphaned articles:  {t['orphans']}")
    print(f"  Broken links:       {t['broken_links']}")
    print(f"  Stale (>30d):       {t['stale']}")
    print(f"  Low confidence:     {t['low_conf']}")
    if result.get("suggested_articles"):
        print(f"\n  Suggested stubs ({len(result['suggested_articles'])}):")
        for s in result["suggested_articles"][:10]:
            print(f"    [[{s}]]")
    print(f"\n  Full report: wiki/.meta/lint_report.md")


def cmd_index(args):
    pipeline = _make_pipeline(args)
    result   = asyncio.run(pipeline.index())
    print(f"✓ Indexed {len(result.get('articles', []))} articles")
    print(f"✓ Tags: {len(result.get('tags', {}))}")
    print(f"✓ Entity map entries: {len(result.get('entities', {}))}")
    print(f"  Catalog: wiki/_catalog.md")


def cmd_render(args):
    renderer = WikiRenderer(WIKI_ROOT, OUTPUTS_DIR)
    fmt = args.format

    if fmt == "md":
        out = renderer.render_markdown_report(
            title=args.title or "Knowledge Base Report",
            category_filter=args.category,
        )
        print(f"✓ Markdown report: {out}")

    elif fmt == "marp":
        out = renderer.render_marp_slides(
            title=args.title or "Knowledge Base",
            max_slides=args.max_slides,
        )
        print(f"✓ Marp slides: {out}")
        print(f"  To convert: npx @marp-team/marp-cli {out} --html")

    elif fmt == "charts":
        charts = renderer.render_charts()
        if charts:
            for c in charts:
                print(f"✓ Chart: {c}")
        else:
            print("No chart data available. Run 'llm-wiki index' first.")

    else:
        print(f"Unknown format: {fmt}. Use: md | marp | charts")
        sys.exit(1)


def cmd_log(args):
    """Show recent log.md entries."""
    log_p = WIKI_ROOT / "log.md"
    if not log_p.exists():
        print("No log.md yet. Run: llm-wiki ingest <file>")
        return
    import re
    text    = log_p.read_text()
    entries = re.split(r"(?=^## \[)", text, flags=re.MULTILINE)
    entries = [e.strip() for e in entries if e.strip().startswith("## [")]
    if getattr(args, "grep", None):
        entries = [e for e in entries if args.grep.lower() in e.lower()]
    tail    = getattr(args, "tail", 10)
    recent  = entries[-tail:]
    print(f"\n  Last {len(recent)} log entries")
    for entry in recent:
        lines   = entry.split("\n")
        header  = lines[0]
        details = [l.strip() for l in lines[1:] if l.strip().startswith("-")][:3]
        print(f"\n{header}")
        for d in details:
            print(f"  {d}")
    print()


def cmd_status(args):
    bl    = BacklinkResolver(WIKI_ROOT)
    stats = bl.stats()

    all_md = list(WIKI_ROOT.rglob("*.md"))
    visible = [p for p in all_md if ".meta" not in str(p)]

    raw_files = [f for f in RAW_ROOT.rglob("*") if f.is_file()] \
                if RAW_ROOT.exists() else []

    meta_index = WIKI_ROOT / ".meta" / "index.json"
    tag_count  = 0
    if meta_index.exists():
        try:
            idx = json.loads(meta_index.read_text())
            tag_count = len(idx.get("tags", {}))
        except Exception:
            pass

    print(f"\n{'═'*40}")
    print(f"  LLM Wiki — Status")
    print(f"{'═'*40}")
    print(f"  Wiki articles:  {len(visible)}")
    print(f"  Raw documents:  {len(raw_files)}")
    print(f"  Backlinks:      {stats.get('total_links', 0)}")
    print(f"  Tags:           {tag_count}")
    print(f"  Orphans:        {stats.get('orphans', '?')}")
    print(f"{'─'*40}")
    print(f"  Wiki root:   {WIKI_ROOT.absolute()}")
    print(f"  Raw root:    {RAW_ROOT.absolute()}")
    print(f"  Outputs:     {OUTPUTS_DIR.absolute()}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-wiki",
        description="LLM Knowledge Base — Karpathy-style living wiki",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                        help="Ollama API URL (default: env OLLAMA_URL or http://localhost:11434)")
    parser.add_argument("--auto-git", action="store_true",
                        help="Auto git-commit after each ingest")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
                        help="Ollama model name (default: env OLLAMA_MODEL or llama3.2:3b)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a single file")
    p_ingest.add_argument("file", help="Path to file in raw/ or elsewhere")
    p_ingest.add_argument("--force", action="store_true",
                         help="Re-ingest even if already in log.md")

    # ingest-all
    p_ingestall = sub.add_parser("ingest-all", help="Ingest all files in raw/")
    p_ingestall.add_argument("--force", action="store_true",
                             help="Re-ingest already-logged files")

    # watch
    sub.add_parser("watch", help="Watch raw/ and auto-ingest new files")

    # query
    p_query = sub.add_parser("query", help="Ask the wiki a question")
    p_query.add_argument("question", help="Question to answer")
    p_query.add_argument("--hops", type=int, default=3,
                         help="Backlink traversal depth (default: 3)")

    # lint
    sub.add_parser("lint", help="Run wiki health checks (Phase 4)")

    # index
    sub.add_parser("index", help="Rebuild wiki index and catalog")

    # render
    p_render = sub.add_parser("render", help="Generate derived outputs")
    p_render.add_argument("--format", choices=["md", "marp", "charts"],
                          default="md", help="Output format")
    p_render.add_argument("--title", default=None)
    p_render.add_argument("--category", default=None,
                           help="Filter articles by category")
    p_render.add_argument("--max-slides", type=int, default=20)

    # log
    p_log = sub.add_parser("log", help="Show recent log.md entries")
    p_log.add_argument("--tail", type=int, default=10)
    p_log.add_argument("--grep", default=None,
                       help="Filter by operation: ingest|query|lint")

    # status
    sub.add_parser("status", help="Show wiki statistics")

    return parser


def _make_pipeline(args) -> WikiPipeline:
    return WikiPipeline(
        ollama_url=args.ollama_url,
        model=args.model,
        auto_git=getattr(args, "auto_git", False),
    )


def _print_json(data):
    print(json.dumps(data, indent=2, default=str))


def main():
    parser = build_parser()
    args   = parser.parse_args()

    WIKI_ROOT.mkdir(exist_ok=True)
    RAW_ROOT.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    commands = {
        "ingest":      cmd_ingest,
        "ingest-all":  cmd_ingest_all,
        "watch":       cmd_watch,
        "query":       cmd_query,
        "lint":        cmd_lint,
        "index":       cmd_index,
        "render":      cmd_render,
        "status":      cmd_status,
        "log":         cmd_log,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
