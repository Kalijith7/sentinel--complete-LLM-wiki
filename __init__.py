"""
LLM Wiki Engine
Karpathy-style living knowledge base compiler.
"""
from .pipeline import WikiPipeline
from .compiler import LLMCompiler, OllamaClient
from .parser   import DocumentParser
from .linter   import WikiLinter
from .indexer  import WikiIndexer
from .backlinks import BacklinkResolver
from .query_engine import WikiQueryEngine
from .renderer import WikiRenderer
from .embedder import EmbeddingIndex, cosine

__all__ = [
    "WikiPipeline",
    "LLMCompiler",
    "OllamaClient",
    "DocumentParser",
    "WikiLinter",
    "WikiIndexer",
    "BacklinkResolver",
    "WikiQueryEngine",
    "WikiRenderer",
    "EmbeddingIndex",
    "cosine",
]
