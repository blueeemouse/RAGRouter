"""
Query Generation Module

This module provides tools for generating benchmark queries for RAG evaluation:
- Single-hop (Factual): Simple factual questions from a single passage
- Multi-hop (Complex Reasoning): Questions requiring multiple documents
- Summary: Synthesis questions about an entity across multiple documents

Submodules:
- Preprocess: IndexBuilder, ChainBuilder for sample preparation
- ValidateDo: QueryValidator for query quality validation
"""

from BenchCore.QueryGeneration.GenerateDo import QueryGenerator
from BenchCore.QueryGeneration.GenerateSave import QueryGenerateSaver
from BenchCore.QueryGeneration.ValidateDo import QueryValidator
from BenchCore.QueryGeneration.Preprocess import IndexBuilder, ChainBuilder

__all__ = [
    "QueryGenerator",
    "QueryGenerateSaver",
    "QueryValidator",
    "IndexBuilder",
    "ChainBuilder",
]
