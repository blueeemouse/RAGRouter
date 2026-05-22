"""
Hidden States Extraction Module

This module provides tools for extracting hidden states from LLM during
the prefill phase. The extracted hidden states are used for training
the RAG Router.

Main Components:
- HiddenStatesExtractor: Core class for extracting hidden states
"""
from HiddenStatesExtraction.extractor import HiddenStatesExtractor

__all__ = ["HiddenStatesExtractor"]
