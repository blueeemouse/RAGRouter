"""
Run Index Building Process

This script loads embeddings and builds FAISS index.

Usage:
    python Run/Process/run_index.py
    python Run/Process/run_index.py --dataset test
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
from RAGCore.Index.IndexDo import IndexProcessor
from RAGCore.Index.IndexSave import IndexSaver


def run_index(dataset_name: str):
    """Run index building for a dataset

    Args:
        dataset_name: Name of the dataset (e.g., "hotpotqa", "test")
    """
    try:
        # Step 1: Load embeddings
        embeddings_by_doc = EmbeddingSaver.load(dataset_name)

        # Step 2: Build index
        index_processor = IndexProcessor()
        index_data = index_processor.build_index(embeddings_by_doc)

        # Step 3: Save index
        IndexSaver.save(index_data, dataset_name)

        return 0
    except Exception as e:
        print(f"Index building failed: {e}")
        return 1


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Run index building")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    args = parser.parse_args()

    exit_code = run_index(args.dataset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
