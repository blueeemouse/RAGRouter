"""
Run Embedding Generation Process

This script chunks the corpus and generates embeddings.

Usage:
    python Run/Process/run_embedding.py
    python Run/Process/run_embedding.py --dataset test
"""
import argparse
import sys

from RAGCore.Chunk.ChunkDo import ChunkProcessor
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
from Config.PathConfig import PathConfig


def run_embedding(dataset_name: str):
    """Run chunking and embedding generation for a dataset

    Args:
        dataset_name: Name of the dataset (e.g., "hotpotqa", "test")
    """
    try:
        # Step 1: Chunking
        chunk_processor = ChunkProcessor()
        corpus_path = PathConfig.get_corpus_path(dataset_name)
        corpus = chunk_processor.load_corpus(corpus_path)
        chunks_by_doc = chunk_processor.process_corpus(corpus)

        # Step 2: Generate embeddings
        embedding_processor = EmbeddingProcessor()
        embeddings_by_doc = embedding_processor.process_chunks(chunks_by_doc)

        # Step 3: Save embeddings
        EmbeddingSaver.save(embeddings_by_doc, dataset_name)

        return 0
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return 1


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Run embedding generation")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    args = parser.parse_args()

    exit_code = run_embedding(args.dataset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
