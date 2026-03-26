"""
Run Graph Generation Process

This script chunks the corpus and builds knowledge graph.

Usage:
    python Run/Process/run_graph.py
    python Run/Process/run_graph.py --dataset test
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RAGCore.Chunk.ChunkDo import ChunkProcessor
from RAGCore.Graph.GraphDo import GraphProcessor
from RAGCore.Graph.GraphSave import GraphSaver
from Config.PathConfig import PathConfig


def run_graph(dataset_name: str):
    """Run chunking and graph generation for a dataset

    Args:
        dataset_name: Name of the dataset (e.g., "hotpotqa", "test")
    """
    try:
        # Step 1: Chunking
        chunk_processor = ChunkProcessor()
        corpus_path = PathConfig.get_corpus_path(dataset_name)
        corpus = chunk_processor.load_corpus(corpus_path)
        chunks_by_doc = chunk_processor.process_corpus(corpus)

        # Step 2: Build graph (extract entities, triplets, and build graph)
        graph_processor = GraphProcessor()
        graph_result = graph_processor.process(chunks_by_doc, dataset_name=dataset_name, resume=True)

        # Step 3: Save graph
        GraphSaver.save(graph_result, dataset_name)

        return 0
    except Exception as e:
        print(f"Graph generation failed: {e}")
        return 1


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Run graph generation")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    args = parser.parse_args()

    exit_code = run_graph(args.dataset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
