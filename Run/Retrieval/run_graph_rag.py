"""
Run Graph RAG QA

This script runs Graph RAG: entity retrieval + graph traversal + LLM generation.
Model configuration is read from Config/LLMConfig.py

Usage:
    python run_graph_rag.py --dataset quality
    python run_graph_rag.py --dataset quality --no-resume
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RAGCore.Retriever.GraphRAG.GraphRAGDo import GraphRAGProcessor
from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
from Config.PathConfig import PathConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Graph RAG QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., quality)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    print("Graph RAG QA")
    print(f"Dataset: {args.dataset}")
    print(f"Resume: {not args.no_resume}")

    # Initialize processor
    processor = GraphRAGProcessor(dataset_name=args.dataset)

    # Process questions
    results = processor.process(
        dataset_name=args.dataset,
        resume=not args.no_resume
    )

    # Save final results
    GraphRAGSaver.save_all(results, processor.model_name, args.dataset)

    # Print summary
    output_path = PathConfig.get_graph_rag_path(processor.model_name, args.dataset)
    print("Graph RAG QA Complete!")
    print(f"Results saved to: {output_path}")
    print(f"  - answer.jsonl: {len(results)} answers")


if __name__ == "__main__":
    main()
