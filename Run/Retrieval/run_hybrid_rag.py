"""
Run Hybrid RAG QA

This script runs Hybrid RAG: combining NaiveRAG and GraphRAG retrieval results + LLM generation.
Model configuration is read from Config/LLMConfig.py

Prerequisites:
    - NaiveRAG retrieval results must exist for the dataset
    - GraphRAG retrieval results must exist for the dataset

Usage:
    python run_hybrid_rag.py --dataset UltraDomain_mix
    python run_hybrid_rag.py --dataset UltraDomain_mix --no-resume
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RAGCore.Retriever.HybridRAG.HybridRAGDo import HybridRAGProcessor
from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
from Config.PathConfig import PathConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Hybrid RAG QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., UltraDomain_mix)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    print("Hybrid RAG QA")
    print(f"Dataset: {args.dataset}")
    print(f"Resume: {not args.no_resume}")

    # Initialize processor
    processor = HybridRAGProcessor(dataset_name=args.dataset)

    # Process questions
    results = processor.process(
        dataset_name=args.dataset,
        resume=not args.no_resume
    )

    # Save final results
    HybridRAGSaver.save_all(results, processor.model_name, args.dataset)

    # Print summary
    output_path = PathConfig.get_hybrid_rag_path(processor.model_name, args.dataset)
    print("Hybrid RAG QA Complete!")
    print(f"Results saved to: {output_path}")
    print(f"  - answer.jsonl: {len(results)} answers")


if __name__ == "__main__":
    main()
