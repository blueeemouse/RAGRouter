"""
Run Iterative RAG QA

This script runs Iterative RAG: multi-round retrieval + LLM evaluation + generation.
Model configuration is read from Config/LLMConfig.py
Retriever configuration is read from Config/RetrieverConfig.py

Usage:
    python run_iterative_rag.py --dataset graphragBench_medical
    python run_iterative_rag.py --dataset graphragBench_medical --no-resume
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RAGCore.Retriever.IterativeRAG.IterativeRAGDo import IterativeRAGProcessor
from RAGCore.Retriever.IterativeRAG.IterativeRAGSave import IterativeRAGSaver
from Config.PathConfig import PathConfig
from Config.RetrieverConfig import RetrieverConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Iterative RAG QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., graphragBench_medical)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    print("Iterative RAG QA")
    print(f"Dataset: {args.dataset}")
    print(f"Retriever: {RetrieverConfig.ITERATIVE_RETRIEVER}")
    print(f"Max Iterations: {RetrieverConfig.ITERATIVE_MAX_ITERATIONS}")
    print(f"Resume: {not args.no_resume}")

    # Initialize processor
    processor = IterativeRAGProcessor(dataset_name=args.dataset)

    # Process questions
    results = processor.process(
        dataset_name=args.dataset,
        resume=not args.no_resume
    )

    # Save final results
    IterativeRAGSaver.save_all(results, processor.model_name, args.dataset)

    # Print summary
    output_path = PathConfig.get_iterative_rag_path(processor.model_name, args.dataset)
    print("Iterative RAG QA Complete!")
    print(f"Results saved to: {output_path}")
    print(f"  - answer.jsonl: {len(results)} answers")

    # Statistics
    if results:
        total_rounds = sum(r.get('rounds', 0) for r in results)
        avg_rounds = total_rounds / len(results)
        print(f"  - Average rounds: {avg_rounds:.2f}")


if __name__ == "__main__":
    main()
