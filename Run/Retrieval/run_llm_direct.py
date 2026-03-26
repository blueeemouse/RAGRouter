"""
Run LLM Direct QA

This script runs direct LLM question answering without any retrieval.
Model configuration is read from Config/LLMConfig.py

Usage:
    python run_llm_direct.py --dataset quality
    python run_llm_direct.py --dataset quality --no-resume
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RAGCore.Retriever.LLMDirect.LLMDirectDo import LLMDirectProcessor
from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
from Config.PathConfig import PathConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM Direct QA (Model configured in LLMConfig.py)",
        epilog="To change the model, edit PROVIDER in Config/LLMConfig.py"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., quality)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    print("LLM Direct QA")
    print(f"Dataset: {args.dataset}")
    print(f"Resume: {not args.no_resume}")

    # Initialize processor
    processor = LLMDirectProcessor()

    # Process questions
    results = processor.process(
        dataset_name=args.dataset,
        resume=not args.no_resume
    )

    # Save final results
    LLMDirectSaver.save_all(results, processor.model_name, args.dataset)

    # Print summary
    output_path = PathConfig.get_llm_direct_path(processor.model_name, args.dataset)
    print("LLM Direct QA Complete!")
    print(f"Results saved to: {output_path}")
    print(f"  - answer.jsonl: {len(results)} answers")


if __name__ == "__main__":
    main()
