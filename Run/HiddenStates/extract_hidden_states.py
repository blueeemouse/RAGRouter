"""
Hidden States Extraction Script

This script extracts hidden states from LLM during the prefill phase
for all questions in a dataset. The extracted hidden states are saved
for training the RAG Router.

Usage:
    python Run/HiddenStates/extract_hidden_states.py --dataset musique
    python Run/HiddenStates/extract_hidden_states.py --dataset musique --no-resume
    python Run/HiddenStates/extract_hidden_states.py --dataset musique --layer-ids 8 16 24 31

Features:
    - Batch processing with progress tracking
    - Resume from previous progress
    - Configurable layer selection
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Config.HiddenStatesConfig import HiddenStatesConfig, AVAILABLE_DATASETS
from Config.PathConfig import PathConfig
from HiddenStatesExtraction.extractor import HiddenStatesExtractor


def load_questions(dataset_name: str) -> List[Dict[str, Any]]:
    """Load questions from dataset

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of question dicts with 'id' and 'question' keys
    """
    question_path = PathConfig.get_question_path(dataset_name)

    with open(question_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            questions = json.load(f)
        else:
            questions = [json.loads(line) for line in f if line.strip()]

    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from LLM for RAG Router training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(AVAILABLE_DATASETS)}

Examples:
  python extract_hidden_states.py --dataset musique
  python extract_hidden_states.py --dataset musique --no-resume
  python extract_hidden_states.py --dataset musique --layer-ids 8 16 24 31
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS,
        help="Dataset name to process"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous progress, start fresh"
    )

    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to extract (default: from config)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (default: from config)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("Hidden States Extraction")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {HiddenStatesConfig.MODEL_NAME}")
    print(f"Layer IDs: {args.layer_ids or HiddenStatesConfig.LAYER_IDS}")
    print(f"Resume: {not args.no_resume}")
    print("=" * 60)

    # Load questions
    print(f"\nLoading questions from {args.dataset}...")
    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions")

    # Limit samples if specified
    if args.max_samples:
        questions = questions[:args.max_samples]
        print(f"Limited to {len(questions)} samples for testing")

    # Setup output directory
    output_dir = HiddenStatesConfig.ensure_dir(args.dataset)
    progress_path = HiddenStatesConfig.get_progress_path(args.dataset)
    print(f"Output directory: {output_dir}")

    # Load progress if resuming
    processed_ids = set()
    if not args.no_resume:
        processed_ids = set()
        # Check existing files
        for fname in os.listdir(output_dir):
            if fname.endswith(".safetensors"):
                sample_id = fname.replace(".safetensors", "")
                processed_ids.add(sample_id)
        print(f"Found {len(processed_ids)} already processed samples")

    # Filter questions to process
    questions_to_process = [
        q for q in questions
        if q.get("id") and q.get("question") and q.get("id") not in processed_ids
    ]

    if not questions_to_process:
        print("\nAll questions already processed!")
        return

    print(f"Processing {len(questions_to_process)} questions...")

    # Initialize extractor
    extractor = HiddenStatesExtractor(layer_ids=args.layer_ids)

    try:
        # Process in batches
        batch_size = args.batch_size or HiddenStatesConfig.BATCH_SIZE
        total_processed = 0

        for i in range(0, len(questions_to_process), batch_size):
            batch = questions_to_process[i:i + batch_size]

            # Extract hidden states
            results = extractor.extract_batch(batch, show_progress=True)

            # Save results immediately
            for result in results:
                output_path = HiddenStatesConfig.get_sample_path(
                    args.dataset, result.sample_id
                )
                extractor.save_result(result, output_path)
                processed_ids.add(result.sample_id)

            total_processed += len(results)

            # Save progress
            extractor.save_progress(processed_ids, progress_path)

            print(f"\nBatch {i // batch_size + 1} complete. Total processed: {total_processed}")

        # Save metadata
        metadata = {
            "model_path": HiddenStatesConfig.MODEL_PATH,
            "model_name": HiddenStatesConfig.MODEL_NAME,
            "layer_ids": args.layer_ids or HiddenStatesConfig.LAYER_IDS,
            "hidden_size": HiddenStatesConfig.HIDDEN_SIZE,
            "dtype": HiddenStatesConfig.DTYPE,
            "aggregation_methods": HiddenStatesConfig.AGGREGATION_METHODS,
            "total_samples": len(processed_ids),
            "dataset": args.dataset,
        }
        metadata_path = HiddenStatesConfig.get_metadata_path(args.dataset)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("Extraction Complete!")
        print("=" * 60)
        print(f"Total processed: {len(processed_ids)}")
        print(f"Output directory: {output_dir}")
        print(f"Metadata saved to: {metadata_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved to:", progress_path)
        print(f"Processed {len(processed_ids)} samples before interruption")

    except Exception as e:
        print(f"\nError during extraction: {e}")
        raise

    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
