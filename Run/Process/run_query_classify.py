"""
Run Query Classification

Classifies questions using different schemes (MemoRAG, etc.)
Saves results to QuestionType.json in the dataset's RawData directory.

Usage:
    python Run/Process/run_query_classify.py --dataset UltraDomain_mix --scheme memorag
    python Run/Process/run_query_classify.py --dataset UltraDomain_legal --scheme memorag
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RAGCore.Query.QueryClassifyDo import QueryClassifier
from RAGCore.Query.QueryClassifySave import QueryClassifySaver
from Config.QueryConfig import QueryConfig


def run_classification(dataset_name: str, scheme: str = "memorag"):
    """Run question classification for a dataset

    Args:
        dataset_name: Name of dataset (e.g., "UltraDomain_mix")
        scheme: Classification scheme to use

    Returns:
        0 if successful, 1 if failed
    """
    try:
        print("Query Classification")
        print(f"Dataset: {dataset_name}")
        print(f"Scheme: {scheme}")
        print(f"LLM Provider: {QueryConfig.PROVIDER}")

        # Step 1: Check if Question.json has existing question_type field
        print("\nChecking for existing classifications in Question.json...")
        extracted = QueryClassifySaver.extract_from_questions(dataset_name, scheme)

        if extracted:
            # Question.json already has question_type, save to QuestionType.json
            print(f"Found {len(extracted)} classifications in Question.json")
            print("Migrating to QuestionType.json...")
            QueryClassifySaver.save_question_types(extracted, dataset_name, scheme)

            # Print statistics
            stats = QueryClassifySaver.get_statistics(dataset_name, scheme)
            print(f"Classification Statistics ({scheme.upper()})")
            for qtype, count in stats["statistics"].items():
                pct = stats["distribution"].get(qtype, 0)
                print(f"  {qtype}: {count} ({pct}%)")
            print(f"  Total: {stats['total']}")

            print("Migration Complete!")
            return 0

        # Step 2: Check QuestionType.json for existing classifications
        print("\nChecking QuestionType.json for existing classifications...")
        existing = QueryClassifySaver.load_question_types(dataset_name, scheme)

        # Step 3: Load questions
        print("\nLoading questions...")
        questions = QueryClassifySaver.load_questions(dataset_name)
        total_questions = len(questions)

        # Find questions that need classification
        if existing:
            print(f"Found {len(existing)} already classified questions")
            to_classify = [q for q in questions if q["id"] not in existing]
            print(f"Remaining to classify: {len(to_classify)}")

            if not to_classify:
                print("All questions already classified!")
                stats = QueryClassifySaver.get_statistics(dataset_name, scheme)
                print("\n" + "=" * 50)
                print(f"Classification Statistics ({scheme.upper()})")
                print("=" * 50)
                for qtype, count in stats["statistics"].items():
                    pct = stats["distribution"].get(qtype, 0)
                    print(f"  {qtype}: {count} ({pct}%)")
                print(f"  Total: {stats['total']}")
                print("=" * 50)
                return 0
        else:
            to_classify = questions

        # Step 4: Initialize classifier and classify
        print("\nInitializing classifier...")
        classifier = QueryClassifier(scheme=scheme)

        print("\nClassifying questions...")
        classified = classifier.classify_batch(to_classify)

        # Step 5: Extract classifications and save
        new_classifications = {q["id"]: q["question_type"] for q in classified}

        print("\nSaving results to QuestionType.json...")
        QueryClassifySaver.save_question_types(new_classifications, dataset_name, scheme)

        # Print statistics
        stats = QueryClassifySaver.get_statistics(dataset_name, scheme)
        print("\n" + "=" * 50)
        print(f"Classification Statistics ({scheme.upper()})")
        print("=" * 50)
        for qtype, count in stats["statistics"].items():
            pct = stats["distribution"].get(qtype, 0)
            print(f"  {qtype}: {count} ({pct}%)")
        print(f"  Total: {stats['total']}")
        print("=" * 50)

        print("\n" + "=" * 60)
        print("Classification Complete!")
        print(f"Results saved to: {QueryClassifySaver.get_question_type_path(dataset_name)}")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Classify questions using different schemes",
        epilog="Example: python run_query_classify.py --dataset UltraDomain_mix --scheme memorag"
    )
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., UltraDomain_mix, UltraDomain_legal)")
    parser.add_argument("--scheme", type=str, default="memorag",
                       choices=["memorag"],
                       help="Classification scheme to use")

    args = parser.parse_args()

    exit_code = run_classification(args.dataset, args.scheme)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
