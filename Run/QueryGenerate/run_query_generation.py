"""
Query Generation Workflow

This script runs the complete query generation pipeline:
Step 1: Generate raw queries
Step 2: Validate queries (answerable, shortcut, leak)
Step 3: Filter and save passed queries

Usage:
    python run_query_generation.py --dataset musique --single-hop 100 --multi-hop 100 --summary 30

    # Skip generation, only validate and filter
    python run_query_generation.py --dataset musique --skip-generate

    # Skip validation, only filter
    python run_query_generation.py --dataset musique --skip-validate

    # Only filter existing validation results
    python run_query_generation.py --dataset musique --filter-only
"""
import os
import sys
import json
import asyncio
import argparse
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = ""
sys.path.insert(0, PROJECT_ROOT)

from Config.PathConfig import PathConfig
from BenchCore.QueryGeneration.Preprocess.ChainBuilder import ChainBuilder
from BenchCore.QueryGeneration.GenerateDo import QueryGenerator
from BenchCore.QueryGeneration.GenerateSave import QueryGenerateSaver
from BenchCore.QueryGeneration.ValidateDo import QueryValidator
from BenchCore.QueryGeneration.ValidateSave import QueryValidateSaver


class QueryGenerationPipeline:
    """Complete Query Generation Pipeline"""

    def __init__(self, dataset_name: str):
        """Initialize pipeline

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name

        # Ensure directories exist
        os.makedirs(PathConfig.get_query_raw_dir(dataset_name), exist_ok=True)
        os.makedirs(PathConfig.get_query_validation_dir(dataset_name), exist_ok=True)

    # Step 1: Generate

    def step1_generate(
        self,
        single_hop_count: int = 0,
        multi_hop_count: int = 0,
        summary_count: int = 0,
        max_concurrent: int = 10,
        chain_multiplier: int = 5,
        append: bool = False
    ) -> Dict[str, int]:
        """Step 1: Generate raw queries

        For multi-hop, uses generate-and-validate mode:
        - Generates query from chain
        - Immediately runs shortcut check
        - If shortcut detected, tries next chain
        - Continues until target count reached or chains exhausted

        Args:
            single_hop_count: Number of single-hop queries
            multi_hop_count: Total number of multi-hop queries (split 50/50 between 2-hop and 3-hop)
            summary_count: Number of summary queries
            max_concurrent: Max concurrent LLM calls
            chain_multiplier: Multiplier for chain pool size (e.g., 5x for retry buffer)
            append: If True, append to existing files with continued IDs

        Returns:
            Dict with generated counts per type
        """
        print("=" * 70)
        print("STEP 1: GENERATE RAW QUERIES" + (" (APPEND MODE)" if append else ""))
        print("=" * 70)

        results = {}

        # Initialize components
        chain_builder = ChainBuilder(self.dataset_name)
        generator = QueryGenerator()
        validator = QueryValidator()

        # Single-hop (no shortcut check needed)
        if single_hop_count > 0:
            # Get starting index for IDs
            start_idx = QueryGenerateSaver.get_next_index(self.dataset_name, "single_hop") if append else 0
            print(f"\n--- Generating {single_hop_count} single-hop queries (starting from ID {start_idx}) ---")

            samples = chain_builder.prepare_single_hop_samples(single_hop_count)
            queries = generator.generate_batch(samples, "single_hop", max_concurrent)

            # Assign IDs with correct starting index
            for i, q in enumerate(queries):
                q["id"] = f"single_hop_{start_idx + i:04d}"

            QueryGenerateSaver.save_queries(queries, self.dataset_name, "single_hop", append=append)
            results["single_hop"] = len(queries)

        # Multi-hop 2-hop (with shortcut validation during generation)
        multi_hop_2_count = multi_hop_count // 2
        if multi_hop_2_count > 0:
            # Get starting index for IDs
            start_idx = QueryGenerateSaver.get_next_index(self.dataset_name, "multi_hop_2") if append else 0
            print(f"\n--- Generating {multi_hop_2_count} 2-hop queries (starting from ID {start_idx}) ---")

            # Generate more chains than needed for retry
            chains = chain_builder.prepare_multihop_chains(
                multi_hop_2_count * chain_multiplier,
                num_hops=2
            )
            # Use generate-and-validate mode (async for parallelism)
            queries = asyncio.run(generator.generate_multihop_validated_async(
                chains=chains,
                target_count=multi_hop_2_count,
                validator=validator,
                num_hops=2,
                max_concurrent=max_concurrent
            ))
            # Reassign IDs with multi_hop_2 prefix and correct starting index
            for i, q in enumerate(queries):
                q["id"] = f"multi_hop_2_{start_idx + i:04d}"

            QueryGenerateSaver.save_queries(queries, self.dataset_name, "multi_hop_2", append=append)
            results["multi_hop_2"] = len(queries)

        # Multi-hop 3-hop (with shortcut validation during generation)
        multi_hop_3_count = multi_hop_count - multi_hop_2_count
        if multi_hop_3_count > 0:
            # Get starting index for IDs
            start_idx = QueryGenerateSaver.get_next_index(self.dataset_name, "multi_hop_3") if append else 0
            print(f"\n--- Generating {multi_hop_3_count} 3-hop queries (starting from ID {start_idx}) ---")

            chains = chain_builder.prepare_multihop_chains(
                multi_hop_3_count * chain_multiplier,
                num_hops=3
            )
            # Use generate-and-validate mode (async for parallelism)
            queries = asyncio.run(generator.generate_multihop_validated_async(
                chains=chains,
                target_count=multi_hop_3_count,
                validator=validator,
                num_hops=3,
                max_concurrent=max_concurrent
            ))
            # Reassign IDs with multi_hop_3 prefix and correct starting index
            for i, q in enumerate(queries):
                q["id"] = f"multi_hop_3_{start_idx + i:04d}"

            QueryGenerateSaver.save_queries(queries, self.dataset_name, "multi_hop_3", append=append)
            results["multi_hop_3"] = len(queries)

        # Summary (no shortcut check needed)
        if summary_count > 0:
            # Get starting index for IDs
            start_idx = QueryGenerateSaver.get_next_index(self.dataset_name, "summary") if append else 0
            print(f"\n--- Generating {summary_count} summary queries (starting from ID {start_idx}) ---")

            clusters = chain_builder.prepare_summary_clusters(summary_count)
            queries = generator.generate_batch(clusters, "summary", max_concurrent)

            # Assign IDs with correct starting index
            for i, q in enumerate(queries):
                q["id"] = f"summary_{start_idx + i:04d}"

            QueryGenerateSaver.save_queries(queries, self.dataset_name, "summary", append=append)
            results["summary"] = len(queries)

        print(f"\n[Step 1 Complete] Generated: {results}")
        return results

    # Step 2: Validate

    def step2_validate(
        self,
        query_types: List[str] = None,
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Step 2: Validate queries

        Runs validation checks:
        - single_hop/summary: answerable + leak
        - multi_hop: answerable + shortcut + leak

        Args:
            query_types: List of query types to validate. If None, validates all.
            max_concurrent: Max concurrent LLM calls

        Returns:
            Dict with validation statistics per type
        """
        print("\n" + "=" * 70)
        print("STEP 2: VALIDATE QUERIES")
        print("=" * 70)

        if query_types is None:
            query_types = ["single_hop", "multi_hop_2", "multi_hop_3", "summary"]

        validator = QueryValidator()
        results = {}

        for query_type in query_types:
            # Load raw queries
            queries = QueryGenerateSaver.load_queries(self.dataset_name, query_type)
            if not queries:
                print(f"\n[{query_type}] No queries found, skipping")
                continue

            print(f"\n--- Validating {len(queries)} {query_type} queries ---")

            # Load existing validation IDs for resume
            existing_ids = QueryValidateSaver.load_existing_ids(self.dataset_name, query_type)
            to_validate = [q for q in queries if q.get("id") not in existing_ids]

            if not to_validate:
                print(f"All {len(queries)} queries already validated")
                stats = QueryValidateSaver.get_statistics(self.dataset_name, query_type)
                results[query_type] = stats
                continue

            print(f"Validating {len(to_validate)} new queries ({len(existing_ids)} already done)")

            # Determine which checks to run
            is_multi_hop = "multi_hop" in query_type

            # Run validation
            validation_results = asyncio.run(
                self._validate_batch_async(
                    to_validate,
                    validator,
                    is_multi_hop,
                    max_concurrent
                )
            )

            # Save validation results incrementally
            for vr in validation_results:
                QueryValidateSaver.save_validation(vr, self.dataset_name, query_type)

            # Get statistics
            stats = QueryValidateSaver.get_statistics(self.dataset_name, query_type)
            results[query_type] = stats

            print(f"[{query_type}] Passed: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)")

        print(f"\n[Step 2 Complete] Validation stats: {results}")
        return results

    async def _validate_batch_async(
        self,
        queries: List[Dict[str, Any]],
        validator: QueryValidator,
        is_multi_hop: bool,
        max_concurrent: int
    ) -> List[Dict[str, Any]]:
        """Validate queries asynchronously

        Args:
            queries: List of query dicts
            validator: QueryValidator instance
            is_multi_hop: Whether to run shortcut check
            max_concurrent: Max concurrent calls

        Returns:
            List of validation result dicts
        """
        from tqdm.asyncio import tqdm as atqdm

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def validate_one(query: Dict) -> Dict:
            async with semaphore:
                # Run validation checks
                answerable = await validator._check_answerable_async(query)
                leak = await validator._check_leak_async(query)

                if is_multi_hop:
                    shortcut = await validator._check_shortcut_async(query)
                else:
                    shortcut = {"passed": True, "shortcut_type": "not_applicable"}

                # Determine overall pass
                overall_passed = (
                    answerable.get("passed", False) and
                    shortcut.get("passed", True) and
                    leak.get("passed", False)
                )

                return {
                    "id": query.get("id"),
                    "question": query.get("question"),
                    "answer": query.get("answer"),
                    "supporting_facts": query.get("supporting_facts", []),
                    "type": query.get("type"),
                    "reasoning": query.get("reasoning"),
                    "bridges": query.get("bridges"),
                    "entity": query.get("entity"),
                    "num_hops": query.get("num_hops"),
                    "validation": {
                        "answerable": answerable,
                        "shortcut": shortcut,
                        "leak": leak,
                        "overall_passed": overall_passed
                    }
                }

        tasks = [validate_one(q) for q in queries]

        async for result in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Validating",
            unit="q"
        ):
            vr = await result
            results.append(vr)

        return results


    # Step 3: Filter

    def step3_filter(
        self,
        query_types: List[str] = None
    ) -> Dict[str, int]:
        """Step 3: Filter validated queries and save final output

        Workflow:
        1. Load all validation files and combine them
        2. Filter out entries where overall_passed is false
        3. Keep only specific keys (id, question, answer, supporting_facts, type)
        4. Save as JSONL format (one JSON per line)

        Args:
            query_types: List of query types to filter. If None, filters all.

        Returns:
            Dict with filtered counts per type
        """
        print("\n" + "=" * 70)
        print("STEP 3: FILTER AND SAVE FINAL QUERIES")
        print("=" * 70)

        if query_types is None:
            query_types = ["single_hop", "multi_hop_2", "multi_hop_3", "summary"]

        all_passed = []
        results = {}

        # Keys to keep in final output
        keys_to_keep = ["id", "question", "answer", "supporting_facts", "type"]

        for query_type in query_types:
            # Load validation results
            validations = QueryValidateSaver.load_validations(self.dataset_name, query_type)

            if not validations:
                print(f"[{query_type}] No validation results")
                results[query_type] = 0
                continue

            # Filter: keep only overall_passed=True
            passed = [
                v for v in validations
                if v.get("validation", {}).get("overall_passed", False)
            ]
            results[query_type] = len(passed)

            # Extract only required keys and normalize type
            for v in passed:
                cleaned = {k: v[k] for k in keys_to_keep if k in v}
                # Normalize type field for multi-hop
                if "multi_hop" in query_type:
                    cleaned["type"] = "multi_hop"
                all_passed.append(cleaned)

            print(f"[{query_type}] Passed: {len(passed)}")

        # Sort by type and ID
        all_passed.sort(key=lambda x: (x.get("type", ""), x.get("id", "")))

        # Save as JSONL format
        output_path = PathConfig.get_query_final_path(self.dataset_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for q in all_passed:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

        print(f"\n[Step 3 Complete] Saved {len(all_passed)} queries to {output_path}")
        print(f"Breakdown: {results}")
        return results


    # Full Pipeline

    def run(
        self,
        single_hop_count: int = 0,
        multi_hop_count: int = 0,
        summary_count: int = 0,
        skip_generate: bool = False,
        skip_validate: bool = False,
        filter_only: bool = False,
        max_concurrent: int = 10,
        append: bool = False
    ) -> Dict[str, Any]:
        """Run full pipeline

        Args:
            single_hop_count: Number of single-hop queries
            multi_hop_count: Total number of multi-hop queries
            summary_count: Number of summary queries
            skip_generate: Skip Step 1
            skip_validate: Skip Step 2
            filter_only: Only run Step 3
            max_concurrent: Max concurrent LLM calls
            append: If True, append to existing files with continued IDs

        Returns:
            Pipeline results
        """
        print("=" * 70)
        print("QUERY GENERATION PIPELINE")
        print(f"Dataset: {self.dataset_name}")
        if append:
            print("Mode: APPEND (continuing from existing IDs)")
        print("=" * 70)

        results = {}

        # Step 1: Generate
        if not skip_generate and not filter_only:
            results["step1"] = self.step1_generate(
                single_hop_count,
                multi_hop_count,
                summary_count,
                max_concurrent,
                append=append
            )

        # Step 2: Validate
        if not skip_validate and not filter_only:
            results["step2"] = self.step2_validate(max_concurrent=max_concurrent)

        # Step 3: Filter
        results["step3"] = self.step3_filter()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Query Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all types
  python run_query_generation.py --dataset musique --single-hop 100 --multi-hop 100 --summary 30

  # Only generate multi-hop queries
  python run_query_generation.py --dataset musique --multi-hop 200

  # Skip generation, only validate and filter
  python run_query_generation.py --dataset musique --skip-generate

  # Only filter existing validation results
  python run_query_generation.py --dataset musique --filter-only

  # Append more queries (continue from existing IDs)
  python run_query_generation.py --dataset musique --single-hop 50 --append
        """
    )

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--single-hop", type=int, default=0, help="Number of single-hop queries")
    parser.add_argument("--multi-hop", type=int, default=0, help="Total multi-hop queries (split 50/50 between 2-hop and 3-hop)")
    parser.add_argument("--summary", type=int, default=0, help="Number of summary queries")
    parser.add_argument("--skip-generate", action="store_true", help="Skip Step 1 (generation)")
    parser.add_argument("--skip-validate", action="store_true", help="Skip Step 2 (validation)")
    parser.add_argument("--filter-only", action="store_true", help="Only run Step 3 (filter)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls")
    parser.add_argument("--append", action="store_true", help="Append to existing files with continued IDs (incremental mode)")

    args = parser.parse_args()

    # Validate arguments
    if not args.skip_generate and not args.filter_only:
        if args.single_hop == 0 and args.multi_hop == 0 and args.summary == 0:
            print("Warning: No query counts specified. Use --single-hop, --multi-hop, or --summary")
            print("Running filter-only mode...")
            args.filter_only = True

    # Run pipeline
    pipeline = QueryGenerationPipeline(args.dataset)
    results = pipeline.run(
        single_hop_count=args.single_hop,
        multi_hop_count=args.multi_hop,
        summary_count=args.summary,
        skip_generate=args.skip_generate,
        skip_validate=args.skip_validate,
        filter_only=args.filter_only,
        max_concurrent=args.max_concurrent,
        append=args.append
    )

    # Print final summary
    print("\nFinal Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
