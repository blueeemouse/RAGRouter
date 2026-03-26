"""
Query Generation Saver

This module handles saving and loading generated queries.
"""
import json
import os
from typing import Dict, List, Any, Optional

from Config.PathConfig import PathConfig


class QueryGenerateSaver:
    """Save and load generated queries"""

    @staticmethod
    def get_output_dir(dataset_name: str) -> str:
        """Get raw query output directory

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to raw output directory
        """
        output_dir = PathConfig.get_query_raw_dir(dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def save_query(query: Dict[str, Any], dataset_name: str) -> None:
        """Save a single query incrementally (append to JSONL)

        Args:
            query: Query dict to save
            dataset_name: Name of the dataset
        """
        output_dir = QueryGenerateSaver.get_output_dir(dataset_name)
        query_type = query.get("type", "unknown")
        output_file = os.path.join(output_dir, f"{query_type}.jsonl")

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    @staticmethod
    def save_queries(
        queries: List[Dict[str, Any]],
        dataset_name: str,
        query_type: str,
        append: bool = False
    ) -> str:
        """Save all queries of a specific type

        Args:
            queries: List of query dicts
            dataset_name: Name of the dataset
            query_type: Type of queries ("single_hop", "multi_hop", "summary")
            append: If True, append to existing file; if False, overwrite

        Returns:
            Path to saved file
        """
        output_dir = QueryGenerateSaver.get_output_dir(dataset_name)
        output_file = os.path.join(output_dir, f"{query_type}.jsonl")

        mode = 'a' if append else 'w'
        with open(output_file, mode, encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')

        action = "Appended" if append else "Saved"
        print(f"{action} {len(queries)} {query_type} queries to {output_file}")
        return output_file

    @staticmethod
    def save_all(queries: List[Dict[str, Any]], dataset_name: str) -> str:
        """Save all queries to a single combined file (final output)

        Args:
            queries: List of all query dicts (mixed types)
            dataset_name: Name of the dataset

        Returns:
            Path to saved file
        """
        # Save to the parent directory (not in raw/)
        output_file = PathConfig.get_query_final_path(dataset_name)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Sort by type and ID for consistent ordering
        queries_sorted = sorted(queries, key=lambda x: (x.get("type", ""), x.get("id", "")))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(queries_sorted, f, ensure_ascii=False, indent=2)

        # Print summary
        type_counts = {}
        for q in queries:
            t = q.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"Saved {len(queries)} queries to {output_file}")
        for t, count in sorted(type_counts.items()):
            print(f"  - {t}: {count}")

        return output_file

    @staticmethod
    def load_queries(dataset_name: str, query_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load generated queries

        Args:
            dataset_name: Name of the dataset
            query_type: Optional type filter ("single_hop", "multi_hop", "summary")
                       If None, loads from combined questions.json

        Returns:
            List of query dicts
        """
        output_dir = QueryGenerateSaver.get_output_dir(dataset_name)

        if query_type:
            # Load specific type from JSONL
            input_file = os.path.join(output_dir, f"{query_type}.jsonl")
            if not os.path.exists(input_file):
                return []

            queries = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))
            return queries

        else:
            # Load from combined JSON file
            input_file = os.path.join(output_dir, "questions.json")
            if not os.path.exists(input_file):
                return []

            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    @staticmethod
    def load_existing_ids(dataset_name: str, query_type: str) -> set:
        """Load existing query IDs for resume support

        Args:
            dataset_name: Name of the dataset
            query_type: Query type to check

        Returns:
            Set of existing query IDs
        """
        queries = QueryGenerateSaver.load_queries(dataset_name, query_type)
        return {q.get("id") for q in queries if q.get("id")}

    @staticmethod
    def get_next_index(dataset_name: str, query_type: str) -> int:
        """Get next available index for a query type

        Args:
            dataset_name: Name of the dataset
            query_type: Query type

        Returns:
            Next available index
        """
        existing_ids = QueryGenerateSaver.load_existing_ids(dataset_name, query_type)

        if not existing_ids:
            return 0

        # Extract indices from IDs like "single_hop_0001"
        max_index = 0
        for id_str in existing_ids:
            try:
                parts = id_str.rsplit("_", 1)
                if len(parts) == 2:
                    idx = int(parts[1])
                    max_index = max(max_index, idx)
            except ValueError:
                continue

        return max_index + 1

    @staticmethod
    def clear_queries(dataset_name: str, query_type: Optional[str] = None) -> None:
        """Clear existing queries

        Args:
            dataset_name: Name of the dataset
            query_type: Optional type to clear. If None, clears all.
        """
        output_dir = QueryGenerateSaver.get_output_dir(dataset_name)

        if query_type:
            # Clear specific type
            file_path = os.path.join(output_dir, f"{query_type}.jsonl")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleared {file_path}")
        else:
            # Clear all files in raw directory
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if filename.endswith(".jsonl"):
                        file_path = os.path.join(output_dir, filename)
                        os.remove(file_path)
                        print(f"Cleared {file_path}")

            # Also clear final questions.json
            final_file = PathConfig.get_query_final_path(dataset_name)
            if os.path.exists(final_file):
                os.remove(final_file)
                print(f"Cleared {final_file}")
