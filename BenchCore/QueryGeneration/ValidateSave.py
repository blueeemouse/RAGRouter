"""
Query Validation Saver

This module handles saving and loading validation results.
"""
import json
import os
from typing import Dict, List, Any, Optional, Set

from Config.PathConfig import PathConfig


class QueryValidateSaver:
    """Save and load validation results"""

    @staticmethod
    def get_validation_dir(dataset_name: str) -> str:
        """Get validation result directory

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to validation directory
        """
        validation_dir = PathConfig.get_query_validation_dir(dataset_name)
        os.makedirs(validation_dir, exist_ok=True)
        return validation_dir

    @staticmethod
    def get_validation_path(dataset_name: str, query_type: str) -> str:
        """Get validation result file path

        Args:
            dataset_name: Name of the dataset
            query_type: Type of queries (e.g., "single_hop", "multi_hop_2", "multi_hop_3", "summary")

        Returns:
            Path to validation file
        """
        validation_dir = QueryValidateSaver.get_validation_dir(dataset_name)
        return os.path.join(validation_dir, f"{query_type}.jsonl")

    @staticmethod
    def save_validation(
        validation_result: Dict[str, Any],
        dataset_name: str,
        query_type: str
    ) -> None:
        """Save a single validation result incrementally (append to JSONL)

        Args:
            validation_result: Validation result dict with keys:
                - id: Query ID
                - question: Original question
                - answer: Expected answer
                - validation: {answerable, shortcut, leak, overall_passed}
            dataset_name: Name of the dataset
            query_type: Type of queries
        """
        output_file = QueryValidateSaver.get_validation_path(dataset_name, query_type)

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(validation_result, ensure_ascii=False) + '\n')

    @staticmethod
    def save_validations(
        validation_results: List[Dict[str, Any]],
        dataset_name: str,
        query_type: str
    ) -> str:
        """Save all validation results of a specific type (overwrite)

        Args:
            validation_results: List of validation result dicts
            dataset_name: Name of the dataset
            query_type: Type of queries

        Returns:
            Path to saved file
        """
        output_file = QueryValidateSaver.get_validation_path(dataset_name, query_type)

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in validation_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Saved {len(validation_results)} {query_type} validation results to {output_file}")
        return output_file

    @staticmethod
    def load_validations(
        dataset_name: str,
        query_type: str
    ) -> List[Dict[str, Any]]:
        """Load validation results

        Args:
            dataset_name: Name of the dataset
            query_type: Type of queries

        Returns:
            List of validation result dicts
        """
        input_file = QueryValidateSaver.get_validation_path(dataset_name, query_type)

        if not os.path.exists(input_file):
            return []

        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        return results

    @staticmethod
    def load_existing_ids(dataset_name: str, query_type: str) -> Set[str]:
        """Load existing validation IDs for resume support

        Args:
            dataset_name: Name of the dataset
            query_type: Query type to check

        Returns:
            Set of existing query IDs that have been validated
        """
        results = QueryValidateSaver.load_validations(dataset_name, query_type)
        return {r.get("id") for r in results if r.get("id")}

    @staticmethod
    def get_passed_queries(
        dataset_name: str,
        query_type: str
    ) -> List[Dict[str, Any]]:
        """Get queries that passed validation (overall_passed=True)

        Args:
            dataset_name: Name of the dataset
            query_type: Type of queries

        Returns:
            List of query dicts (id, question, answer, supporting_facts, type, ...)
        """
        results = QueryValidateSaver.load_validations(dataset_name, query_type)

        passed = []
        for r in results:
            validation = r.get("validation", {})
            if validation.get("overall_passed", False):
                # Extract original query fields (exclude validation details)
                query = {
                    "id": r.get("id"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "supporting_facts": r.get("supporting_facts", []),
                    "type": r.get("type", query_type.split("_")[0])
                }

                # Include optional fields if present
                if "reasoning" in r:
                    query["reasoning"] = r["reasoning"]
                if "bridges" in r:
                    query["bridges"] = r["bridges"]
                if "entity" in r:
                    query["entity"] = r["entity"]
                if "num_hops" in r:
                    query["num_hops"] = r["num_hops"]

                passed.append(query)

        return passed

    @staticmethod
    def get_passed_ids(dataset_name: str, query_type: str) -> Set[str]:
        """Get IDs of queries that passed validation

        Args:
            dataset_name: Name of the dataset
            query_type: Type of queries

        Returns:
            Set of query IDs that passed validation
        """
        results = QueryValidateSaver.load_validations(dataset_name, query_type)
        return {
            r.get("id") for r in results
            if r.get("validation", {}).get("overall_passed", False) and r.get("id")
        }

    @staticmethod
    def get_statistics(dataset_name: str, query_type: str) -> Dict[str, Any]:
        """Get validation statistics

        Args:
            dataset_name: Name of the dataset
            query_type: Type of queries

        Returns:
            Statistics dict with counts and percentages
        """
        results = QueryValidateSaver.load_validations(dataset_name, query_type)

        if not results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

        total = len(results)
        passed = sum(1 for r in results if r.get("validation", {}).get("overall_passed", False))
        failed = total - passed

        # Count failures by type
        answerable_failed = 0
        shortcut_failed = 0
        leak_failed = 0

        for r in results:
            v = r.get("validation", {})
            if not v.get("overall_passed", False):
                if not v.get("answerable", {}).get("passed", True):
                    answerable_failed += 1
                if not v.get("shortcut", {}).get("passed", True):
                    shortcut_failed += 1
                if not v.get("leak", {}).get("passed", True):
                    leak_failed += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total * 100 if total > 0 else 0.0,
            "failure_breakdown": {
                "answerable": answerable_failed,
                "shortcut": shortcut_failed,
                "leak": leak_failed
            }
        }

    @staticmethod
    def clear_validations(dataset_name: str, query_type: Optional[str] = None) -> None:
        """Clear existing validation results

        Args:
            dataset_name: Name of the dataset
            query_type: Optional type to clear. If None, clears all.
        """
        validation_dir = PathConfig.get_query_validation_dir(dataset_name)

        if query_type:
            # Clear specific type
            file_path = os.path.join(validation_dir, f"{query_type}.jsonl")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleared {file_path}")
        else:
            # Clear all validation files
            if os.path.exists(validation_dir):
                for filename in os.listdir(validation_dir):
                    if filename.endswith(".jsonl"):
                        file_path = os.path.join(validation_dir, filename)
                        os.remove(file_path)
                        print(f"Cleared {file_path}")
