"""
Query Validation Processor

This module validates generated queries with three checks:
1. Answerable: Can the query be answered with supporting docs?
2. Shortcut: Does multi-hop query require all docs? (no single-doc shortcut)
3. Leak: Is the answer leaked in question or known by LLM?
"""
import json
import asyncio
import re
from typing import Dict, List, Any, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from Config.LLMConfig import LLMConfig
from Config.RetrieverConfig import RetrieverConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


class QueryValidator:
    """Validate generated queries for quality assurance"""

    def __init__(self):
        """Initialize Query Validator with LLM clients"""
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup sync and async LLM clients"""
        model_config = LLMConfig.get_model_config()

        self.provider = LLMConfig.PROVIDER
        self.model_name = model_config["model"]

        self.llm_client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        self.async_llm_client = AsyncOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        print(f"Initialized Validator LLM: Provider={self.provider}, Model={self.model_name}")


    # 1. Answerable Check
    def check_answerable(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Check if query can be answered with supporting documents

        Uses RAG_SYSTEM with all supporting_facts as context.
        Uses LLM-as-judge to compare generated answer with expected answer.

        Args:
            query: Query dict with keys: question, answer, supporting_facts

        Returns:
            {passed: bool, generated_answer: str, reason: str}
        """
        question = query.get("question", "")
        expected_answer = query.get("answer", "")
        supporting_facts = query.get("supporting_facts", [])

        # Build context from supporting facts
        context = self._build_context(supporting_facts)

        if not context:
            return {
                "passed": False,
                "generated_answer": "",
                "reason": "No supporting facts provided"
            }

        # Generate answer with RAG
        generated_answer = self._generate_with_context(question, context)

        if generated_answer is None:
            return {
                "passed": False,
                "generated_answer": "",
                "reason": "LLM generation failed"
            }

        # Check if answer matches using LLM-as-judge
        passed = self._check_answer_match(question, expected_answer, generated_answer)

        return {
            "passed": passed,
            "generated_answer": generated_answer,
            "reason": "Answer matches" if passed else "Answer does not match"
        }

    async def _check_answerable_async(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of answerable check"""
        question = query.get("question", "")
        expected_answer = query.get("answer", "")
        supporting_facts = query.get("supporting_facts", [])

        context = self._build_context(supporting_facts)

        if not context:
            return {
                "passed": False,
                "generated_answer": "",
                "reason": "No supporting facts provided"
            }

        generated_answer = await self._generate_with_context_async(question, context)

        if generated_answer is None:
            return {
                "passed": False,
                "generated_answer": "",
                "reason": "LLM generation failed"
            }

        passed = await self._check_answer_match_async(question, expected_answer, generated_answer)

        return {
            "passed": passed,
            "generated_answer": generated_answer,
            "reason": "Answer matches" if passed else "Answer does not match"
        }


    # 2. Shortcut Detection (Multi-hop only)
    def check_shortcut(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Check if multi-hop query has shortcuts

        Tests if any single document can answer the question alone.
        Uses LLM-as-judge for answer matching.

        Args:
            query: Query dict (must be multi_hop type)

        Returns:
            {passed: bool, single_doc_tests: [...], shortcut_type: str}
        """
        query_type = query.get("type", "")

        # Only check multi-hop queries
        if query_type != "multi_hop":
            return {
                "passed": True,
                "single_doc_tests": [],
                "shortcut_type": "not_applicable",
                "reason": "Only multi-hop queries need shortcut check"
            }

        question = query.get("question", "")
        expected_answer = query.get("answer", "")
        supporting_facts = query.get("supporting_facts", [])

        single_doc_tests = []
        has_single_doc_shortcut = False

        # Test: Single doc tests
        for i, fact in enumerate(supporting_facts):
            single_context = self._build_context([fact])
            generated = self._generate_with_context(question, single_context)

            if generated:
                can_answer = self._check_answer_match(question, expected_answer, generated)

                single_doc_tests.append({
                    "doc_id": fact.get("doc_id"),
                    "doc_index": i,
                    "can_answer": can_answer,
                    "generated_answer": generated
                })

                if can_answer:
                    has_single_doc_shortcut = True

        # Determine shortcut type
        shortcut_type = "single_doc" if has_single_doc_shortcut else "none"
        passed = not has_single_doc_shortcut

        return {
            "passed": passed,
            "single_doc_tests": single_doc_tests,
            "shortcut_type": shortcut_type,
            "reason": f"Shortcut detected: {shortcut_type}" if not passed else "No shortcuts"
        }

    async def _check_shortcut_async(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of shortcut check"""
        query_type = query.get("type", "")

        if query_type != "multi_hop":
            return {
                "passed": True,
                "single_doc_tests": [],
                "shortcut_type": "not_applicable",
                "reason": "Only multi-hop queries need shortcut check"
            }

        question = query.get("question", "")
        expected_answer = query.get("answer", "")
        supporting_facts = query.get("supporting_facts", [])

        # Generate answers for all single docs concurrently
        async def test_single_doc(i, fact):
            single_context = self._build_context([fact])
            generated = await self._generate_with_context_async(question, single_context)
            if generated:
                can_answer = await self._check_answer_match_async(question, expected_answer, generated)
                return {
                    "doc_id": fact.get("doc_id"),
                    "doc_index": i,
                    "can_answer": can_answer,
                    "generated_answer": generated
                }
            return None

        tasks = [test_single_doc(i, fact) for i, fact in enumerate(supporting_facts)]
        results = await asyncio.gather(*tasks)

        single_doc_tests = [r for r in results if r is not None]
        has_single_doc_shortcut = any(r["can_answer"] for r in single_doc_tests)

        shortcut_type = "single_doc" if has_single_doc_shortcut else "none"
        passed = not has_single_doc_shortcut

        return {
            "passed": passed,
            "single_doc_tests": single_doc_tests,
            "shortcut_type": shortcut_type,
            "reason": f"Shortcut detected: {shortcut_type}" if not passed else "No shortcuts"
        }


    # 3. Leak Detection
    def check_leak(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Check if answer is leaked in question or known by LLM

        Tests:
        A) Exact match: Is answer substring of question?
        B) LLM knowledge: Can LLM answer without context? (uses LLM-as-judge)

        Args:
            query: Query dict with keys: question, answer

        Returns:
            {passed: bool, exact_match: bool, llm_knows: bool, llm_answer: str, leak_type: str}
        """
        question = query.get("question", "")
        expected_answer = query.get("answer", "")

        # Test A: Exact match (case-insensitive)
        exact_match = expected_answer.lower().strip() in question.lower()

        # Test B: LLM knowledge (use QA_SYSTEM without context)
        llm_answer = self._generate_direct(question)
        llm_knows = False

        if llm_answer:
            # Check if LLM's answer matches expected using LLM-as-judge
            llm_knows = self._check_answer_match(question, expected_answer, llm_answer)

        # Determine leak type
        if exact_match and llm_knows:
            leak_type = "both"
        elif exact_match:
            leak_type = "exact_match"
        elif llm_knows:
            leak_type = "llm_knowledge"
        else:
            leak_type = "none"

        passed = leak_type == "none"

        return {
            "passed": passed,
            "exact_match": exact_match,
            "llm_knows": llm_knows,
            "llm_answer": llm_answer or "",
            "leak_type": leak_type,
            "reason": f"Leak detected: {leak_type}" if not passed else "No leaks"
        }

    async def _check_leak_async(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of leak check"""
        question = query.get("question", "")
        expected_answer = query.get("answer", "")

        # Test A: Exact match
        exact_match = expected_answer.lower().strip() in question.lower()

        # Test B: LLM knowledge
        llm_answer = await self._generate_direct_async(question)
        llm_knows = False

        if llm_answer:
            llm_knows = await self._check_answer_match_async(question, expected_answer, llm_answer)

        if exact_match and llm_knows:
            leak_type = "both"
        elif exact_match:
            leak_type = "exact_match"
        elif llm_knows:
            leak_type = "llm_knowledge"
        else:
            leak_type = "none"

        passed = leak_type == "none"

        return {
            "passed": passed,
            "exact_match": exact_match,
            "llm_knows": llm_knows,
            "llm_answer": llm_answer or "",
            "leak_type": leak_type,
            "reason": f"Leak detected: {leak_type}" if not passed else "No leaks"
        }


    # Combined Validation
    def validate(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Run all validation checks on a query

        Args:
            query: Query dict

        Returns:
            {query_id, answerable, shortcut, leak, overall_passed}
        """
        query_id = query.get("id", "unknown")
        query_type = query.get("type", "unknown")

        # Run all checks
        answerable = self.check_answerable(query)
        shortcut = self.check_shortcut(query)
        leak = self.check_leak(query)

        # Overall pass requires all checks to pass
        overall_passed = answerable["passed"] and shortcut["passed"] and leak["passed"]

        return {
            "query_id": query_id,
            "query_type": query_type,
            "answerable": answerable,
            "shortcut": shortcut,
            "leak": leak,
            "overall_passed": overall_passed
        }

    async def _validate_async(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of validate"""
        query_id = query.get("id", "unknown")
        query_type = query.get("type", "unknown")

        # Run all checks concurrently
        answerable, shortcut, leak = await asyncio.gather(
            self._check_answerable_async(query),
            self._check_shortcut_async(query),
            self._check_leak_async(query)
        )

        overall_passed = answerable["passed"] and shortcut["passed"] and leak["passed"]

        return {
            "query_id": query_id,
            "query_type": query_type,
            "answerable": answerable,
            "shortcut": shortcut,
            "leak": leak,
            "overall_passed": overall_passed
        }


    # Batch Validation
    async def validate_batch_async(
        self,
        queries: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Validate queries in batch with concurrency control

        Args:
            queries: List of query dicts
            max_concurrent: Maximum concurrent validations

        Returns:
            List of validation result dicts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def bounded_validate(query: Dict):
            async with semaphore:
                return await self._validate_async(query)

        tasks = [bounded_validate(q) for q in queries]

        async for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Validating", unit="q"):
            validation = await result
            results.append(validation)

        # Sort by query_id
        results.sort(key=lambda x: x["query_id"])

        return results

    def validate_batch(
        self,
        queries: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Sync wrapper for batch validation

        Args:
            queries: List of query dicts
            max_concurrent: Maximum concurrent validations

        Returns:
            List of validation result dicts
        """
        return asyncio.run(self.validate_batch_async(queries, max_concurrent))


    # Helper Methods
    def _build_context(self, supporting_facts: List[Dict[str, Any]]) -> str:
        """Build context string from supporting facts

        Args:
            supporting_facts: List of {doc_id, text} dicts

        Returns:
            Context string
        """
        texts = [fact.get("text", "") for fact in supporting_facts if fact.get("text")]
        return "\n\n".join(texts)

    def _generate_with_context(self, question: str, context: str) -> Optional[str]:
        """Generate answer with context using RAG_SYSTEM

        Uses temperature=0 for deterministic evaluation.
        """
        try:
            messages = PromptTemplate.get_rag_messages(context, question)

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating with context: {e}")
            return None

    async def _generate_with_context_async(self, question: str, context: str) -> Optional[str]:
        """Async version of generate with context"""
        try:
            messages = PromptTemplate.get_rag_messages(context, question)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating with context (async): {e}")
            return None

    def _generate_direct(self, question: str) -> Optional[str]:
        """Generate answer without context using QA_SYSTEM

        Uses temperature=0 for deterministic evaluation.
        """
        try:
            messages = PromptTemplate.get_qa_messages(question)

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating direct: {e}")
            return None

    async def _generate_direct_async(self, question: str) -> Optional[str]:
        """Async version of generate direct"""
        try:
            messages = PromptTemplate.get_qa_messages(question)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=RetrieverConfig.RAG_MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating direct (async): {e}")
            return None

    def _check_answer_match(self, question: str, expected: str, generated: str) -> bool:
        """Check if generated answer matches expected using LLM-as-judge

        Uses ANSWER_LABEL_SYSTEM prompt to determine semantic equivalence.

        Args:
            question: The question being answered
            expected: Expected/gold answer
            generated: Generated answer

        Returns:
            True if answers match (label == "correct"), False otherwise
        """
        if not expected or not generated:
            return False

        # Quick check for refusal
        generated_lower = generated.lower().strip()
        if "cannot answer" in generated_lower or "i don't know" in generated_lower:
            return False

        try:
            messages = PromptTemplate.get_answer_label_messages(question, expected, generated)

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=100,
                timeout=LLMConfig.TIMEOUT
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("label") == "correct"

            return False

        except Exception as e:
            print(f"Error in LLM answer match: {e}")
            return False

    async def _check_answer_match_async(self, question: str, expected: str, generated: str) -> bool:
        """Async version of LLM-based answer matching"""
        if not expected or not generated:
            return False

        generated_lower = generated.lower().strip()
        if "cannot answer" in generated_lower or "i don't know" in generated_lower:
            return False

        try:
            messages = PromptTemplate.get_answer_label_messages(question, expected, generated)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=100,
                timeout=LLMConfig.TIMEOUT
            )

            result_text = response.choices[0].message.content.strip()

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("label") == "correct"

            return False

        except Exception as e:
            print(f"Error in LLM answer match (async): {e}")
            return False

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics from validation results

        Args:
            results: List of validation result dicts

        Returns:
            Statistics dict
        """
        total = len(results)
        if total == 0:
            return {"total": 0}

        # Count passes
        overall_passed = sum(1 for r in results if r.get("overall_passed", False))
        answerable_passed = sum(1 for r in results if r.get("answerable", {}).get("passed", False))
        shortcut_passed = sum(1 for r in results if r.get("shortcut", {}).get("passed", False))
        leak_passed = sum(1 for r in results if r.get("leak", {}).get("passed", False))

        # Count by type
        type_counts = {}
        type_passed = {}
        for r in results:
            qtype = r.get("query_type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            if r.get("overall_passed", False):
                type_passed[qtype] = type_passed.get(qtype, 0) + 1

        # Shortcut breakdown (multi_hop only)
        multihop_results = [r for r in results if r.get("query_type") == "multi_hop"]
        shortcut_types = {}
        for r in multihop_results:
            stype = r.get("shortcut", {}).get("shortcut_type", "unknown")
            shortcut_types[stype] = shortcut_types.get(stype, 0) + 1

        # Leak breakdown
        leak_types = {}
        for r in results:
            ltype = r.get("leak", {}).get("leak_type", "unknown")
            leak_types[ltype] = leak_types.get(ltype, 0) + 1

        return {
            "total": total,
            "overall_passed": overall_passed,
            "overall_pass_rate": overall_passed / total,
            "answerable_passed": answerable_passed,
            "answerable_pass_rate": answerable_passed / total,
            "shortcut_passed": shortcut_passed,
            "shortcut_pass_rate": shortcut_passed / total,
            "leak_passed": leak_passed,
            "leak_pass_rate": leak_passed / total,
            "by_type": {
                qtype: {
                    "total": type_counts.get(qtype, 0),
                    "passed": type_passed.get(qtype, 0),
                    "pass_rate": type_passed.get(qtype, 0) / type_counts.get(qtype, 1)
                }
                for qtype in type_counts
            },
            "shortcut_breakdown": shortcut_types,
            "leak_breakdown": leak_types
        }


# Usage example
if __name__ == "__main__":
    import json

    validator = QueryValidator()

    # Example query
    query = {
        "id": "multi_hop_0001",
        "question": "What is the capital of the state where Jurassic Park was filmed?",
        "answer": "Honolulu",
        "supporting_facts": [
            {"doc_id": 1, "text": "Jurassic Park was filmed on location in Hawaii."},
            {"doc_id": 2, "text": "Hawaii is a U.S. state. The capital of Hawaii is Honolulu."}
        ],
        "type": "multi_hop"
    }

    # Run validation
    result = validator.validate(query)
    print(json.dumps(result, indent=2))