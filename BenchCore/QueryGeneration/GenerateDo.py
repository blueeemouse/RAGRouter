"""
Query Generation Processor

This module generates three types of queries for RAG benchmarking:
- Single-hop (Factual): Simple factual questions from a single passage
- Multi-hop (Complex Reasoning): Questions requiring multiple documents
- Summary: Synthesis questions about an entity across multiple documents
"""
import json
import asyncio
from typing import Dict, List, Any, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from Config.LLMConfig import LLMConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


class QueryGenerator:
    """Generate queries for RAG benchmarking"""

    def __init__(self):
        """Initialize Query Generator with LLM clients"""
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup sync and async LLM clients from LLMConfig"""
        model_config = LLMConfig.get_model_config()

        self.provider = LLMConfig.PROVIDER
        self.model_name = model_config["model"]

        # Sync client for single queries
        self.llm_client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        # Async client for batch processing
        self.async_llm_client = AsyncOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        print(f"Initialized LLM: Provider={self.provider}, Model={self.model_name}")


    # Single-hop Query Generation
    def generate_single_hop(self, passage: str, doc_id: int, index: int) -> Optional[Dict[str, Any]]:
        """Generate a single-hop factual query from a passage

        Args:
            passage: Source passage text
            doc_id: Document ID
            index: Query index for ID generation

        Returns:
            Query dict with keys: id, question, answer, supporting_facts, type
            Returns None if generation fails
        """
        try:
            messages = PromptTemplate.get_single_hop_gen_messages(passage)

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)

            if not result or "question" not in result or "answer" not in result:
                return None

            return {
                "id": f"single_hop_{index:04d}",
                "question": result["question"],
                "answer": result["answer"],
                "supporting_facts": [{"doc_id": doc_id, "text": passage}],
                "type": "single_hop"
            }

        except Exception as e:
            print(f"Error generating single-hop query: {e}")
            return None

    async def _generate_single_hop_async(
        self,
        passage: str,
        doc_id: int,
        index: int,
        attempt: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Async version of single-hop query generation with retry"""
        try:
            messages = PromptTemplate.get_single_hop_gen_messages(passage)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)

            if not result or "question" not in result or "answer" not in result:
                if attempt < LLMConfig.MAX_RETRIES:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    return await self._generate_single_hop_async(passage, doc_id, index, attempt + 1)
                return None

            return {
                "id": f"single_hop_{index:04d}",
                "question": result["question"],
                "answer": result["answer"],
                "supporting_facts": [{"doc_id": doc_id, "text": passage}],
                "type": "single_hop"
            }

        except Exception as e:
            if attempt < LLMConfig.MAX_RETRIES:
                await asyncio.sleep(LLMConfig.RETRY_DELAY ** (attempt + 1))
                return await self._generate_single_hop_async(passage, doc_id, index, attempt + 1)
            print(f"Error generating single-hop query after {LLMConfig.MAX_RETRIES} retries: {e}")
            return None


    # Multi-hop Query Generation
    def generate_multihop(
        self,
        documents: List[str],
        bridges: List[str],
        doc_ids: List[int],
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a multi-hop reasoning query from a document chain

        Args:
            documents: List of document texts [doc1, doc2, ..., docN]
            bridges: List of bridge entities [bridge_1, bridge_2, ...] (len = len(docs) - 1)
            doc_ids: List of document IDs corresponding to documents
            index: Query index for ID generation

        Returns:
            Query dict with keys: id, question, answer, supporting_facts, type, reasoning
            Returns None if generation fails
        """
        try:
            messages = PromptTemplate.get_multihop_nhop_gen_messages(documents, bridges)

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)

            if not result or "question" not in result or "answer" not in result:
                return None

            # Build supporting_facts from documents
            supporting_facts = [
                {"doc_id": doc_id, "text": doc_text}
                for doc_id, doc_text in zip(doc_ids, documents)
            ]

            return {
                "id": f"multi_hop_{index:04d}",
                "question": result["question"],
                "answer": result["answer"],
                "supporting_facts": supporting_facts,
                "type": "multi_hop",
                "reasoning": result.get("reasoning", ""),
                "bridges": bridges  # Store bridge entities for reference
            }

        except Exception as e:
            print(f"Error generating multi-hop query: {e}")
            return None

    async def _generate_multihop_async(
        self,
        documents: List[str],
        bridges: List[str],
        doc_ids: List[int],
        index: int,
        attempt: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Async version of multi-hop query generation with retry"""
        try:
            messages = PromptTemplate.get_multihop_nhop_gen_messages(documents, bridges)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)

            if not result or "question" not in result or "answer" not in result:
                if attempt < LLMConfig.MAX_RETRIES:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    return await self._generate_multihop_async(documents, bridges, doc_ids, index, attempt + 1)
                return None

            supporting_facts = [
                {"doc_id": doc_id, "text": doc_text}
                for doc_id, doc_text in zip(doc_ids, documents)
            ]

            return {
                "id": f"multi_hop_{index:04d}",
                "question": result["question"],
                "answer": result["answer"],
                "supporting_facts": supporting_facts,
                "type": "multi_hop",
                "reasoning": result.get("reasoning", ""),
                "bridges": bridges
            }

        except Exception as e:
            if attempt < LLMConfig.MAX_RETRIES:
                await asyncio.sleep(LLMConfig.RETRY_DELAY ** (attempt + 1))
                return await self._generate_multihop_async(documents, bridges, doc_ids, index, attempt + 1)
            print(f"Error generating multi-hop query after {LLMConfig.MAX_RETRIES} retries: {e}")
            return None


    # Summary Query Generation
    def generate_summary(
        self,
        entity: str,
        documents: List[str],
        doc_ids: List[int],
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a summary/synthesis query from multiple documents about an entity

        Args:
            entity: Target entity for summarization
            documents: List of documents containing the entity
            doc_ids: List of document IDs corresponding to documents
            index: Query index for ID generation

        Returns:
            Query dict with keys: id, question, answer, supporting_facts, type, reasoning
            Returns None if generation fails or LLM returns {"discard": true}
        """
        try:
            messages = PromptTemplate.get_summary_gen_messages(entity, documents)

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)

            if not result:
                return None

            # Check for discard signal (entity inconsistency)
            if result.get("discard", False):
                print(f"Discarded summary for entity '{entity}': documents refer to different entities")
                return None

            if "question" not in result or "answer" not in result:
                return None

            # Build supporting_facts from documents
            supporting_facts = [
                {"doc_id": doc_id, "text": doc_text}
                for doc_id, doc_text in zip(doc_ids, documents)
            ]

            return {
                "id": f"summary_{index:04d}",
                "question": result["question"],
                "answer": result["answer"],
                "supporting_facts": supporting_facts,
                "type": "summary",
                "reasoning": result.get("reasoning", ""),
                "entity": entity  # Store target entity for reference
            }

        except Exception as e:
            print(f"Error generating summary query: {e}")
            return None

    async def _generate_summary_async(
        self,
        entity: str,
        documents: List[str],
        doc_ids: List[int],
        index: int,
        attempt: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Async version of summary query generation with retry"""
        try:
            messages = PromptTemplate.get_summary_gen_messages(entity, documents)

            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=LLMConfig.TEMPERATURE,
                max_tokens=LLMConfig.MAX_TOKENS,
                timeout=LLMConfig.TIMEOUT
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)

            if not result:
                if attempt < LLMConfig.MAX_RETRIES:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    return await self._generate_summary_async(entity, documents, doc_ids, index, attempt + 1)
                return None

            # Check for discard signal
            if result.get("discard", False):
                return None

            if "question" not in result or "answer" not in result:
                if attempt < LLMConfig.MAX_RETRIES:
                    await asyncio.sleep(LLMConfig.RETRY_DELAY)
                    return await self._generate_summary_async(entity, documents, doc_ids, index, attempt + 1)
                return None

            supporting_facts = [
                {"doc_id": doc_id, "text": doc_text}
                for doc_id, doc_text in zip(doc_ids, documents)
            ]

            return {
                "id": f"summary_{index:04d}",
                "question": result["question"],
                "answer": result["answer"],
                "supporting_facts": supporting_facts,
                "type": "summary",
                "reasoning": result.get("reasoning", ""),
                "entity": entity
            }

        except Exception as e:
            if attempt < LLMConfig.MAX_RETRIES:
                await asyncio.sleep(LLMConfig.RETRY_DELAY ** (attempt + 1))
                return await self._generate_summary_async(entity, documents, doc_ids, index, attempt + 1)
            print(f"Error generating summary query after {LLMConfig.MAX_RETRIES} retries: {e}")
            return None


    # Batch Processing


    async def generate_batch_async(
        self,
        samples: List[Dict[str, Any]],
        query_type: str,
        max_concurrent: int = 10,
        start_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Generate queries in batch with concurrency control

        Args:
            samples: List of sample dicts prepared by PreprocessDo
                - single_hop: {"doc_id": int, "text": str}
                - multi_hop: {"documents": [...], "bridges": [...], "doc_ids": [...]}
                - summary: {"entity": str, "documents": [...], "doc_ids": [...]}
            query_type: "single_hop", "multi_hop", or "summary"
            max_concurrent: Maximum concurrent API calls
            start_index: Starting index for ID generation

        Returns:
            List of generated query dicts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def bounded_generate(sample: Dict, idx: int):
            async with semaphore:
                if query_type == "single_hop":
                    return await self._generate_single_hop_async(
                        passage=sample["text"],
                        doc_id=sample["doc_id"],
                        index=idx
                    )
                elif query_type == "multi_hop":
                    return await self._generate_multihop_async(
                        documents=sample["documents"],
                        bridges=sample["bridges"],
                        doc_ids=sample["doc_ids"],
                        index=idx
                    )
                elif query_type == "summary":
                    return await self._generate_summary_async(
                        entity=sample["entity"],
                        documents=sample["documents"],
                        doc_ids=sample["doc_ids"],
                        index=idx
                    )
                else:
                    raise ValueError(f"Unknown query type: {query_type}")

        # Create tasks
        tasks = [
            bounded_generate(sample, start_index + i)
            for i, sample in enumerate(samples)
        ]

        # Execute with progress bar
        desc = f"Generating {query_type}"
        async for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc, unit="q"):
            query = await result
            if query is not None:
                results.append(query)

        # Sort by ID for consistent ordering
        results.sort(key=lambda x: x["id"])

        return results

    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        query_type: str,
        max_concurrent: int = 10,
        start_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Sync wrapper for batch generation

        Args:
            samples: List of sample dicts
            query_type: "single_hop", "multi_hop", or "summary"
            max_concurrent: Maximum concurrent API calls
            start_index: Starting index for ID generation

        Returns:
            List of generated query dicts
        """
        return asyncio.run(
            self.generate_batch_async(samples, query_type, max_concurrent, start_index)
        )


    # Validated Multi-hop Generation


    def generate_multihop_validated(
        self,
        chains: List[Dict[str, Any]],
        target_count: int,
        validator: 'QueryValidator',
        num_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate multi-hop queries with shortcut validation, auto-retry on failure

        Generates queries one by one, validates each with shortcut check.
        If validation fails, uses next chain from pool and retries.

        Args:
            chains: Pre-generated chain pool (should be 3-5x target_count)
            target_count: Number of valid queries to generate
            validator: QueryValidator instance for shortcut check
            num_hops: Number of hops (for metadata)

        Returns:
            List of validated query dicts
        """
        valid_queries = []
        chain_index = 0
        query_index = 0
        failed_count = 0

        pbar = tqdm(total=target_count, desc=f"Generating validated {num_hops}-hop", unit="q")

        while len(valid_queries) < target_count and chain_index < len(chains):
            chain = chains[chain_index]
            chain_index += 1

            # Generate query
            query = self.generate_multihop(
                documents=chain["documents"],
                bridges=chain["bridges"],
                doc_ids=chain["doc_ids"],
                index=query_index
            )

            if query is None:
                continue

            # Immediate shortcut validation
            shortcut_result = validator.check_shortcut(query)

            if shortcut_result["passed"]:
                # Add metadata
                query["num_hops"] = num_hops
                valid_queries.append(query)
                query_index += 1
                pbar.update(1)
            else:
                failed_count += 1
                # Optionally log failure reason
                # print(f"Query failed shortcut: {shortcut_result['shortcut_type']}")

        pbar.close()

        # Print statistics
        total_attempts = chain_index
        success_rate = len(valid_queries) / total_attempts * 100 if total_attempts > 0 else 0
        print(f"\nGeneration complete:")
        print(f"  Valid queries: {len(valid_queries)}/{target_count}")
        print(f"  Total attempts: {total_attempts}")
        print(f"  Failed shortcut: {failed_count}")
        print(f"  Success rate: {success_rate:.1f}%")

        if len(valid_queries) < target_count:
            print(f"  Warning: Only generated {len(valid_queries)} queries, chain pool exhausted")

        return valid_queries

    async def generate_multihop_validated_async(
        self,
        chains: List[Dict[str, Any]],
        target_count: int,
        validator: 'QueryValidator',
        num_hops: int = 2,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Async version of validated multi-hop generation

        Uses streaming approach with as_completed for better throughput.
        Stops early when target count is reached.

        Args:
            chains: Pre-generated chain pool
            target_count: Number of valid queries to generate
            validator: QueryValidator instance
            num_hops: Number of hops (for metadata)
            max_concurrent: Max concurrent generation+validation

        Returns:
            List of validated query dicts
        """
        valid_queries = []
        failed_count = 0
        completed_count = 0

        # Use higher concurrency since each task does 2 LLM calls
        semaphore = asyncio.Semaphore(max_concurrent * 2)

        async def generate_and_validate(chain: Dict, idx: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                # Generate
                query = await self._generate_multihop_async(
                    documents=chain["documents"],
                    bridges=chain["bridges"],
                    doc_ids=chain["doc_ids"],
                    index=idx
                )

                if query is None:
                    return None

                # Validate (async)
                shortcut_result = await validator._check_shortcut_async(query)

                if shortcut_result["passed"]:
                    query["num_hops"] = num_hops
                    return query
                else:
                    return None

        pbar = tqdm(total=target_count, desc=f"Generating validated {num_hops}-hop", unit="q")

        # Create all tasks upfront but limit concurrency via semaphore
        tasks = [
            asyncio.create_task(generate_and_validate(chain, i))
            for i, chain in enumerate(chains)
        ]

        # Process results as they complete (streaming approach)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed_count += 1

            if result is not None:
                valid_queries.append(result)
                pbar.update(1)

                # Early termination when target reached
                if len(valid_queries) >= target_count:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
            else:
                failed_count += 1

        pbar.close()

        # Re-index to ensure sequential IDs
        for i, query in enumerate(valid_queries[:target_count]):
            query["id"] = f"multi_hop_{i:04d}"

        # Print statistics
        success_rate = len(valid_queries) / completed_count * 100 if completed_count > 0 else 0
        print(f"\nGeneration complete:")
        print(f"  Valid queries: {len(valid_queries)}/{target_count}")
        print(f"  Total attempts: {completed_count}")
        print(f"  Failed shortcut: {failed_count}")
        print(f"  Success rate: {success_rate:.1f}%")

        return valid_queries[:target_count]


    # Utility Methods


    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code blocks

        Args:
            content: Raw LLM response

        Returns:
            Parsed dict or None if parsing fails
        """
        try:
            # Remove markdown code blocks if present
            if "```" in content:
                # Extract content between ```json and ```
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                else:
                    content = content.split("```")[1].split("```")[0]
                content = content.strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content: {content[:200]}...")
            return None


# Usage example
if __name__ == "__main__":
    # Example usage
    generator = QueryGenerator()

    # Single-hop example
    passage = "Albert Einstein was born on March 14, 1879, in Ulm, Germany."
    result = generator.generate_single_hop(passage, doc_id=1, index=1)
    if result:
        print(f"Single-hop: {json.dumps(result, indent=2)}")

    # Multi-hop example
    documents = [
        "Steven Spielberg directed the movie Jurassic Park.",
        "Jurassic Park was filmed on location in Hawaii.",
        "The capital of Hawaii is Honolulu."
    ]
    bridges = ["Jurassic Park", "Hawaii"]
    doc_ids = [1, 2, 3]
    result = generator.generate_multihop(documents, bridges, doc_ids, index=1)
    if result:
        print(f"Multi-hop: {json.dumps(result, indent=2)}")

    # Summary example
    entity = "Albert Einstein"
    documents = [
        "Albert Einstein developed the theory of relativity.",
        "Einstein received the Nobel Prize in Physics in 1921.",
        "Einstein was born in Germany but later became a US citizen."
    ]
    doc_ids = [1, 2, 3]
    result = generator.generate_summary(entity, documents, doc_ids, index=1)
    if result:
        print(f"Summary: {json.dumps(result, indent=2)}")
