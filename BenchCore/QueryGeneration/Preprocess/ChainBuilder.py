"""
Chain Builder for Query Generation

This module prepares samples for query generation:
- Single-hop: Random passages from corpus
- Multi-hop: Document chains via random walk on entities
- Summary: Entity clusters with multiple documents
"""
import random
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict

from BenchCore.QueryGeneration.Preprocess.IndexBuilder import IndexBuilder


class ChainBuilder:
    """Build document chains and clusters for query generation"""

    def __init__(self, dataset_name: str, seed: int = 42):
        """Initialize Chain Builder

        Args:
            dataset_name: Name of the dataset
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.index_builder = IndexBuilder(dataset_name)
        self.index_builder.build_entity_doc_index()

        random.seed(seed)


    # Single-hop Sample Preparation
    def prepare_single_hop_samples(
        self,
        n: int,
        min_chunk_length: int = 100,
        max_chunk_length: int = 2000
    ) -> List[Dict[str, Any]]:
        """Prepare samples for single-hop query generation

        Randomly selects chunks that have entities extracted.
        Uses chunk-level text (source_text from triplet extraction).

        Args:
            n: Number of samples to generate
            min_chunk_length: Minimum chunk length (characters)
            max_chunk_length: Maximum chunk length (characters)

        Returns:
            List of {doc_id, chunk_idx, text} dicts
        """
        # Collect all unique chunks from entity_to_chunks
        all_chunks = {}  # (doc_id, chunk_idx) -> text
        for entity, chunks in self.index_builder.entity_to_chunks.items():
            for chunk_info in chunks:
                key = (chunk_info["doc_id"], chunk_info["chunk_idx"])
                if key not in all_chunks:
                    all_chunks[key] = chunk_info["text"]

        # Filter by length
        filtered_chunks = []
        for (doc_id, chunk_idx), text in all_chunks.items():
            if min_chunk_length <= len(text) <= max_chunk_length:
                filtered_chunks.append({
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "text": text
                })

        if len(filtered_chunks) < n:
            print(f"Warning: Only {len(filtered_chunks)} valid chunks, requested {n}")
            n = len(filtered_chunks)

        # Random sample
        samples = random.sample(filtered_chunks, n)
        print(f"Prepared {len(samples)} single-hop samples")

        return samples


    # Multi-hop Chain Building (Random Walk)
    def prepare_multihop_chains(
        self,
        n: int,
        num_hops: int = 2,
        max_attempts: int = 1000
    ) -> List[Dict[str, Any]]:
        """Prepare document chains for multi-hop query generation

        Uses random walk on entity-document graph to build chains.
        Each chain: doc1 --[bridge1]--> doc2 --[bridge2]--> doc3 ...

        Args:
            n: Number of chains to generate
            num_hops: Number of hops (2 = 2 docs, 3 = 3 docs, etc.)
            max_attempts: Maximum attempts to find valid chains

        Returns:
            List of {documents: [...], bridges: [...], doc_ids: [...]} dicts
        """
        chains = []
        attempts = 0
        used_doc_sets = set()  # Avoid duplicate chains

        while len(chains) < n and attempts < max_attempts:
            attempts += 1

            chain = self._build_single_chain(num_hops)

            if chain is None:
                continue

            # Check for duplicate
            doc_set = frozenset(chain["doc_ids"])
            if doc_set in used_doc_sets:
                continue

            used_doc_sets.add(doc_set)
            chains.append(chain)

        print(f"Prepared {len(chains)} multi-hop chains ({num_hops} hops) in {attempts} attempts")

        return chains

    def _build_single_chain(self, num_hops: int) -> Optional[Dict[str, Any]]:
        """Build a single document chain via random walk

        Uses chunk-level text and ensures each hop goes to a DIFFERENT doc_id.
        This enforces cross-document reasoning for multi-hop queries.

        Args:
            num_hops: Number of hops (documents in chain = num_hops, so 2-hop = 2 docs)

        Returns:
            {documents: [...], bridges: [...], doc_ids: [...], chunk_indices: [...]}
            or None if failed
        """
        num_docs = num_hops  # 2-hop means 2 documents connected by 1 bridge

        # Start with a random document that has entities
        all_doc_ids = list(self.index_builder.doc_to_entities.keys())
        if not all_doc_ids:
            return None

        current_doc_id = random.choice(all_doc_ids)
        chain_doc_ids = [current_doc_id]
        chain_chunk_indices = []
        chain_bridges = []
        chain_texts = []

        # Get a random chunk from the starting document
        start_chunks = [
            chunk for chunks in self.index_builder.entity_to_chunks.values()
            for chunk in chunks
            if chunk["doc_id"] == current_doc_id
        ]
        if not start_chunks:
            return None
        start_chunk = random.choice(start_chunks)
        chain_chunk_indices.append(start_chunk["chunk_idx"])
        chain_texts.append(start_chunk["text"])

        # Random walk - must go to DIFFERENT doc_id each hop
        for hop in range(num_docs - 1):
            # Get entities in current document
            current_entities = self.index_builder.get_entities_in_doc(current_doc_id)

            if not current_entities:
                return None

            # Shuffle entities and try to find a valid bridge
            entity_list = list(current_entities)
            random.shuffle(entity_list)

            found_next = False
            for bridge_entity in entity_list:
                # Get chunks containing this entity with DIFFERENT doc_id
                entity_chunks = self.index_builder.get_chunks_for_entity(bridge_entity)
                candidate_chunks = [
                    chunk for chunk in entity_chunks
                    if chunk["doc_id"] not in chain_doc_ids  # Must be different doc_id
                ]

                if candidate_chunks:
                    # Choose a random next chunk
                    next_chunk = random.choice(candidate_chunks)
                    chain_doc_ids.append(next_chunk["doc_id"])
                    chain_chunk_indices.append(next_chunk["chunk_idx"])
                    chain_texts.append(next_chunk["text"])
                    chain_bridges.append(bridge_entity)
                    current_doc_id = next_chunk["doc_id"]
                    found_next = True
                    break

            if not found_next:
                return None

        # Verify all texts have content
        if any(not text for text in chain_texts):
            return None

        return {
            "documents": chain_texts,  # Chunk-level texts
            "bridges": chain_bridges,
            "doc_ids": chain_doc_ids,
            "chunk_indices": chain_chunk_indices
        }


    # Summary Cluster Building
    def prepare_summary_clusters(
        self,
        n: int,
        min_docs: int = 2,
        max_docs: int = 5
    ) -> List[Dict[str, Any]]:
        """Prepare entity clusters for summary query generation

        Finds entities that appear in multiple documents (different doc_ids).
        Uses chunk-level text for each document.

        Args:
            n: Number of clusters to generate
            min_docs: Minimum distinct documents per entity
            max_docs: Maximum documents to include per entity

        Returns:
            List of {entity, documents: [...], doc_ids: [...], chunk_indices: [...]} dicts
        """
        # Get entities with multiple distinct documents
        multi_doc_entities = self.index_builder.get_multi_doc_entities(min_docs=min_docs)

        if not multi_doc_entities:
            print(f"Warning: No entities found with >= {min_docs} documents")
            return []

        # Sort by number of distinct documents (prefer entities with more docs)
        def count_distinct_docs(chunks):
            return len(set(chunk["doc_id"] for chunk in chunks))

        sorted_entities = sorted(
            multi_doc_entities.items(),
            key=lambda x: count_distinct_docs(x[1]),
            reverse=True
        )

        clusters = []
        used_entities = set()

        for entity, chunks in sorted_entities:
            if len(clusters) >= n:
                break

            # Skip if entity name is too short or generic
            if len(entity) < 3:
                continue

            # Group chunks by doc_id, take one chunk per doc
            doc_to_chunk = {}
            for chunk in chunks:
                doc_id = chunk["doc_id"]
                if doc_id not in doc_to_chunk:
                    doc_to_chunk[doc_id] = chunk

            # Limit number of documents
            selected_items = list(doc_to_chunk.items())[:max_docs]

            # Extract data
            doc_ids = [doc_id for doc_id, _ in selected_items]
            chunk_indices = [chunk["chunk_idx"] for _, chunk in selected_items]
            documents = [chunk["text"] for _, chunk in selected_items]

            # Verify all documents have content
            if any(not doc for doc in documents):
                continue

            clusters.append({
                "entity": entity,
                "documents": documents,
                "doc_ids": doc_ids,
                "chunk_indices": chunk_indices
            })
            used_entities.add(entity)

        # If not enough, sample randomly from remaining
        if len(clusters) < n:
            remaining = [
                (entity, chunks) for entity, chunks in sorted_entities
                if entity not in used_entities and len(entity) >= 3
            ]

            if remaining:
                additional_needed = n - len(clusters)
                additional = random.sample(remaining, min(additional_needed, len(remaining)))

                for entity, chunks in additional:
                    # Group chunks by doc_id
                    doc_to_chunk = {}
                    for chunk in chunks:
                        doc_id = chunk["doc_id"]
                        if doc_id not in doc_to_chunk:
                            doc_to_chunk[doc_id] = chunk

                    selected_items = list(doc_to_chunk.items())[:max_docs]
                    doc_ids = [doc_id for doc_id, _ in selected_items]
                    chunk_indices = [chunk["chunk_idx"] for _, chunk in selected_items]
                    documents = [chunk["text"] for _, chunk in selected_items]

                    if any(not doc for doc in documents):
                        continue

                    clusters.append({
                        "entity": entity,
                        "documents": documents,
                        "doc_ids": doc_ids,
                        "chunk_indices": chunk_indices
                    })

        print(f"Prepared {len(clusters)} summary clusters")

        return clusters


    # Combined Preparation
    def prepare_all_samples(
        self,
        single_hop_count: int = 100,
        multihop_count: int = 50,
        summary_count: int = 30,
        num_hops: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Prepare all sample types at once

        Args:
            single_hop_count: Number of single-hop samples
            multihop_count: Number of multi-hop chains
            summary_count: Number of summary clusters
            num_hops: Number of hops for multi-hop chains

        Returns:
            Dict with keys: single_hop, multi_hop, summary
        """
        return {
            "single_hop": self.prepare_single_hop_samples(single_hop_count),
            "multi_hop": self.prepare_multihop_chains(multihop_count, num_hops=num_hops),
            "summary": self.prepare_summary_clusters(summary_count)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about available samples

        Returns:
            Dict with statistics
        """
        index_stats = self.index_builder.get_statistics()

        # Count possible multi-hop chains (rough estimate)
        entities_with_multi_docs = len(self.index_builder.get_multi_doc_entities(min_docs=2))

        return {
            **index_stats,
            "entities_for_multihop": entities_with_multi_docs,
            "entities_for_summary": entities_with_multi_docs,
        }


# Usage example
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build document chains for query generation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--single-hop", type=int, default=10, help="Number of single-hop samples")
    parser.add_argument("--multi-hop", type=int, default=5, help="Number of multi-hop chains")
    parser.add_argument("--summary", type=int, default=5, help="Number of summary clusters")
    parser.add_argument("--num-hops", type=int, default=2, help="Number of hops for multi-hop")
    args = parser.parse_args()

    builder = ChainBuilder(args.dataset)

    # Print statistics
    stats = builder.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Prepare samples
    print("\nPreparing samples...")
    samples = builder.prepare_all_samples(
        single_hop_count=args.single_hop,
        multihop_count=args.multi_hop,
        summary_count=args.summary,
        num_hops=args.num_hops
    )

    # Show sample examples
    print("\n=== Single-hop Sample ===")
    if samples["single_hop"]:
        sample = samples["single_hop"][0]
        print(f"doc_id: {sample['doc_id']}, chunk_idx: {sample['chunk_idx']}")
        print(f"text: {sample['text'][:200]}...")

    print("\n=== Multi-hop Chain ===")
    if samples["multi_hop"]:
        chain = samples["multi_hop"][0]
        print(f"doc_ids: {chain['doc_ids']}")
        print(f"chunk_indices: {chain['chunk_indices']}")
        print(f"bridges: {chain['bridges']}")
        for i, doc in enumerate(chain['documents']):
            print(f"chunk_{i} (doc {chain['doc_ids'][i]}): {doc[:100]}...")

    print("\n=== Summary Cluster ===")
    if samples["summary"]:
        cluster = samples["summary"][0]
        print(f"entity: {cluster['entity']}")
        print(f"doc_ids: {cluster['doc_ids']}")
        print(f"chunk_indices: {cluster['chunk_indices']}")
        for i, doc in enumerate(cluster['documents']):
            print(f"chunk_{i} (doc {cluster['doc_ids'][i]}): {doc[:100]}...")
