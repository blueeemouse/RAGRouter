"""
Index Builder for Query Generation

This module builds entity-to-document indices from triplet_sources
for use in multi-hop chain building and summary clustering.
"""
import json
import os
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

from Config.PathConfig import PathConfig


class IndexBuilder:
    """Build indices for query generation preprocessing"""

    def __init__(self, dataset_name: str):
        """Initialize Index Builder

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
        self.corpus: Dict[int, str] = {}  # doc_id -> full text (for reference)
        self.triplet_sources: Dict[Tuple[str, str, str], List[Dict]] = {}

        # Chunk-level indices (用于 query generation)
        self.entity_to_chunks: Dict[str, List[Dict[str, Any]]] = {}  # entity -> [{doc_id, chunk_idx, text}, ...]
        self.doc_to_entities: Dict[int, Set[str]] = {}  # doc_id -> set of entities

        # Legacy alias for compatibility
        self.entity_to_docs = self.entity_to_chunks

        self._load_corpus()
        self._load_triplet_sources()

    def _load_corpus(self):
        """Load corpus from dataset (supports JSON and JSONL formats)"""
        corpus_path = PathConfig.get_corpus_path(self.dataset_name)

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # Standard JSON array
                corpus_data = json.load(f)
                for doc in corpus_data:
                    doc_id = doc.get("id", doc.get("doc_id"))
                    text = doc.get("text", doc.get("content", ""))
                    if doc_id is not None and text:
                        self.corpus[int(doc_id)] = text
            elif first_char == '{':
                # Try JSON dict first, fallback to JSONL
                try:
                    corpus_data = json.load(f)
                    for doc_id, doc_content in corpus_data.items():
                        if isinstance(doc_content, str):
                            self.corpus[int(doc_id)] = doc_content
                        elif isinstance(doc_content, dict):
                            text = doc_content.get("text", doc_content.get("content", ""))
                            if text:
                                self.corpus[int(doc_id)] = text
                except json.JSONDecodeError:
                    # JSONL format
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            doc_id = doc.get("id", doc.get("doc_id"))
                            text = doc.get("text", doc.get("content", ""))
                            if doc_id is not None and text:
                                self.corpus[int(doc_id)] = text
            else:
                # JSONL format (each line is a JSON object)
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        doc_id = doc.get("id", doc.get("doc_id"))
                        text = doc.get("text", doc.get("content", ""))
                        if doc_id is not None and text:
                            self.corpus[int(doc_id)] = text

        print(f"Loaded corpus: {len(self.corpus)} documents")

    def _load_triplet_sources(self):
        """Load triplet_sources from processed data"""
        triplet_dir = PathConfig.get_triplet_path(self.dataset_name)
        resource_path = os.path.join(triplet_dir, "triplet_resource.json")

        if not os.path.exists(resource_path):
            print(f"Warning: triplet_resource.json not found at {resource_path}")
            print("Run graph processing first: python Run/Process/run_graph.py --dataset <name>")
            return

        with open(resource_path, 'r', encoding='utf-8') as f:
            resource_data = json.load(f)

        # Convert string keys back to tuples
        for key, sources in resource_data.items():
            parts = key.split("|||")
            if len(parts) == 3:
                self.triplet_sources[tuple(parts)] = sources

        print(f"Loaded triplet_sources: {len(self.triplet_sources)} triplets")

    def build_entity_doc_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build entity -> chunks index from triplet_sources

        Uses chunk-level text (source_text) instead of full document text
        to handle long documents that exceed token limits.

        Returns:
            Dict mapping entity -> [{doc_id, chunk_idx, text}, ...]
        """
        # entity -> {(doc_id, chunk_idx): text} for deduplication
        entity_chunks = defaultdict(dict)

        for (subject, relation, obj), sources in self.triplet_sources.items():
            for source_info in sources:
                doc_id = source_info.get("doc_id")
                chunk_idx = source_info.get("chunk_idx", 0)
                # Use chunk-level text from triplet extraction
                chunk_text = source_info.get("source_text", "")

                if doc_id is not None and chunk_text:
                    chunk_key = (doc_id, chunk_idx)

                    # Add chunk for subject entity
                    if subject and subject.strip():
                        entity_chunks[subject][chunk_key] = chunk_text

                    # Add chunk for object entity
                    if obj and obj.strip():
                        entity_chunks[obj][chunk_key] = chunk_text

        # Convert to list format with chunk info
        self.entity_to_chunks = {
            entity: [
                {"doc_id": doc_id, "chunk_idx": chunk_idx, "text": text}
                for (doc_id, chunk_idx), text in chunks.items()
            ]
            for entity, chunks in entity_chunks.items()
        }

        # Legacy alias for compatibility
        self.entity_to_docs = self.entity_to_chunks

        # Build reverse index: doc_id -> entities
        self.doc_to_entities = defaultdict(set)
        for entity, chunks in self.entity_to_chunks.items():
            for chunk_info in chunks:
                self.doc_to_entities[chunk_info["doc_id"]].add(entity)

        print(f"Built entity-chunk index: {len(self.entity_to_chunks)} entities")
        print(f"Built doc-entity index: {len(self.doc_to_entities)} documents")

        return self.entity_to_chunks

    def get_entities_in_doc(self, doc_id: int) -> Set[str]:
        """Get all entities mentioned in a document

        Args:
            doc_id: Document ID

        Returns:
            Set of entity names
        """
        return self.doc_to_entities.get(doc_id, set())

    def get_chunks_for_entity(self, entity: str) -> List[Dict[str, Any]]:
        """Get all chunks containing an entity

        Args:
            entity: Entity name

        Returns:
            List of {doc_id, chunk_idx, text} dicts
        """
        return self.entity_to_chunks.get(entity, [])

    # Legacy alias
    def get_docs_for_entity(self, entity: str) -> List[Dict[str, Any]]:
        """Legacy alias for get_chunks_for_entity"""
        return self.get_chunks_for_entity(entity)

    def get_shared_entities(self, doc_id1: int, doc_id2: int) -> Set[str]:
        """Get entities shared between two documents

        Args:
            doc_id1: First document ID
            doc_id2: Second document ID

        Returns:
            Set of shared entity names
        """
        entities1 = self.get_entities_in_doc(doc_id1)
        entities2 = self.get_entities_in_doc(doc_id2)
        return entities1 & entities2

    def get_document_text(self, doc_id: int) -> str:
        """Get document text by ID

        Args:
            doc_id: Document ID

        Returns:
            Document text or empty string if not found
        """
        return self.corpus.get(doc_id, "")

    def get_all_doc_ids(self) -> List[int]:
        """Get all document IDs in corpus

        Returns:
            List of document IDs
        """
        return list(self.corpus.keys())

    def get_multi_doc_entities(self, min_docs: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Get entities that appear in multiple documents (not just multiple chunks)

        Args:
            min_docs: Minimum number of distinct documents required

        Returns:
            Dict mapping entity -> [{doc_id, chunk_idx, text}, ...] for entities with >= min_docs
        """
        result = {}
        for entity, chunks in self.entity_to_chunks.items():
            # Count distinct doc_ids
            distinct_doc_ids = set(chunk["doc_id"] for chunk in chunks)
            if len(distinct_doc_ids) >= min_docs:
                result[entity] = chunks
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics

        Returns:
            Dict with statistics
        """
        if not self.entity_to_chunks:
            self.build_entity_doc_index()

        chunk_counts = [len(chunks) for chunks in self.entity_to_chunks.values()]
        entity_counts = [len(entities) for entities in self.doc_to_entities.values()]

        # Count distinct docs per entity
        doc_counts_per_entity = []
        for chunks in self.entity_to_chunks.values():
            distinct_docs = set(chunk["doc_id"] for chunk in chunks)
            doc_counts_per_entity.append(len(distinct_docs))

        return {
            "corpus_size": len(self.corpus),
            "triplet_count": len(self.triplet_sources),
            "entity_count": len(self.entity_to_chunks),
            "avg_chunks_per_entity": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0,
            "max_chunks_per_entity": max(chunk_counts) if chunk_counts else 0,
            "avg_docs_per_entity": sum(doc_counts_per_entity) / len(doc_counts_per_entity) if doc_counts_per_entity else 0,
            "max_docs_per_entity": max(doc_counts_per_entity) if doc_counts_per_entity else 0,
            "avg_entities_per_doc": sum(entity_counts) / len(entity_counts) if entity_counts else 0,
            "entities_in_multiple_docs": len(self.get_multi_doc_entities(min_docs=2)),
        }


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build entity-document index")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    args = parser.parse_args()

    builder = IndexBuilder(args.dataset)
    builder.build_entity_doc_index()

    stats = builder.get_statistics()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
