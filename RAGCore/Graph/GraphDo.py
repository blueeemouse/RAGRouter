import json
import asyncio
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from Config.LLMConfig import LLMConfig
from Config.EmbConfig import EmbConfig
from Config.GraphConfig import GraphConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


class GraphProcessor:
    """Extract entities and triplets from chunks, then build knowledge graph"""

    def __init__(self):
        # --- [Original] Use unified model config (supports deepseek, llama/local, etc.) ---
        # model_config = LLMConfig.get_model_config()
        # self.llm_client = OpenAI(
        #     api_key=model_config["api_key"],
        #     base_url=model_config["base_url"]
        # )
        # self.llm_model = model_config["model"]
        # --- [End Original] ---
        # --- [Original] DeepSeek hardcoded ---
        # self.llm_client = OpenAI(
        #     api_key=LLMConfig.DEEPSEEK_API_KEY,
        #     base_url=LLMConfig.DEEPSEEK_BASE_URL
        # )
        # self.llm_model = LLMConfig.DEEPSEEK_MODEL
        # --- [End Original] ---
        # --- [Modified] Hardcoded Llama 3.1 70B AWQ-INT4 for graph construction ---
        self.llm_client = OpenAI(
            api_key="not-needed",
            base_url="http://localhost:8000/v1"
        )
        self.llm_model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
        # --- [End Modified] ---

        # Initialize async LLM client for concurrent extraction
        # --- [Original] Use unified model config ---
        # self.async_llm_client = AsyncOpenAI(
        #     api_key=model_config["api_key"],
        #     base_url=model_config["base_url"]
        # )
        # --- [End Original] ---
        # --- [Original] DeepSeek hardcoded ---
        # self.async_llm_client = AsyncOpenAI(
        #     api_key=LLMConfig.DEEPSEEK_API_KEY,
        #     base_url=LLMConfig.DEEPSEEK_BASE_URL
        # )
        # --- [End Original] ---
        # --- [Modified] Hardcoded Llama 3.1 70B AWQ-INT4 for graph construction ---
        self.async_llm_client = AsyncOpenAI(
            api_key="not-needed",
            base_url="http://localhost:8000/v1"
        )
        # --- [End Modified] ---

        # Initialize embedding client for entity embeddings
        # [Original] self.emb_client = OpenAI(api_key=EmbConfig.OPENAI_API_KEY)
        # [Original] self.emb_model = EmbConfig.OPENAI_MODEL
        # --- [Added] Support local embedding provider ---
        if EmbConfig.PROVIDER == "local":
            from sentence_transformers import SentenceTransformer
            self.emb_model_local = SentenceTransformer(
                EmbConfig.LOCAL_MODEL,
                device=EmbConfig.LOCAL_DEVICE
            )
            self.emb_provider = "local"
        else:
            self.emb_client = OpenAI(api_key=EmbConfig.OPENAI_API_KEY)
            self.emb_model = EmbConfig.OPENAI_MODEL
            self.emb_provider = "openai"
        # --- [End Added] ---

    def extract_triplets(self, chunks_by_doc: Dict[int, List[str]]) -> Dict[int, List[Tuple[str, str, str]]]:
        """Extract triplets (subject, predicate, object) from chunks using LLM

        Args:
            chunks_by_doc: {doc_id: [chunk1, chunk2, ...]}

        Returns:
            {doc_id: [(subject, predicate, object), ...]}
        """
        triplets_by_doc = {}

        # Calculate total chunks
        total_chunks = sum(len(chunks) for chunks in chunks_by_doc.values())

        # Progress bar for all chunks
        with tqdm(total=total_chunks, desc="Extracting triplets", unit="chunk") as pbar:
            for doc_id, chunks in chunks_by_doc.items():
                doc_triplets = []

                for chunk in chunks:
                    # Use prompt template (optimized for DeepSeek caching)
                    messages = PromptTemplate.get_triplet_messages(chunk)

                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        temperature=0.0
                    )

                    try:
                        triplets = json.loads(response.choices[0].message.content)
                        for t in triplets:
                            if len(t) == 3:
                                doc_triplets.append(tuple(t))
                    except:
                        pass

                    pbar.update(1)

                triplets_by_doc[doc_id] = doc_triplets

        return triplets_by_doc

    def build_graph(self, triplets_by_doc: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Build knowledge graph from triplets with optimized storage

        Strategy:
        1. Build graph structure from triplets (nodes store entity_id only)
        2. Create entity_id ↔ entity_text mapping
        3. Generate embeddings for all entities
        4. Build triplet_sources for sentence mapping
        5. Return graph + mapping + embeddings + triplet_sources separately

        Args:
            triplets_by_doc: {doc_id: [{"triplet": [s, p, o], "doc_id": int, "chunk_idx": int, "source_text": str}, ...]}

        Returns:
            {
                "graph": NetworkX graph (nodes only contain entity_id),
                "entity_mapping": {entity_id: entity_text},
                "entity_embeddings": {entity_id: embedding_vector},
                "triplet_sources": {(subject, predicate, object): [{"doc_id": int, "chunk_idx": int, "source_text": str}, ...]}
            }
        """
        # Use directed or undirected graph based on config
        G = nx.DiGraph() if GraphConfig.DIRECTED else nx.Graph()

        # Step 1: Collect all unique entities and build triplet_sources
        all_entities = set()
        triplet_sources = {}  # Maps triplet tuple to list of source info

        for doc_id, triplet_records in triplets_by_doc.items():
            for record in triplet_records:
                # Handle new format: {"triplet": [s, p, o], "doc_id": ..., "chunk_idx": ..., "source_text": ...}
                if isinstance(record, dict) and "triplet" in record:
                    triplet = record["triplet"]
                    source_info = {
                        "doc_id": record.get("doc_id", doc_id),
                        "chunk_idx": record.get("chunk_idx", 0),
                        "source_text": record.get("source_text", "")
                    }
                else:
                    # Handle old format: (s, p, o) or [s, p, o]
                    triplet = record
                    source_info = {"doc_id": doc_id, "chunk_idx": 0, "source_text": ""}

                if len(triplet) != 3:
                    continue

                subject, predicate, obj = triplet

                # Skip invalid triplets (containing None or empty strings)
                if not (subject and predicate and obj and
                        isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str) and
                        subject.strip() and predicate.strip() and obj.strip()):
                    continue

                all_entities.add(subject)
                all_entities.add(obj)

                # Store source info for this triplet
                triplet_key = (subject, predicate, obj)
                if triplet_key not in triplet_sources:
                    triplet_sources[triplet_key] = []
                triplet_sources[triplet_key].append(source_info)

        # Create entity_id mapping
        entity_to_id = {entity: idx for idx, entity in enumerate(sorted(all_entities))}
        id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}

        # Step 2: Build graph using entity_ids
        for doc_id, triplet_records in triplets_by_doc.items():
            for record in triplet_records:
                # Handle new format
                if isinstance(record, dict) and "triplet" in record:
                    triplet = record["triplet"]
                else:
                    triplet = record

                if len(triplet) != 3:
                    continue

                subject, predicate, obj = triplet

                # Skip invalid triplets
                if not (subject and predicate and obj and
                        isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str) and
                        subject.strip() and predicate.strip() and obj.strip()):
                    continue

                subject_id = entity_to_id[subject]
                obj_id = entity_to_id[obj]

                # Add nodes with entity_id
                if subject_id not in G.nodes:
                    G.add_node(subject_id, entity_id=subject_id)
                if obj_id not in G.nodes:
                    G.add_node(obj_id, entity_id=obj_id)

                # Add edge
                G.add_edge(subject_id, obj_id,
                          relation=predicate,
                          doc_id=doc_id)

        if not all_entities:
            return {
                "graph": G,
                "entity_mapping": {},
                "entity_embeddings": {},
                "triplet_sources": {}
            }

        # Step 3: Generate embeddings for all entities
        print(f"Generating embeddings for {len(all_entities)} entities...")
        entity_list = [id_to_entity[i] for i in range(len(all_entities))]
        entity_text_to_embedding = self._embed_entities(entity_list)

        # Convert to entity_id → embedding mapping
        entity_embeddings = {
            entity_to_id[entity]: embedding
            for entity, embedding in entity_text_to_embedding.items()
        }

        return {
            "graph": G,
            "entity_mapping": id_to_entity,
            "entity_embeddings": entity_embeddings,
            "triplet_sources": triplet_sources
        }

    def _embed_entities(self, entities: List[str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for entities with batch processing

        Args:
            entities: List of entity names

        Returns:
            {entity: embedding_vector}
        """
        if not entities:
            return {}

        # Filter out empty or whitespace-only entities
        valid_entities = [e for e in entities if e and e.strip()]

        if not valid_entities:
            return {}

        embeddings = {}
        # [Original] batch_size = EmbConfig.BATCH_SIZE
        # [Original] total_batches = (len(valid_entities) + batch_size - 1) // batch_size
        # [Original]
        # [Original] # Process in batches with progress bar
        # [Original] with tqdm(total=total_batches, desc="Generating entity embeddings", unit="batch") as pbar:
        # [Original]     for i in range(0, len(valid_entities), batch_size):
        # [Original]         batch = valid_entities[i:i + batch_size]
        # [Original]
        # [Original]         try:
        # [Original]             response = self.emb_client.embeddings.create(
        # [Original]                 model=self.emb_model,
        # [Original]                 input=batch
        # [Original]             )
        # [Original]
        # [Original]             for j, entity in enumerate(batch):
        # [Original]                 embeddings[entity] = np.array(response.data[j].embedding)
        # [Original]         except Exception as e:
        # [Original]             print(f"\nWarning: Failed to generate embeddings for batch {i//batch_size + 1}")
        # [Original]             print(f"Error: {e}")
        # [Original]             print(f"Batch size: {len(batch)}")
        # [Original]             print(f"Sample entities: {batch[:3]}")
        # [Original]             # Continue with next batch instead of failing completely
        # [Original]
        # [Original]         pbar.update(1)

        # --- [Added] Support both local and openai embedding providers ---
        batch_size = EmbConfig.LOCAL_BATCH_SIZE if self.emb_provider == "local" else EmbConfig.BATCH_SIZE
        total_batches = (len(valid_entities) + batch_size - 1) // batch_size

        if self.emb_provider == "local":
            # Use local sentence-transformers model (all at once, much faster)
            print(f"Generating embeddings for {len(valid_entities)} entities using local model...")
            all_embeddings = self.emb_model_local.encode(
                valid_entities,
                batch_size=batch_size,
                normalize_embeddings=EmbConfig.NORMALIZE_EMBEDDINGS,
                show_progress_bar=True
            )
            for entity, emb in zip(valid_entities, all_embeddings):
                embeddings[entity] = np.array(emb)
        else:
            # Use OpenAI API with batching (original logic)
            with tqdm(total=total_batches, desc="Generating entity embeddings", unit="batch") as pbar:
                for i in range(0, len(valid_entities), batch_size):
                    batch = valid_entities[i:i + batch_size]

                    try:
                        response = self.emb_client.embeddings.create(
                            model=self.emb_model,
                            input=batch
                        )

                        for j, entity in enumerate(batch):
                            embeddings[entity] = np.array(response.data[j].embedding)
                    except Exception as e:
                        print(f"\nWarning: Failed to generate embeddings for batch {i//batch_size + 1}")
                        print(f"Error: {e}")
                        print(f"Batch size: {len(batch)}")
                        print(f"Sample entities: {batch[:3]}")

                    pbar.update(1)
        # --- [End Added] ---

        return embeddings

    def process(self, chunks_by_doc: Dict[int, List[str]], dataset_name: str = None, resume: bool = True) -> Dict[str, Any]:
        """Complete pipeline: extract triplets and build graph

        Args:
            chunks_by_doc: {doc_id: [chunk1, chunk2, ...]}
            dataset_name: Name of dataset (for incremental save and resume)
            resume: If True, skip already processed documents

        Returns:
            {
                "entities": {doc_id: [entity1, ...]},  # Derived from triplets
                "triplets": {doc_id: [(s, p, o), ...]},
                "graph": NetworkX graph object
            }
        """
        # Load existing progress if resume is enabled
        if resume and dataset_name:
            from RAGCore.Graph.GraphSave import GraphSaver
            triplets_by_doc = GraphSaver.load_existing_triplets(dataset_name)

            if triplets_by_doc:
                print(f"Resuming from checkpoint: {len(triplets_by_doc)} documents already processed")
        else:
            triplets_by_doc = {}

        # Extract triplets with incremental save
        # (the incremental version only uses the async version of extract_triplets func)
        print("Extracting triplets...")
        triplets_by_doc = self.extract_triplets_incremental(chunks_by_doc, dataset_name, triplets_by_doc)

        # Retry failed chunks before building graph
        if dataset_name:
            from RAGCore.Graph.GraphSave import GraphSaver
            failed_chunks = GraphSaver.load_failed_chunks(dataset_name)

            if failed_chunks:
                print(f"\n{'='*60}")
                print(f"Retrying {len(failed_chunks)} failed chunks...")
                print(f"{'='*60}")

                triplets_by_doc = self.retry_failed_chunks(failed_chunks, dataset_name, triplets_by_doc)

                # Clear failed chunks file after retry (whether successful or not)
                GraphSaver.clear_failed_chunks(dataset_name)

        # Build graph (entities are derived from triplets inside build_graph)
        print("Building knowledge graph...")
        graph_data = self.build_graph(triplets_by_doc)

        # Derive entities_by_doc from triplets for compatibility
        entities_by_doc = self._derive_entities_from_triplets(triplets_by_doc)

        return {
            "entities": entities_by_doc,  # Derived from triplets
            "triplets": triplets_by_doc,
            "graph": graph_data["graph"],
            "entity_mapping": graph_data["entity_mapping"],
            "entity_embeddings": graph_data["entity_embeddings"],
            "triplet_sources": graph_data["triplet_sources"]
        }

    def _derive_entities_from_triplets(self, triplets_by_doc: Dict[int, List[Any]]) -> Dict[int, List[str]]:
        """Derive entities from triplets for each document

        Args:
            triplets_by_doc: {doc_id: [triplet_records, ...]}
                triplet_records can be either:
                - New format: {"triplet": [s, p, o], "doc_id": ..., "chunk_idx": ..., "source_text": ...}
                - Old format: (s, p, o) or [s, p, o]

        Returns:
            {doc_id: [entity1, entity2, ...]}
        """
        entities_by_doc = {}
        for doc_id, triplet_records in triplets_by_doc.items():
            doc_entities = set()
            for record in triplet_records:
                # Handle new format
                if isinstance(record, dict) and "triplet" in record:
                    triplet = record["triplet"]
                else:
                    triplet = record

                if len(triplet) != 3:
                    continue

                subject, predicate, obj = triplet

                # Skip invalid triplets
                if not (subject and predicate and obj and
                        isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str) and
                        subject.strip() and predicate.strip() and obj.strip()):
                    continue
                doc_entities.add(subject)
                doc_entities.add(obj)
            entities_by_doc[doc_id] = list(doc_entities)
        return entities_by_doc

    def extract_triplets_incremental(self, chunks_by_doc: Dict[int, List[str]], dataset_name: str,
                                    existing_triplets: Dict[int, List[Tuple[str, str, str]]]) -> Dict[int, List[Tuple[str, str, str]]]:
        """Extract triplets with async concurrent processing and incremental save support"""
        # Run async extraction
        return asyncio.run(self._extract_triplets_async(chunks_by_doc, dataset_name, existing_triplets))

    async def _extract_triplets_async(self, chunks_by_doc: Dict[int, List[str]], dataset_name: str,
                                     existing_triplets: Dict[int, List[Tuple[str, str, str]]]) -> Dict[int, List[Tuple[str, str, str]]]:
        """Async implementation of triplet extraction with batch processing and concurrency control

        Processes documents in batches to control memory usage and enable incremental saving.
        Each batch processes BATCH_SIZE documents in parallel, then saves results before moving to next batch.
        """
        from RAGCore.Graph.GraphSave import GraphSaver

        # Filter out already processed documents
        docs_to_process = [(doc_id, chunks) for doc_id, chunks in chunks_by_doc.items()
                          if doc_id not in existing_triplets]

        if not docs_to_process:
            return existing_triplets

        # Calculate total chunks for progress bar
        total_chunks = sum(len(chunks) for _, chunks in docs_to_process)

        print(f"Processing {len(docs_to_process)} documents ({total_chunks} chunks) in batches of {GraphConfig.BATCH_SIZE} documents")

        # Process in batches
        batch_size = GraphConfig.BATCH_SIZE
        num_batches = (len(docs_to_process) + batch_size - 1) // batch_size

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(GraphConfig.MAX_CONCURRENT)

        # Progress bar for all chunks
        with atqdm(total=total_chunks, desc="Extracting triplets", unit="chunk") as pbar:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(docs_to_process))
                batch_docs = docs_to_process[batch_start:batch_end]

                print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (documents {batch_start}-{batch_end-1})")

                # Prepare tasks for this batch
                tasks = []
                for doc_id, chunks in batch_docs:
                    for chunk_idx, chunk in enumerate(chunks):
                        tasks.append(self._extract_single_triplet_async(doc_id, chunk_idx, chunk, dataset_name=dataset_name))

                # Wrap tasks with semaphore
                async def bounded_task(task):
                    async with semaphore:
                        result = await task
                        pbar.update(1)  # Update progress bar after each chunk
                        return result

                bounded_tasks = [bounded_task(task) for task in tasks]

                # Execute batch
                results = await asyncio.gather(*bounded_tasks)

                # Organize results by doc_id for this batch
                batch_triplets = {}
                for doc_id, chunk_idx, triplets in results:
                    if doc_id not in batch_triplets:
                        batch_triplets[doc_id] = []
                    batch_triplets[doc_id].extend(triplets)

                # Update existing_triplets with batch results
                for doc_id, doc_triplets in batch_triplets.items():
                    existing_triplets[doc_id] = doc_triplets

                # Save batch to disk (single write for entire batch)
                if dataset_name and batch_triplets:
                    GraphSaver.save_batch_triplets(batch_triplets, dataset_name)
                    print(f"✓ Saved batch {batch_idx + 1}/{num_batches} ({len(batch_triplets)} documents)")

                # Clear batch results to free memory
                batch_triplets.clear()

        return existing_triplets

    def retry_failed_chunks(self, failed_chunks: List[Dict[str, Any]], dataset_name: str,
                           existing_triplets: Dict[int, List[Tuple[str, str, str]]]) -> Dict[int, List[Tuple[str, str, str]]]:
        """Retry extraction for failed chunks

        Args:
            failed_chunks: List of failed chunk records from GraphSaver.load_failed_chunks()
            dataset_name: Name of the dataset
            existing_triplets: Existing triplets to update

        Returns:
            Updated triplets_by_doc with successful retries
        """
        return asyncio.run(self._retry_failed_chunks_async(failed_chunks, dataset_name, existing_triplets))

    async def _retry_failed_chunks_async(self, failed_chunks: List[Dict[str, Any]], dataset_name: str,
                                        existing_triplets: Dict[int, List[Tuple[str, str, str]]]) -> Dict[int, List[Tuple[str, str, str]]]:
        """Async retry of failed chunks with longer timeout"""
        if not failed_chunks:
            return existing_triplets

        from RAGCore.Graph.GraphSave import GraphSaver

        # Prepare retry tasks
        tasks = []
        for chunk_record in failed_chunks:
            doc_id = chunk_record["doc_id"]
            chunk_idx = chunk_record["chunk_idx"]
            chunk_text = chunk_record["chunk_text"]
            tasks.append(self._extract_single_triplet_async(doc_id, chunk_idx, chunk_text, dataset_name=None))  # Don't save failures again

        # Execute with progress bar
        print(f"Retrying {len(tasks)} failed chunks with concurrency limit {GraphConfig.MAX_CONCURRENT}...")

        semaphore = asyncio.Semaphore(GraphConfig.MAX_CONCURRENT)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]

        # Execute all retry tasks
        with atqdm(total=len(tasks), desc="Retrying failed chunks", unit="chunk") as pbar:
            results = []
            for coro in asyncio.as_completed(bounded_tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        # Organize successful retries by doc_id
        retry_triplets = {}
        successful_retries = 0
        for doc_id, chunk_idx, triplets in results:
            if triplets:  # Only count successful extractions
                if doc_id not in retry_triplets:
                    retry_triplets[doc_id] = []
                retry_triplets[doc_id].extend(triplets)
                successful_retries += 1

        # Update existing triplets with successful retries
        for doc_id, doc_triplets in retry_triplets.items():
            if doc_id in existing_triplets:
                existing_triplets[doc_id].extend(doc_triplets)
            else:
                existing_triplets[doc_id] = doc_triplets

        # Save updated triplets
        if retry_triplets:
            GraphSaver.save_batch_triplets(retry_triplets, dataset_name)
            print(f"✓ Successfully retried {successful_retries}/{len(failed_chunks)} failed chunks")
        else:
            print(f"✗ All retry attempts failed ({len(failed_chunks)} chunks still failed)")

        return existing_triplets

    async def _extract_single_triplet_async(self, doc_id: int, chunk_idx: int, chunk: str,
                                           attempt: int = 0, dataset_name: str = None) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Extract triplets from a single chunk with retry logic

        Returns:
            Tuple of (doc_id, chunk_idx, triplets) where triplets is a list of dicts:
            [{"triplet": [s, p, o], "doc_id": int, "chunk_idx": int, "source_text": str}, ...]
        """
        try:
            messages = PromptTemplate.get_triplet_messages(chunk)
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                timeout=GraphConfig.REQUEST_TIMEOUT
            )

            content = response.choices[0].message.content
            triplets_raw = json.loads(content)
            triplets = []
            for t in triplets_raw:
                if len(t) == 3:
                    # Store triplet with source information
                    triplets.append({
                        "triplet": list(t),
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "source_text": chunk
                    })

            return (doc_id, chunk_idx, triplets)

        except (json.JSONDecodeError, Exception) as e:
            # Retry logic
            if attempt < GraphConfig.MAX_RETRIES:
                await asyncio.sleep(GraphConfig.RETRY_DELAY ** attempt)  # Exponential backoff
                return await self._extract_single_triplet_async(doc_id, chunk_idx, chunk, attempt + 1, dataset_name)
            else:
                # Final failure - save to failed_chunks.json
                print(f"\nWarning: Failed to extract triplets for doc_id={doc_id}, chunk_idx={chunk_idx} after {GraphConfig.MAX_RETRIES} retries")
                print(f"Error: {e}")

                # Save failed chunk (only if dataset_name is provided, to avoid saving retry failures)
                if dataset_name:
                    from RAGCore.Graph.GraphSave import GraphSaver
                    GraphSaver.save_failed_chunk(doc_id, chunk_idx, chunk, str(e), dataset_name)

                return (doc_id, chunk_idx, [])


# Usage example
if __name__ == "__main__":
    from Config.PathConfig import PathConfig
    from RAGCore.Chunk.ChunkDo import ChunkProcessor

    # Load and chunk corpus
    dataset_name = "hotpotqa"
    chunk_processor = ChunkProcessor()
    corpus_path = PathConfig.get_corpus_path(dataset_name)
    corpus = chunk_processor.load_corpus(corpus_path)
    chunks_by_doc = chunk_processor.process_corpus(corpus)

    # Process graph
    graph_processor = GraphProcessor()
    result = graph_processor.process(chunks_by_doc)

    print(f"\nResults:")
    print(f"Documents: {len(result['entities'])}")
    print(f"Total entities: {sum(len(e) for e in result['entities'].values())}")
    print(f"Total triplets: {sum(len(t) for t in result['triplets'].values())}")
    print(f"Graph nodes: {result['graph'].number_of_nodes()}")
    print(f"Graph edges: {result['graph'].number_of_edges()}")
