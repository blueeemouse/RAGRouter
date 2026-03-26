import numpy as np
from openai import OpenAI
from typing import Dict, List
from tqdm import tqdm
from Config.EmbConfig import EmbConfig


class EmbeddingProcessor:
    """Generate embeddings for chunks using configured embedding model"""

    def __init__(self):
        self.provider = EmbConfig.PROVIDER
        self.batch_size = EmbConfig.BATCH_SIZE
        self.normalize = EmbConfig.NORMALIZE_EMBEDDINGS

        # Initialize OpenAI client
        if self.provider == "openai":
            self.client = OpenAI(api_key=EmbConfig.OPENAI_API_KEY)
            self.model = EmbConfig.OPENAI_MODEL
        # --- [Added] local provider using sentence-transformers ---
        elif self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(
                EmbConfig.LOCAL_MODEL,
                device=EmbConfig.LOCAL_DEVICE
            )
            self.batch_size = EmbConfig.LOCAL_BATCH_SIZE
        # --- [End Added] ---
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented yet")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.provider == "openai":
            return self._embed_openai(texts)
        # --- [Added] local provider ---
        elif self.provider == "local":
            return self._embed_local(texts)
        # --- [End Added] ---
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    # --- [Added] local embedding method ---
    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local sentence-transformers model"""
        embeddings = self.local_model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return np.array(embeddings)
    # --- [End Added] ---

    def _embed_openai(self, texts: List[str], pbar=None) -> np.ndarray:
        """Generate embeddings using OpenAI API with batching

        Args:
            texts: List of text strings to embed
            pbar: Optional tqdm progress bar to update
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Update progress bar if provided
            if pbar:
                pbar.update(1)

        embeddings = np.array(all_embeddings)

        # Normalize if configured
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    def process_chunks(self, chunks_by_doc: Dict[int, List[str]], dataset_name: str = None,
                      resume: bool = True) -> Dict[int, Dict[str, any]]:
        """Generate embeddings for all chunks with resume support

        Args:
            chunks_by_doc: Output from ChunkProcessor.process_corpus()
                          Format: {doc_id: [chunk1_text, chunk2_text, ...]}
            dataset_name: Name of dataset (for incremental save and resume)
            resume: If True, skip already processed documents

        Returns:
            Dict mapping doc_id to chunk data with embeddings
            Format: {
                doc_id: {
                    "chunks": [chunk1_text, chunk2_text, ...],
                    "embeddings": numpy array of shape (num_chunks, embedding_dim)
                }
            }
        """
        # Load existing embeddings if resume is enabled
        if resume and dataset_name:
            from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
            embeddings_by_doc = EmbeddingSaver.load_existing(dataset_name)

            if embeddings_by_doc:
                print(f"Resuming from checkpoint: {len(embeddings_by_doc)} documents already processed")
        else:
            embeddings_by_doc = {}

        # Calculate total batches for remaining documents
        total_batches = 0
        for doc_id, chunks in chunks_by_doc.items():
            if doc_id not in embeddings_by_doc:
                total_batches += (len(chunks) + self.batch_size - 1) // self.batch_size

        if total_batches == 0:
            print("All documents already processed")
            return embeddings_by_doc

        # Single progress bar for all batches
        with tqdm(total=total_batches, desc="Generating chunk embeddings", unit="batch") as pbar:
            for doc_id, chunks in chunks_by_doc.items():
                # Skip if already processed
                if doc_id in embeddings_by_doc:
                    continue

                # Generate embeddings for all chunks in this document
                # embeddings = self._embed_openai(chunks, pbar=pbar)  # [Original]
                embeddings = self.embed_texts(chunks)  # [Added] supports both openai and local
                if pbar:
                    pbar.update(1)

                embeddings_by_doc[doc_id] = {
                    "chunks": chunks,
                    "embeddings": embeddings
                }

                # Incremental save
                if dataset_name:
                    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
                    EmbeddingSaver.save_incremental(doc_id, chunks, embeddings, dataset_name)

        return embeddings_by_doc


# Usage example
if __name__ == "__main__":
    from Config.PathConfig import PathConfig
    from RAGCore.Chunk.ChunkDo import ChunkProcessor

    # Step 1: Load and chunk corpus
    chunk_processor = ChunkProcessor()
    dataset_name = "hotpotqa"
    corpus_path = PathConfig.get_corpus_path(dataset_name)
    corpus = chunk_processor.load_corpus(corpus_path)
    chunks_by_doc = chunk_processor.process_corpus(corpus)

    print(f"Chunked {len(chunks_by_doc)} documents")

    # Step 2: Generate embeddings
    embedding_processor = EmbeddingProcessor()
    embeddings_by_doc = embedding_processor.process_chunks(chunks_by_doc)

    print(f"Generated embeddings for {len(embeddings_by_doc)} documents")

    # Show example
    first_doc_id = list(embeddings_by_doc.keys())[0]
    doc_data = embeddings_by_doc[first_doc_id]
    print(f"\nDoc {first_doc_id}:")
    print(f"  Num chunks: {len(doc_data['chunks'])}")
    print(f"  Embeddings shape: {doc_data['embeddings'].shape}")
    print(f"  First chunk: {doc_data['chunks'][0][:100]}...")
