import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("")

class EmbConfig:
    """Configuration for embedding models"""

    # Embedding Model
    # Options: "openai", "huggingface", "sentence-transformers", "cohere"
    # PROVIDER = "openai"
    # Options: "openai", "local"  (local uses sentence-transformers for offline inference)
    PROVIDER = "local"

    # Model names by provider
    OPENAI_MODEL = "text-embedding-3-small"        # or "text-embedding-3-large", "text-embedding-ada-002"
    HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    COHERE_MODEL = "embed-english-v3.0"

    # --- [Added] Local model configuration (sentence-transformers) ---
    # LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # or "BAAI/bge-large-en-v1.5", etc.
    LOCAL_MODEL = "/home/lhz/code/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    LOCAL_BATCH_SIZE = 128      # Local models can handle larger batches
    LOCAL_DEVICE = "cuda"       # "cuda" or "cpu"
    # --- [End Added] ---

    # Embedding Dimensions
    EMBEDDING_DIM = 1536          # Dimension of embedding vectors (1536 for OpenAI ada-002)

    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # Batch Processing
    BATCH_SIZE = 30               # Number of texts to embed at once (reduced to avoid rate limit)
    MAX_TOKENS_PER_REQUEST = 8000 # Maximum tokens per API request

    # Caching
    USE_CACHE = True              # Cache embeddings to avoid recomputation

    # Performance
    NORMALIZE_EMBEDDINGS = True   # Normalize vectors to unit length
    DEVICE = "cuda"               # "cuda" or "cpu" for local models

    # BERTScore Configuration (for Semantic F1)
    BERTSCORE_MODEL = "roberta-large"  # Options: "roberta-large", "bert-base-uncased", "distilbert-base-uncased"
    BERTSCORE_LANG = "en"              # Language for BERTScore
    BERTSCORE_DEVICE = "cuda"          # "cuda" or "cpu"

    # SentenceTransformer Configuration (for Soft COV and Faithfulness)
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2"
    SENTENCE_TRANSFORMER_DEVICE = "cuda"              # "cuda" or "cpu"

    # Faithfulness Score Configuration
    FAITHFULNESS_THRESHOLD = 0.7  # Threshold for Hard FS (sentence is faithful if max_sim >= threshold)


# Usage example
if __name__ == "__main__":
    print(f"Provider: {EmbConfig.PROVIDER}")
    print(f"Model: {EmbConfig.OPENAI_MODEL}")
    print(f"Embedding Dimension: {EmbConfig.EMBEDDING_DIM}")
    print(f"Batch Size: {EmbConfig.BATCH_SIZE}")
