# RAGRouter

RAGRouter is a RAG benchmark framework with trainable router support, built on top of RAGRouter-Bench.

## Features

- **5 RAG Paradigms**: NaiveRAG, GraphRAG, HybridRAG, IterativeRAG, LLMDirect
- **Comprehensive Evaluation**: Semantic F1, coverage, faithfulness, LLM-judge metrics
- **Trainable Router**: Support for routing queries to optimal RAG strategies (planned)

## Local Deployment Support

This project supports fully local deployment without external API dependencies.

### LLM Configuration

We separate LLM configurations for different purposes:

| Purpose | Model | Port | Configuration |
|---------|-------|------|---------------|
| Graph Construction | Llama-3.1-70B-AWQ-INT4 | 8000 | Hardcoded in `GraphDo.py` |
| RAG Answering | Llama-3.1-8B-AWQ-INT4 | 8001 | Configurable in `LLMConfig.py` |

**Design Rationale**:
- Graph construction requires stronger reasoning capability for entity/relation extraction → use 70B model
- RAG answering can use lighter model for faster inference → use 8B model
- Hardcoding graph LLM follows the original pattern (was DeepSeek), now replaced with local model

### Embedding Configuration

Supports local sentence-transformers for offline inference:
```python
# Config/EmbConfig.py
PROVIDER = "local"
LOCAL_MODEL = "/path/to/sentence-transformers/all-MiniLM-L6-v2"
```

## Quick Start

### 1. Start vLLM Services

```bash
# Terminal 1: 70B for graph construction
CUDA_VISIBLE_DEVICES=0 vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --port 8000 --quantization awq --max-model-len 8192

# Terminal 2: 8B for RAG answering
CUDA_VISIBLE_DEVICES=1 vllm serve /home/lhz/code/model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --port 8001 --quantization awq --max-model-len 8192
```

### 2. Process Dataset

```bash
python main.py process all --dataset musique
```

### 3. Run RAG and Evaluate

```bash
python main.py retrieve graph --dataset musique
python main.py evaluate result --dataset musique --method graph_rag
```

## Project Structure

```
RAGRouter/
├── Config/                 # Configuration files
│   ├── LLMConfig.py       # LLM provider settings
│   ├── EmbConfig.py       # Embedding model settings
│   └── GraphConfig.py     # Graph construction settings
├── RAGCore/               # RAG implementations
│   ├── Retriever/        # 5 RAG paradigms
│   ├── Graph/            # Knowledge graph construction
│   └── Embedding/        # Embedding generation
├── BenchCore/            # Evaluation modules
├── Dataset/              # Data storage
└── main.py               # CLI entry point
```

## Roadmap

- [ ] Migrate trainable router from routing_rag project
- [ ] Support automatic label generation from evaluation results
- [ ] Add internal representation extraction for router training

## Changes from RAGRouter-Bench

1. **Local Embedding Support**: Added sentence-transformers provider for offline inference
2. **Decoupled LLM Configs**: Separate models for graph construction (70B) and RAG answering (8B)
3. **Router Training Support**: (Planned) Integration of trainable router module