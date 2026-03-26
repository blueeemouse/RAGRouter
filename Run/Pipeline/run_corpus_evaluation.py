"""
Complete Corpus Processing and Evaluation Pipeline

This pipeline runs the entire workflow from raw corpus to evaluation results:
1. Chunking: Split documents into chunks
2. Embedding: Generate embeddings for chunks
3. Graph: Extract entities/triplets and build knowledge graph
4. Evaluation: Run structure and semantic evaluations

Usage:
    python -m Run.Pipeline.run_corpus_evaluation --dataset test
    python -m Run.Pipeline.run_corpus_evaluation --dataset hotpotqa --hubness_k 15
    python -m Run.Pipeline.run_corpus_evaluation --dataset mix --skip-chunk --skip-embedding
"""
import argparse
import sys
import os

from Config.PathConfig import PathConfig
from RAGCore.Chunk.ChunkDo import ChunkProcessor
from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
from RAGCore.Graph.GraphDo import GraphProcessor
from RAGCore.Graph.GraphSave import GraphSaver
from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationDo import StructureEvaluator
from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationSave import StructureEvaluationSaver
from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationDo import SemanticEvaluator
from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationSave import SemanticEvaluationSaver


def run_corpus_evaluation(dataset_name: str, hubness_k: int = 10,
                         skip_chunk: bool = False, skip_embedding: bool = False,
                         skip_graph: bool = False):
    """Run complete corpus processing and evaluation pipeline

    Args:
        dataset_name: Name of the dataset (e.g., "hotpotqa")
        hubness_k: K value for hubness calculation (default: 10)
        skip_chunk: Skip chunking step (use existing chunks)
        skip_embedding: Skip embedding step (use existing embeddings)
        skip_graph: Skip graph step (use existing graph)
    """
    print("=" * 80)
    print(f"Corpus Processing and Evaluation Pipeline - {dataset_name}")
    print("=" * 80)

    chunks_by_doc = None
    embeddings_by_doc = None

    # Step 1: Chunking
    if not skip_chunk:
        print("\n[1/5] Chunking corpus...")
        try:
            chunk_processor = ChunkProcessor()
            corpus_path = PathConfig.get_corpus_path(dataset_name)

            if not os.path.exists(corpus_path):
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

            corpus = chunk_processor.load_corpus(corpus_path)
            chunks_by_doc = chunk_processor.process_corpus(corpus)
            print(f"✓ Chunking completed: {len(chunks_by_doc)} documents")
        except Exception as e:
            print(f"✗ Chunking failed: {e}")
            return 1
    else:
        print("\n[1/5] Skipping chunking")

    # Step 2: Embedding
    if not skip_embedding:
        print("\n[2/5] Generating embeddings...")
        try:
            if chunks_by_doc is None:
                chunk_processor = ChunkProcessor()
                corpus_path = PathConfig.get_corpus_path(dataset_name)
                corpus = chunk_processor.load_corpus(corpus_path)
                chunks_by_doc = chunk_processor.process_corpus(corpus)

            embedding_processor = EmbeddingProcessor()
            embeddings_by_doc = embedding_processor.process_chunks(chunks_by_doc, dataset_name=dataset_name, resume=True)
            EmbeddingSaver.save(embeddings_by_doc, dataset_name)
            print(f"✓ Embedding completed")
        except Exception as e:
            print(f"✗ Embedding failed: {e}")
            return 1
    else:
        print("\n[2/5] Skipping embedding")

    # Step 3: Graph Building
    if not skip_graph:
        print("\n[3/5] Building knowledge graph...")
        try:
            # Check if graph already exists (including entity index)
            graph_dir = PathConfig.get_graph_path(dataset_name)
            entity_index_path = os.path.join(graph_dir, "entity_index.faiss")

            if os.path.exists(entity_index_path):
                print(f"  Graph and entity index already exist, skipping graph building")
                print(f"  (Use --skip-graph to explicitly skip this step)")
            else:
                if chunks_by_doc is None:
                    chunk_processor = ChunkProcessor()
                    corpus_path = PathConfig.get_corpus_path(dataset_name)
                    corpus = chunk_processor.load_corpus(corpus_path)
                    chunks_by_doc = chunk_processor.process_corpus(corpus)

                graph_processor = GraphProcessor()
                graph_result = graph_processor.process(chunks_by_doc, dataset_name=dataset_name, resume=True)
                GraphSaver.save(graph_result, dataset_name)
            print(f"✓ Graph building completed")
        except Exception as e:
            print(f"✗ Graph building failed: {e}")
            return 1
    else:
        print("\n[3/5] Skipping graph building")

    # Step 4: Structure Evaluation
    print("\n[4/5] Structure evaluation...")
    structure_success = False
    try:
        graph_data = GraphSaver.load(dataset_name)
        structure_evaluator = StructureEvaluator()
        structure_metrics = structure_evaluator.evaluate(graph_data["graph"])
        StructureEvaluationSaver.save(structure_metrics, dataset_name)
        structure_success = True
        print(f"✓ Structure evaluation completed")
    except Exception as e:
        print(f"✗ Structure evaluation failed: {e}")

    # Step 5: Semantic Evaluation
    print("\n[5/5] Semantic evaluation...")
    semantic_success = False
    try:
        if embeddings_by_doc is None:
            embeddings_by_doc = EmbeddingSaver.load(dataset_name)

        semantic_evaluator = SemanticEvaluator(hubness_k=hubness_k)
        semantic_metrics = semantic_evaluator.evaluate(embeddings_by_doc)

        if semantic_metrics.get('status') != 'success':
            raise RuntimeError(f"Evaluation failed: {semantic_metrics.get('error')}")

        SemanticEvaluationSaver.save(semantic_metrics, dataset_name)
        semantic_success = True
        print(f"✓ Semantic evaluation completed")
    except Exception as e:
        print(f"✗ Semantic evaluation failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Chunking:             {'✓' if not skip_chunk else '○ Skipped'}")
    print(f"  Embedding:            {'✓' if not skip_embedding else '○ Skipped'}")
    print(f"  Graph:                {'✓' if not skip_graph else '○ Skipped'}")
    print(f"  Structure Evaluation: {'✓' if structure_success else '✗ Failed'}")
    print(f"  Semantic Evaluation:  {'✓' if semantic_success else '✗ Failed'}")
    print("=" * 80)

    if structure_success and semantic_success:
        return 0
    else:
        return 1


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run complete corpus processing and evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline from scratch
  python -m Run.Pipeline.run_corpus_evaluation --dataset test

  # Skip already-processed steps
  python -m Run.Pipeline.run_corpus_evaluation --dataset test --skip-chunk --skip-embedding

  # Run only evaluations (all processing steps already done)
  python -m Run.Pipeline.run_corpus_evaluation --dataset mix --skip-chunk --skip-embedding --skip-graph
        """
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (required)")
    parser.add_argument("--hubness_k", type=int, default=10, help="K for hubness calculation (default: 10)")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip chunking step")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding step")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph building step")
    args = parser.parse_args()

    exit_code = run_corpus_evaluation(
        dataset_name=args.dataset,
        hubness_k=args.hubness_k,
        skip_chunk=args.skip_chunk,
        skip_embedding=args.skip_embedding,
        skip_graph=args.skip_graph
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
