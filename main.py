import argparse
import sys
import os

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Process Commands
def cmd_process_embedding(args):
    """Generate embeddings for a dataset"""
    from RAGCore.Chunk.ChunkDo import ChunkProcessor
    from RAGCore.Embedding.EmbeddingDo import EmbeddingProcessor
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
    from Config.PathConfig import PathConfig

    print(f"[Process] Generating embeddings for dataset: {args.dataset}")

    # Step 1: Chunking
    chunk_processor = ChunkProcessor()
    corpus_path = PathConfig.get_corpus_path(args.dataset)
    corpus = chunk_processor.load_corpus(corpus_path)
    chunks_by_doc = chunk_processor.process_corpus(corpus)

    # Step 2: Generate embeddings
    embedding_processor = EmbeddingProcessor()
    embeddings_by_doc = embedding_processor.process_chunks(chunks_by_doc)

    # Step 3: Save embeddings
    EmbeddingSaver.save(embeddings_by_doc, args.dataset)

    print("[Process] Embedding generation complete!")
    return 0


def cmd_process_index(args):
    """Build FAISS index for a dataset"""
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
    from RAGCore.Index.IndexDo import IndexProcessor
    from RAGCore.Index.IndexSave import IndexSaver

    print(f"[Process] Building index for dataset: {args.dataset}")

    # Step 1: Load embeddings
    embeddings_by_doc = EmbeddingSaver.load(args.dataset)

    # Step 2: Build index
    index_processor = IndexProcessor()
    index_data = index_processor.build_index(embeddings_by_doc)

    # Step 3: Save index
    IndexSaver.save(index_data, args.dataset)

    print("[Process] Index building complete!")
    return 0


def cmd_process_graph(args):
    """Build knowledge graph for a dataset"""
    from RAGCore.Chunk.ChunkDo import ChunkProcessor
    from RAGCore.Graph.GraphDo import GraphProcessor
    from RAGCore.Graph.GraphSave import GraphSaver
    from Config.PathConfig import PathConfig

    print(f"[Process] Building knowledge graph for dataset: {args.dataset}")

    # Step 1: Chunking
    chunk_processor = ChunkProcessor()
    corpus_path = PathConfig.get_corpus_path(args.dataset)
    corpus = chunk_processor.load_corpus(corpus_path)
    chunks_by_doc = chunk_processor.process_corpus(corpus)

    # Step 2: Build graph
    graph_processor = GraphProcessor()
    graph_result = graph_processor.process(
        chunks_by_doc,
        dataset_name=args.dataset,
        resume=not args.no_resume
    )

    # Step 3: Save graph
    GraphSaver.save(graph_result, args.dataset)

    print("[Process] Graph building complete!")
    return 0


def cmd_process_all(args):
    """Run all preprocessing steps"""
    print(f"[Process] Running all preprocessing for dataset: {args.dataset}")
    print("=" * 60)

    # Step 1: Embedding
    print("\n[1/3] Generating embeddings...")
    cmd_process_embedding(args)

    # Step 2: Index
    print("\n[2/3] Building index...")
    cmd_process_index(args)

    # Step 3: Graph
    print("\n[3/3] Building knowledge graph...")
    cmd_process_graph(args)

    print("\n" + "=" * 60)
    print("[Process] All preprocessing complete!")
    return 0


# Retrieve Commands
def cmd_retrieve_naive(args):
    """Run Naive RAG retrieval (with parallel processing)"""
    from RAGCore.Retriever.NaiveRAG.NaiveRAGDo import NaiveRAGProcessor
    from RAGCore.Retriever.NaiveRAG.NaiveRAGSave import NaiveRAGSaver
    from Config.PathConfig import PathConfig

    print(f"[Retrieve] Running Naive RAG on dataset: {args.dataset}")

    processor = NaiveRAGProcessor(dataset_name=args.dataset)

    # Use async parallel processing
    max_concurrent = getattr(args, 'max_concurrent', 10)
    results = processor.process_async(
        dataset_name=args.dataset,
        resume=not args.no_resume,
        max_concurrent=max_concurrent
    )
    NaiveRAGSaver.save_all(results, processor.model_name, args.dataset)

    output_path = PathConfig.get_naive_rag_path(processor.model_name, args.dataset)
    print(f"[Retrieve] Naive RAG complete! Results: {output_path}")
    print(f"  - Total answers: {len(results)}")
    return 0


def cmd_retrieve_graph(args):
    """Run Graph RAG retrieval (with parallel processing)"""
    from RAGCore.Retriever.GraphRAG.GraphRAGDo import GraphRAGProcessor
    from RAGCore.Retriever.GraphRAG.GraphRAGSave import GraphRAGSaver
    from Config.PathConfig import PathConfig

    print(f"[Retrieve] Running Graph RAG on dataset: {args.dataset}")

    processor = GraphRAGProcessor(dataset_name=args.dataset)

    # Use async parallel processing
    max_concurrent = getattr(args, 'max_concurrent', 10)
    results = processor.process_async(
        dataset_name=args.dataset,
        resume=not args.no_resume,
        max_concurrent=max_concurrent
    )
    GraphRAGSaver.save_all(results, processor.model_name, args.dataset)

    output_path = PathConfig.get_graph_rag_path(processor.model_name, args.dataset)
    print(f"[Retrieve] Graph RAG complete! Results: {output_path}")
    print(f"  - Total answers: {len(results)}")
    return 0


def cmd_retrieve_hybrid(args):
    """Run Hybrid RAG retrieval (with parallel processing)"""
    from RAGCore.Retriever.HybridRAG.HybridRAGDo import HybridRAGProcessor
    from RAGCore.Retriever.HybridRAG.HybridRAGSave import HybridRAGSaver
    from Config.PathConfig import PathConfig

    print(f"[Retrieve] Running Hybrid RAG on dataset: {args.dataset}")

    processor = HybridRAGProcessor(dataset_name=args.dataset)

    # Use async parallel processing
    max_concurrent = getattr(args, 'max_concurrent', 10)
    results = processor.process_async(
        dataset_name=args.dataset,
        resume=not args.no_resume,
        max_concurrent=max_concurrent
    )
    HybridRAGSaver.save_all(results, processor.model_name, args.dataset)

    output_path = PathConfig.get_hybrid_rag_path(processor.model_name, args.dataset)
    print(f"[Retrieve] Hybrid RAG complete! Results: {output_path}")
    print(f"  - Total answers: {len(results)}")
    return 0


def cmd_retrieve_iterative(args):
    """Run Iterative RAG retrieval (with parallel processing support)"""
    from RAGCore.Retriever.IterativeRAG.IterativeRAGDo import IterativeRAGProcessor
    from RAGCore.Retriever.IterativeRAG.IterativeRAGSave import IterativeRAGSaver
    from Config.PathConfig import PathConfig
    from Config.RetrieverConfig import RetrieverConfig

    print(f"[Retrieve] Running Iterative RAG on dataset: {args.dataset}")
    print(f"  - Retriever: {RetrieverConfig.ITERATIVE_RETRIEVER}")
    print(f"  - Max iterations: {RetrieverConfig.ITERATIVE_MAX_ITERATIONS}")

    processor = IterativeRAGProcessor(dataset_name=args.dataset)

    # Use async parallel processing by default
    max_concurrent = getattr(args, 'max_concurrent', 3)
    results = processor.process_async(
        dataset_name=args.dataset,
        resume=not args.no_resume,
        max_concurrent=max_concurrent
    )
    IterativeRAGSaver.save_all(results, processor.model_name, args.dataset, processor.retriever_type)

    output_path = PathConfig.get_iterative_rag_path(processor.model_name, args.dataset, processor.retriever_type)
    print(f"[Retrieve] Iterative RAG complete! Results: {output_path}")
    print(f"  - Total answers: {len(results)}")

    if results:
        total_rounds = sum(r.get('rounds', 0) for r in results)
        avg_rounds = total_rounds / len(results)
        print(f"  - Average rounds: {avg_rounds:.2f}")
    return 0


def cmd_retrieve_llm_direct(args):
    """Run LLM Direct QA (no retrieval, with parallel processing)"""
    from RAGCore.Retriever.LLMDirect.LLMDirectDo import LLMDirectProcessor
    from RAGCore.Retriever.LLMDirect.LLMDirectSave import LLMDirectSaver
    from Config.PathConfig import PathConfig

    print(f"[Retrieve] Running LLM Direct on dataset: {args.dataset}")

    processor = LLMDirectProcessor()

    # Use async parallel processing
    max_concurrent = getattr(args, 'max_concurrent', 10)
    results = processor.process_async(
        dataset_name=args.dataset,
        resume=not args.no_resume,
        max_concurrent=max_concurrent
    )
    LLMDirectSaver.save_all(results, processor.model_name, args.dataset)

    output_path = PathConfig.get_llm_direct_path(processor.model_name, args.dataset)
    print(f"[Retrieve] LLM Direct complete! Results: {output_path}")
    print(f"  - Total answers: {len(results)}")
    return 0


def cmd_evaluate_result(args):
    """Evaluate RAG results with all metrics (semantic_f1, coverage, faithfulness, llm)"""
    from BenchCore.Evaluation.ResultEvaluation.EvaluationDo import ResultEvaluator
    from Config.LLMConfig import LLMConfig
    from Config.PathConfig import PathConfig

    model_config = LLMConfig.get_model_config()
    model_name = model_config["model"]

    # Get retriever_type for iterative_rag
    retriever_type = getattr(args, 'retriever_type', 'graph')

    print(f"[Evaluate] Running result evaluation")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Method: {args.method}")
    print(f"  - Model: {model_name}")
    if args.method == "iterative_rag":
        print(f"  - Retriever Type: {retriever_type}")
    print(f"  - Skip LLM: {getattr(args, 'skip_llm', False)}")

    evaluator = ResultEvaluator()
    max_concurrent = getattr(args, 'max_concurrent', 10)
    results = evaluator.evaluate_all_metrics(
        model_name=model_name,
        dataset_name=args.dataset,
        method=args.method,
        retriever_type=retriever_type,
        resume=not args.no_resume,
        skip_llm=getattr(args, 'skip_llm', False),
        max_concurrent=max_concurrent
    )

    if results:
        output_path = PathConfig.get_result_eval_path(model_name, args.dataset, args.method)
        print(f"\n[Evaluate] Results saved to: {output_path}")
    return 0


def cmd_evaluate_semantic(args):
    """Evaluate semantic quality of embeddings"""
    from RAGCore.Embedding.EmbeddingSave import EmbeddingSaver
    from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationDo import SemanticEvaluator
    from BenchCore.Evaluation.CorpusEvaluation.SemanticEvaluation.EvaluationSave import SemanticEvaluationSaver

    print(f"[Evaluate] Running semantic evaluation for dataset: {args.dataset}")

    # Load embeddings
    embeddings_by_doc = EmbeddingSaver.load(args.dataset)

    # Evaluate
    evaluator = SemanticEvaluator(hubness_k=args.hubness_k)
    metrics = evaluator.evaluate(embeddings_by_doc)

    # Save
    SemanticEvaluationSaver.save(metrics, args.dataset)

    print("[Evaluate] Semantic evaluation complete!")
    return 0


def cmd_evaluate_structure(args):
    """Evaluate graph structure"""
    from RAGCore.Graph.GraphSave import GraphSaver
    from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationDo import StructureEvaluator
    from BenchCore.Evaluation.CorpusEvaluation.StructureEvaluation.EvaluationSave import StructureEvaluationSaver

    print(f"[Evaluate] Running structure evaluation for dataset: {args.dataset}")

    # Load graph
    result = GraphSaver.load(args.dataset)
    graph = result['graph']

    # Evaluate
    evaluator = StructureEvaluator()
    metrics = evaluator.evaluate(graph)

    # Save
    StructureEvaluationSaver.save(metrics, args.dataset)

    print("[Evaluate] Structure evaluation complete!")
    return 0


# Pipeline Command
def cmd_pipeline(args):
    """Run full pipeline: process -> retrieve -> evaluate"""
    print("=" * 60)
    print(f"[Pipeline] Running full pipeline")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Method: {args.method}")
    print("=" * 60)

    # Step 1: Process (if not skip)
    if not args.skip_process:
        print("\n[Pipeline] Step 1: Preprocessing...")
        cmd_process_all(args)
    else:
        print("\n[Pipeline] Step 1: Skipping preprocessing (--skip-process)")

    # Step 2: Retrieve
    print(f"\n[Pipeline] Step 2: Running {args.method} retrieval...")
    if args.method == "naive":
        cmd_retrieve_naive(args)
    elif args.method == "graph":
        cmd_retrieve_graph(args)
    elif args.method == "hybrid":
        cmd_retrieve_hybrid(args)
    elif args.method == "iterative":
        cmd_retrieve_iterative(args)
    elif args.method == "llm_direct":
        cmd_retrieve_llm_direct(args)
    elif args.method == "all":
        print("  Running all methods...")
        cmd_retrieve_naive(args)
        cmd_retrieve_graph(args)
        cmd_retrieve_hybrid(args)
        cmd_retrieve_iterative(args)

    # Step 3: Evaluate (if not skip)
    if not args.skip_eval:
        print("\n[Pipeline] Step 3: Evaluation...")
        if args.method in ["naive", "graph"]:
            args.method = f"{args.method}_rag"
            cmd_evaluate_result(args)
    else:
        print("\n[Pipeline] Step 3: Skipping evaluation (--skip-eval)")

    print("\n" + "=" * 60)
    print("[Pipeline] Full pipeline complete!")
    return 0


# Main Entry Point
def main():
    parser = argparse.ArgumentParser(
        prog="RAGRouter-Bench",
        description="A Comprehensive RAG Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process all --dataset musique
  python main.py retrieve graph --dataset musique
  python main.py evaluate result --dataset musique --method graph_rag
  python main.py pipeline --dataset musique --method graph
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process Subcommand
    process_parser = subparsers.add_parser("process", help="Data preprocessing")
    process_sub = process_parser.add_subparsers(dest="process_cmd")

    # process embedding
    p_emb = process_sub.add_parser("embedding", help="Generate embeddings")
    p_emb.add_argument("--dataset", type=str, required=True, help="Dataset name")

    # process index
    p_idx = process_sub.add_parser("index", help="Build FAISS index")
    p_idx.add_argument("--dataset", type=str, required=True, help="Dataset name")

    # process graph
    p_graph = process_sub.add_parser("graph", help="Build knowledge graph")
    p_graph.add_argument("--dataset", type=str, required=True, help="Dataset name")
    p_graph.add_argument("--no-resume", action="store_true", help="Don't resume from existing")

    # process all
    p_all = process_sub.add_parser("all", help="Run all preprocessing")
    p_all.add_argument("--dataset", type=str, required=True, help="Dataset name")
    p_all.add_argument("--no-resume", action="store_true", help="Don't resume from existing")

    # Retrieve Subcommand
    retrieve_parser = subparsers.add_parser("retrieve", help="Run RAG retrieval")
    retrieve_sub = retrieve_parser.add_subparsers(dest="retrieve_cmd")

    # Common arguments for all retrieve commands
    for name, help_text in [
        ("naive", "Naive RAG (vector retrieval)"),
        ("graph", "Graph RAG (graph traversal)"),
        ("hybrid", "Hybrid RAG (naive + graph)"),
        ("iterative", "Iterative RAG (multi-round)"),
        ("llm_direct", "LLM Direct (no retrieval)")
    ]:
        p = retrieve_sub.add_parser(name, help=help_text)
        p.add_argument("--dataset", type=str, required=True, help="Dataset name")
        p.add_argument("--no-resume", action="store_true", help="Don't resume from existing")
        p.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls (default: 10)")

    # retry null answers
    p_retry = retrieve_sub.add_parser("retry", help="Retry null answers using existing retrieval")
    p_retry.add_argument("--dataset", type=str, required=True, help="Dataset name")
    p_retry.add_argument("--method", type=str, required=True, choices=["naive", "graph"],
                         help="Method to retry (naive or graph)")
    p_retry.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls (default: 10)")

    # Evaluate Subcommand
    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    evaluate_sub = evaluate_parser.add_subparsers(dest="evaluate_cmd")

    # evaluate result
    e_result = evaluate_sub.add_parser("result", help="Evaluate RAG results with all metrics")
    e_result.add_argument("--dataset", type=str, required=True, help="Dataset name")
    e_result.add_argument("--method", type=str, required=True,
                          choices=["naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"],
                          help="Method to evaluate")
    e_result.add_argument("--retriever-type", type=str, default="graph",
                          choices=["naive", "graph"],
                          help="Base retriever for iterative_rag (default: graph)")
    e_result.add_argument("--no-resume", action="store_true", help="Don't resume from existing")
    e_result.add_argument("--skip-llm", action="store_true", help="Skip LLM-based evaluation (faster)")
    e_result.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls (default: 10)")

    # evaluate semantic
    e_sem = evaluate_sub.add_parser("semantic", help="Evaluate embedding quality")
    e_sem.add_argument("--dataset", type=str, required=True, help="Dataset name")
    e_sem.add_argument("--hubness_k", type=int, default=10, help="K for hubness calculation")

    # evaluate structure
    e_struct = evaluate_sub.add_parser("structure", help="Evaluate graph structure")
    e_struct.add_argument("--dataset", type=str, required=True, help="Dataset name")

    # Pipeline Subcommand
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    pipeline_parser.add_argument("--method", type=str, default="graph",
                                 choices=["naive", "graph", "hybrid", "iterative", "llm_direct", "all"],
                                 help="RAG method to use")
    pipeline_parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing")
    pipeline_parser.add_argument("--skip-process", action="store_true", help="Skip preprocessing")
    pipeline_parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    try:
        if args.command == "process":
            if args.process_cmd == "embedding":
                return cmd_process_embedding(args)
            elif args.process_cmd == "index":
                return cmd_process_index(args)
            elif args.process_cmd == "graph":
                return cmd_process_graph(args)
            elif args.process_cmd == "all":
                return cmd_process_all(args)
            else:
                process_parser.print_help()

        elif args.command == "retrieve":
            if args.retrieve_cmd == "naive":
                return cmd_retrieve_naive(args)
            elif args.retrieve_cmd == "graph":
                return cmd_retrieve_graph(args)
            elif args.retrieve_cmd == "hybrid":
                return cmd_retrieve_hybrid(args)
            elif args.retrieve_cmd == "iterative":
                return cmd_retrieve_iterative(args)
            elif args.retrieve_cmd == "llm_direct":
                return cmd_retrieve_llm_direct(args)
            elif args.retrieve_cmd == "retry":
                return cmd_retrieve_retry_null(args)
            else:
                retrieve_parser.print_help()

        elif args.command == "evaluate":
            if args.evaluate_cmd == "result":
                return cmd_evaluate_result(args)
            elif args.evaluate_cmd == "semantic":
                return cmd_evaluate_semantic(args)
            elif args.evaluate_cmd == "structure":
                return cmd_evaluate_structure(args)
            else:
                evaluate_parser.print_help()

        elif args.command == "pipeline":
            return cmd_pipeline(args)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n[Interrupted] Operation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n[Error] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
