"""
Result Evaluation Processor

Evaluates RAG results using LLM scoring:
- llm_retrieval_score: How relevant is the retrieved context for answering the question
- llm_ans_score: How correct is the predicted answer compared to ground truth
- semantic_f1: BERTScore-based semantic similarity between prediction and ground truth

Input:
- RetrievalResultData/{method}/{model}/{dataset}/answer.jsonl
- RetrievalResultData/{method}/{model}/{dataset}/retrieval.jsonl
- RawData/{dataset}/Question.json (ground truth)

Output:
- EvaluationData/ResultEvaluation/{model}/{dataset}/{method}.json
"""
import json
import os
import asyncio
import numpy as np
import spacy
from typing import Dict, List, Any, Optional
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

from Config.LLMConfig import LLMConfig
from Config.PathConfig import PathConfig
from Config.EmbConfig import EmbConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate
from BenchCore.Evaluation.ResultEvaluation.EvaluationSave import EvaluationSaver


class ResultEvaluator:
    """Evaluate RAG results with LLM scoring"""

    def __init__(self):
        """Initialize evaluator with LLM client"""
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup LLM client for evaluation (sync and async)"""
        # 评估专用 LLM: Llama 3.1 70B AWQ-INT4 (独立于 LLMConfig，避免影响 retrieve 阶段)
        # 端口 8002，独立于提取三元组的 8000 端口，两者可以同时运行
        self.model_name = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
        self.llm_client = OpenAI(
            api_key="not-needed",
            base_url="http://localhost:8002/v1"
        )
        self.async_llm_client = AsyncOpenAI(
            api_key="not-needed",
            base_url="http://localhost:8002/v1"
        )
        print(f"Initialized Evaluator LLM: {self.model_name}")
        # 原代码: 从 LLMConfig 读取配置
        # model_config = LLMConfig.get_model_config()
        # self.model_name = model_config["model"]
        # self.llm_client = OpenAI(
        #     api_key=model_config["api_key"],
        #     base_url=model_config["base_url"]
        # )
        # self.async_llm_client = AsyncOpenAI(
        #     api_key=model_config["api_key"],
        #     base_url=model_config["base_url"]
        # )
        # print(f"Initialized Evaluator LLM: {self.model_name}")

    def load_questions(self, dataset_name: str) -> Dict[int, Dict[str, Any]]:
        """Load questions with ground truth answers

        Returns:
            Dict mapping id -> {question, answer}
        """
        question_path = PathConfig.get_question_path(dataset_name)

        with open(question_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                questions = json.load(f)
            else:
                questions = [json.loads(line) for line in f if line.strip()]

        return {q['id']: q for q in questions}

    def load_answers(self, model_name: str, dataset_name: str, method: str,
                     retriever_type: str = "naive") -> Dict[int, str]:
        """Load RAG answers

        Args:
            model_name: Model name (e.g., "deepseek-chat")
            dataset_name: Dataset name
            method: One of "naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")

        Returns:
            Dict mapping id -> rag_answer
        """
        method_lower = method.lower().replace("-", "_")

        if method_lower in ["naive_rag", "naiverag"]:
            answer_path = PathConfig.get_naive_rag_path(model_name, dataset_name)
        elif method_lower in ["graph_rag", "graphrag"]:
            answer_path = PathConfig.get_graph_rag_path(model_name, dataset_name)
        elif method_lower in ["hybrid_rag", "hybridrag"]:
            answer_path = PathConfig.get_hybrid_rag_path(model_name, dataset_name)
        elif method_lower in ["iterative_rag", "iterativerag"]:
            answer_path = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        elif method_lower in ["llm_direct", "llmdirect"]:
            answer_path = PathConfig.get_llm_direct_path(model_name, dataset_name)
        else:
            raise ValueError(f"Unknown method: {method}")

        answers = {}
        if not os.path.exists(answer_path):
            print(f"Warning: Answer file not found: {answer_path}")
            return answers

        with open(answer_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Handle different answer field names
                    answer = record.get('rag_answer') or record.get('answer') or record.get('final_answer', '')
                    answers[record['id']] = answer

        return answers

    def load_retrievals(self, model_name: str, dataset_name: str, method: str,
                        retriever_type: str = "naive") -> Dict[int, List[str]]:
        """Load retrieval results (source sentences)

        Args:
            model_name: Model name (e.g., "deepseek-chat")
            dataset_name: Dataset name
            method: One of "naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")

        Returns:
            Dict mapping id -> list of source sentences
        """
        method_lower = method.lower().replace("-", "_")

        # LLM Direct has no retrieval
        if method_lower in ["llm_direct", "llmdirect"]:
            return {}

        if method_lower in ["naive_rag", "naiverag"]:
            base_path = PathConfig.get_naive_rag_path(model_name, dataset_name)
        elif method_lower in ["graph_rag", "graphrag"]:
            base_path = PathConfig.get_graph_rag_path(model_name, dataset_name)
        elif method_lower in ["hybrid_rag", "hybridrag"]:
            base_path = PathConfig.get_hybrid_rag_path(model_name, dataset_name)
        elif method_lower in ["iterative_rag", "iterativerag"]:
            base_path = PathConfig.get_iterative_rag_path(model_name, dataset_name, retriever_type)
        else:
            raise ValueError(f"Unknown method: {method}")

        retrieval_path = base_path.replace("answer.jsonl", "retrieval.jsonl")

        retrievals = {}
        if not os.path.exists(retrieval_path):
            print(f"Warning: Retrieval file not found: {retrieval_path}")
            return retrievals

        with open(retrieval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    qid = record['id']

                    # Handle different retrieval formats
                    if 'source_sentences' in record:
                        # GraphRAG format - can be list of dicts with 'text' field or list of strings
                        source_sentences = record['source_sentences']
                        if source_sentences and isinstance(source_sentences[0], dict):
                            retrievals[qid] = [s.get('text', '') for s in source_sentences]
                        else:
                            retrievals[qid] = source_sentences
                    elif 'chunks' in record:
                        # NaiveRAG format
                        chunks = record['chunks']
                        if chunks and isinstance(chunks[0], dict):
                            retrievals[qid] = [c.get('text', '') for c in chunks]
                        else:
                            retrievals[qid] = chunks if chunks else []
                    elif 'retrieved_chunks' in record:
                        # HybridRAG format
                        chunks = record['retrieved_chunks']
                        if chunks and isinstance(chunks[0], dict):
                            retrievals[qid] = [c.get('text', '') for c in chunks]
                        else:
                            retrievals[qid] = chunks if chunks else []
                    elif 'history' in record:
                        # IterativeRAG format - get retrieval from last round
                        history = record.get('history', [])
                        if history:
                            last_round = history[-1]
                            context = last_round.get('context', '')
                            retrievals[qid] = [context] if context else []
                        else:
                            retrievals[qid] = []
                    else:
                        retrievals[qid] = []

        return retrievals

    def evaluate_semantic_f1(self, ground_truth: str, predicted: str) -> float:
        """Evaluate semantic similarity using BERTScore

        Args:
            ground_truth: Ground truth answer
            predicted: Predicted answer

        Returns:
            F1 score from 0-1 (returns 0.0 for refusal answers)
        """
        if not predicted or not ground_truth:
            return 0.0

        # Return 0 for refusal answers
        if self._is_refusal(predicted):
            return 0.0

        try:
            _, _, f1 = bert_score(
                [predicted],
                [ground_truth],
                model_type=EmbConfig.BERTSCORE_MODEL,
                lang=EmbConfig.BERTSCORE_LANG,
                device=EmbConfig.BERTSCORE_DEVICE,
                verbose=False
            )
            return float(f1[0])
        except Exception as e:
            print(f"Warning: Failed to compute semantic F1: {e}")
            return 0.0

    def evaluate_semantic_f1_batch(self, ground_truths: List[str], predictions: List[str]) -> List[float]:
        """Evaluate semantic similarity for a batch using BERTScore

        Args:
            ground_truths: List of ground truth answers
            predictions: List of predicted answers

        Returns:
            List of F1 scores from 0-1 (0.0 for refusal answers)
        """
        if not predictions or not ground_truths:
            return []

        try:
            _, _, f1 = bert_score(
                predictions,
                ground_truths,
                model_type=EmbConfig.BERTSCORE_MODEL,
                lang=EmbConfig.BERTSCORE_LANG,
                device=EmbConfig.BERTSCORE_DEVICE,
                verbose=True
            )
            scores = f1.tolist()

            # Set refusal answers to 0.0
            for i, pred in enumerate(predictions):
                if self._is_refusal(pred):
                    scores[i] = 0.0

            return scores
        except Exception as e:
            print(f"Warning: Failed to compute semantic F1 batch: {e}")
            return [0.0] * len(predictions)

    def evaluate_soft_coverage(self, ground_truth: str, predicted: str) -> float:
        """Evaluate soft coverage using SentenceTransformer

        Soft COV: Mean of max similarities between GT (as fact) and prediction sentences.
        Measures how well the prediction covers the ground truth semantically.

        Args:
            ground_truth: Ground truth answer (treated as a single fact)
            predicted: Predicted answer

        Returns:
            Soft coverage score from 0-1 (returns 0.0 for refusal answers)
        """
        if not ground_truth or not predicted or not predicted.strip():
            return 0.0

        # Return 0 for refusal answers
        if self._is_refusal(predicted):
            return 0.0

        try:
            # Lazy load spacy and sentence transformer
            if not hasattr(self, '_nlp'):
                self._nlp = spacy.load("en_core_web_sm")
            if not hasattr(self, '_st_model'):
                self._st_model = SentenceTransformer(
                    EmbConfig.SENTENCE_TRANSFORMER_MODEL,
                    device=EmbConfig.SENTENCE_TRANSFORMER_DEVICE
                )

            # Split prediction into sentences using spacy
            doc = self._nlp(predicted)
            pred_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not pred_sentences:
                pred_sentences = [predicted]

            # Treat GT as a single fact
            facts = [ground_truth]

            # Compute embeddings
            fact_embs = self._st_model.encode(facts, normalize_embeddings=True)
            pred_embs = self._st_model.encode(pred_sentences, normalize_embeddings=True)

            # Compute cosine similarity matrix: facts x pred_sentences
            sim_matrix = np.dot(fact_embs, pred_embs.T)

            # For each fact, find max similarity with any prediction sentence
            max_sims = sim_matrix.max(axis=1)

            # Soft COV: mean of max similarities
            soft_cov = float(max_sims.mean())

            return soft_cov

        except Exception as e:
            print(f"Warning: Failed to compute soft coverage: {e}")
            return 0.0

    def _is_refusal(self, answer: str) -> bool:
        """Check if answer is a refusal to answer"""
        if not answer:
            return True
        answer_lower = answer.lower()
        return "i cannot answer" in answer_lower or "cannot answer" in answer_lower

    def evaluate_faithfulness(self, predicted: str, retrieval_texts: List[str],
                               threshold: float = None) -> Optional[float]:
        """Evaluate faithfulness score using SentenceTransformer

        Faithfulness Score (FS): Measures how well the answer is grounded in retrieval.
        Each answer sentence is checked if it has support from retrieval content.

        Hard FS = (number of faithful sentences) / (total sentences)
        A sentence is faithful if max_sim(sentence, any_retrieval_sentence) >= threshold

        Args:
            predicted: Predicted answer
            retrieval_texts: List of retrieval content (chunks or source_sentences)
            threshold: Similarity threshold for Hard FS (default: from EmbConfig)

        Returns:
            Faithfulness score from 0-1, or None if answer is refusal
        """
        # Skip if answer is refusal
        if self._is_refusal(predicted):
            return None

        if not retrieval_texts:
            return 0.0

        if threshold is None:
            threshold = EmbConfig.FAITHFULNESS_THRESHOLD

        try:
            # Lazy load spacy and sentence transformer
            if not hasattr(self, '_nlp'):
                self._nlp = spacy.load("en_core_web_sm")
            if not hasattr(self, '_st_model'):
                self._st_model = SentenceTransformer(
                    EmbConfig.SENTENCE_TRANSFORMER_MODEL,
                    device=EmbConfig.SENTENCE_TRANSFORMER_DEVICE
                )

            # Split answer into sentences
            doc = self._nlp(predicted)
            pred_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not pred_sentences:
                pred_sentences = [predicted]

            # Split retrieval texts into sentences
            retrieval_sentences = []
            for text in retrieval_texts:
                if text:
                    ret_doc = self._nlp(text)
                    retrieval_sentences.extend([sent.text.strip() for sent in ret_doc.sents if sent.text.strip()])

            if not retrieval_sentences:
                return 0.0

            # Compute embeddings
            pred_embs = self._st_model.encode(pred_sentences, normalize_embeddings=True)
            ret_embs = self._st_model.encode(retrieval_sentences, normalize_embeddings=True)

            # Compute cosine similarity matrix: pred_sentences x retrieval_sentences
            sim_matrix = np.dot(pred_embs, ret_embs.T)

            # For each answer sentence, find max similarity with any retrieval sentence
            max_sims = sim_matrix.max(axis=1)

            # Hard FS: count sentences with max_sim >= threshold
            faithful_count = (max_sims >= threshold).sum()
            hard_fs = float(faithful_count / len(pred_sentences))

            return hard_fs

        except Exception as e:
            print(f"Warning: Failed to compute faithfulness: {e}")
            return 0.0

    def evaluate_faithfulness_soft(self, predicted: str, retrieval_texts: List[str]) -> Optional[float]:
        """Evaluate soft faithfulness score (mean of max similarities)

        Args:
            predicted: Predicted answer
            retrieval_texts: List of retrieval content

        Returns:
            Soft faithfulness score from 0-1, or None if answer is refusal
        """
        # Skip if answer is refusal
        if self._is_refusal(predicted):
            return None

        if not retrieval_texts:
            return 0.0

        try:
            # Lazy load spacy and sentence transformer
            if not hasattr(self, '_nlp'):
                self._nlp = spacy.load("en_core_web_sm")
            if not hasattr(self, '_st_model'):
                self._st_model = SentenceTransformer(
                    EmbConfig.SENTENCE_TRANSFORMER_MODEL,
                    device=EmbConfig.SENTENCE_TRANSFORMER_DEVICE
                )

            # Split answer into sentences
            doc = self._nlp(predicted)
            pred_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not pred_sentences:
                pred_sentences = [predicted]

            # Split retrieval texts into sentences
            retrieval_sentences = []
            for text in retrieval_texts:
                if text:
                    ret_doc = self._nlp(text)
                    retrieval_sentences.extend([sent.text.strip() for sent in ret_doc.sents if sent.text.strip()])

            if not retrieval_sentences:
                return 0.0

            # Compute embeddings
            pred_embs = self._st_model.encode(pred_sentences, normalize_embeddings=True)
            ret_embs = self._st_model.encode(retrieval_sentences, normalize_embeddings=True)

            # Compute cosine similarity matrix
            sim_matrix = np.dot(pred_embs, ret_embs.T)

            # Soft FS: mean of max similarities
            max_sims = sim_matrix.max(axis=1)
            soft_fs = float(max_sims.mean())

            return soft_fs

        except Exception as e:
            print(f"Warning: Failed to compute soft faithfulness: {e}")
            return 0.0

    def classify_answer(self, question: str, ground_truth: str, predicted: str) -> Dict[str, str]:
        """Classify answer correctness using LLM

        Args:
            question: The question
            ground_truth: Ground truth answer
            predicted: Predicted answer

        Returns:
            Dict with keys: label (correct/incorrect/incomplete), reason
        """
        if not predicted or not predicted.strip():
            return {"label": "incomplete", "reason": "No answer provided."}

        try:
            messages = PromptTemplate.get_answer_label_messages(question, ground_truth, predicted)
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            # Validate label
            if result.get("label") not in ["correct", "incorrect", "incomplete"]:
                result["label"] = "incomplete"

            return result

        except Exception as e:
            print(f"Warning: Failed to classify answer: {e}")
            return {"label": "incomplete", "reason": "Classification failed."}

    async def classify_answer_async(self, question: str, ground_truth: str, predicted: str) -> Dict[str, str]:
        """Classify answer correctness using LLM (async version)

        Args:
            question: The question
            ground_truth: Ground truth answer
            predicted: Predicted answer

        Returns:
            Dict with keys: label (correct/incorrect/incomplete), reason
        """
        if not predicted or not predicted.strip():
            return {"label": "incomplete", "reason": "No answer provided."}

        try:
            messages = PromptTemplate.get_answer_label_messages(question, ground_truth, predicted)
            response = await self.async_llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            # Validate label
            if result.get("label") not in ["correct", "incorrect", "incomplete"]:
                result["label"] = "incomplete"

            return result

        except Exception as e:
            print(f"Warning: Failed to classify answer: {e}")
            return {"label": "incomplete", "reason": "Classification failed."}

    def classify_answers_comparison(
        self,
        question: str,
        ground_truth: str,
        naive_answer: str,
        graph_answer: str,
        qid: Any,
        qtype: str = "unknown"
    ) -> Dict[str, Any]:
        """Classify both naive and graph answers for comparison

        Args:
            question: The question
            ground_truth: Ground truth answer
            naive_answer: Answer from Naive RAG
            graph_answer: Answer from Graph RAG
            qid: Question ID
            qtype: Question type

        Returns:
            Dict with keys: id, type, naive, graph
        """
        naive_result = self.classify_answer(question, ground_truth, naive_answer)
        graph_result = self.classify_answer(question, ground_truth, graph_answer)

        return {
            "id": qid,
            "type": qtype,
            "naive": naive_result,
            "graph": graph_result
        }

    def evaluate(self, model_name: str, dataset_name: str, method: str,
                 resume: bool = True) -> List[Dict[str, Any]]:
        """Evaluate all results for a dataset and method

        Args:
            model_name: Model name (e.g., "deepseek-chat")
            dataset_name: Dataset name
            method: "naive_rag" or "graph_rag"
            resume: If True, skip already evaluated questions

        Returns:
            List of evaluation results
        """
        print(f"Evaluating {method} on {dataset_name} (model: {model_name})")

        # Load data
        questions = self.load_questions(dataset_name)
        answers = self.load_answers(model_name, dataset_name, method)

        print(f"Loaded {len(questions)} questions, {len(answers)} answers")

        # Load existing results if resuming
        results = []
        evaluated_ids = set()

        if resume:
            results = EvaluationSaver.load(model_name, dataset_name, method)
            if results:
                evaluated_ids = {r['id'] for r in results}
                print(f"Resuming: Found {len(evaluated_ids)} already evaluated")

        # Evaluate each question
        to_evaluate = [qid for qid in questions.keys() if qid in answers and qid not in evaluated_ids]

        if not to_evaluate:
            print("All questions already evaluated!")
            return results

        print(f"Evaluating {len(to_evaluate)} questions...")

        for qid in tqdm(to_evaluate, desc="Evaluating"):
            q_data = questions[qid]
            question = q_data['question']
            ground_truth = q_data['answer']
            predicted = answers.get(qid, '')

            # Evaluate using label classification
            label_result = self.classify_answer(question, ground_truth, predicted)

            result = {
                "id": qid,
                "label": label_result.get("label", "incomplete"),
                "reason": label_result.get("reason", "")
            }
            results.append(result)

            # Save incrementally
            if resume:
                EvaluationSaver.save(results, model_name, dataset_name, method)

        # Final save
        EvaluationSaver.save(results, model_name, dataset_name, method)

        return results

    def _ensure_models_loaded(self):
        """Ensure spacy and sentence transformer models are loaded"""
        if not hasattr(self, '_nlp'):
            print("Loading spacy model...")
            self._nlp = spacy.load("en_core_web_sm")
        if not hasattr(self, '_st_model'):
            print("Loading SentenceTransformer model...")
            self._st_model = SentenceTransformer(
                EmbConfig.SENTENCE_TRANSFORMER_MODEL,
                device=EmbConfig.SENTENCE_TRANSFORMER_DEVICE
            )

    def _compute_faithfulness_fast(self, predicted: str, retrieval_texts: List[str]) -> tuple:
        """Fast faithfulness computation without sentence splitting for retrieval

        Returns:
            (hard_score, soft_score) or (None, None) if refusal
        """
        if self._is_refusal(predicted):
            return None, None

        if not retrieval_texts:
            return 0.0, 0.0

        try:
            # Split answer into sentences
            doc = self._nlp(predicted)
            pred_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not pred_sentences:
                pred_sentences = [predicted]

            # Use retrieval texts directly (no sentence splitting - much faster)
            retrieval_texts_clean = [t for t in retrieval_texts if t and t.strip()]
            if not retrieval_texts_clean:
                return 0.0, 0.0

            # Compute embeddings
            pred_embs = self._st_model.encode(pred_sentences, normalize_embeddings=True)
            ret_embs = self._st_model.encode(retrieval_texts_clean, normalize_embeddings=True)

            # Compute similarity
            sim_matrix = np.dot(pred_embs, ret_embs.T)
            max_sims = sim_matrix.max(axis=1)

            # Hard FS
            threshold = EmbConfig.FAITHFULNESS_THRESHOLD
            hard_fs = float((max_sims >= threshold).sum() / len(pred_sentences))
            # Soft FS
            soft_fs = float(max_sims.mean())

            return hard_fs, soft_fs

        except Exception as e:
            print(f"Warning: Failed to compute faithfulness: {e}")
            return 0.0, 0.0

    def _compute_soft_coverage_fast(self, ground_truth: str, predicted: str) -> float:
        """Fast soft coverage computation"""
        if not ground_truth or not predicted or not predicted.strip():
            return 0.0
        if self._is_refusal(predicted):
            return 0.0

        try:
            # Split prediction into sentences
            doc = self._nlp(predicted)
            pred_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not pred_sentences:
                pred_sentences = [predicted]

            # Compute embeddings
            gt_emb = self._st_model.encode([ground_truth], normalize_embeddings=True)
            pred_embs = self._st_model.encode(pred_sentences, normalize_embeddings=True)

            # Compute similarity
            sim_matrix = np.dot(gt_emb, pred_embs.T)
            return float(sim_matrix.max())

        except Exception as e:
            print(f"Warning: Failed to compute soft coverage: {e}")
            return 0.0

    def evaluate_comparison_batch(
        self,
        data_list: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "dataset"
    ) -> Dict[str, List[Dict]]:
        """Batch evaluate both NaiveRAG and GraphRAG results

        Args:
            data_list: List of dicts with keys:
                - id: question id
                - type: question type
                - ground_truth: ground truth answer
                - naive_ans: NaiveRAG answer
                - graph_ans: GraphRAG answer
                - naive_ret: NaiveRAG retrieval texts (list)
                - graph_ret: GraphRAG retrieval texts (list)
            output_dir: Directory to save evaluation results
            dataset_name: Name of dataset (for display)

        Returns:
            Dict with keys: semantic_f1, soft_coverage, faithfulness
        """
        os.makedirs(output_dir, exist_ok=True)

        # Extract lists for batch processing
        ids = [d['id'] for d in data_list]
        types = [d.get('type', 'unknown') for d in data_list]
        ground_truths = [d['ground_truth'] for d in data_list]
        naive_answers = [d['naive_ans'] for d in data_list]
        graph_answers = [d['graph_ans'] for d in data_list]

        # ============ Semantic F1 (BERTScore batch) ============
        print("\n[1/3] Computing Semantic F1 (BERTScore batch)...")
        print("  Computing naive BERTScore...")
        naive_f1_scores = self.evaluate_semantic_f1_batch(ground_truths, naive_answers)
        print("  Computing graph BERTScore...")
        graph_f1_scores = self.evaluate_semantic_f1_batch(ground_truths, graph_answers)

        semantic_f1_results = [
            {"id": ids[i], "type": types[i], "naive_score": naive_f1_scores[i], "graph_score": graph_f1_scores[i]}
            for i in range(len(data_list))
        ]

        # ============ Soft Coverage & Faithfulness (Fast mode) ============
        print("\n[2/3] Computing Soft Coverage...")
        print("[3/3] Computing Faithfulness...")

        self._ensure_models_loaded()

        soft_coverage_results = []
        faithfulness_results = []

        for i, d in enumerate(tqdm(data_list, desc="Coverage & Faithfulness")):
            gt = d['ground_truth']
            naive_ans = d['naive_ans']
            graph_ans = d['graph_ans']
            naive_ret = d.get('naive_ret', [])
            graph_ret = d.get('graph_ret', [])

            # Soft Coverage (fast)
            naive_cov = self._compute_soft_coverage_fast(gt, naive_ans)
            graph_cov = self._compute_soft_coverage_fast(gt, graph_ans)
            soft_coverage_results.append({
                "id": ids[i], "type": types[i],
                "naive_score": naive_cov, "graph_score": graph_cov
            })

            # Faithfulness (fast - no retrieval sentence splitting)
            naive_hard, naive_soft = self._compute_faithfulness_fast(naive_ans, naive_ret)
            graph_hard, graph_soft = self._compute_faithfulness_fast(graph_ans, graph_ret)
            faithfulness_results.append({
                "id": ids[i], "type": types[i],
                "hard": {"naive": naive_hard, "graph": graph_hard},
                "soft": {"naive": naive_soft, "graph": graph_soft}
            })

        # ============ Save Results ============
        with open(os.path.join(output_dir, "semantic_f1.json"), 'w', encoding='utf-8') as f:
            json.dump(semantic_f1_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "soft_coverage.json"), 'w', encoding='utf-8') as f:
            json.dump(soft_coverage_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "faithfulness.json"), 'w', encoding='utf-8') as f:
            json.dump(faithfulness_results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_dir}")

        return {
            "semantic_f1": semantic_f1_results,
            "soft_coverage": soft_coverage_results,
            "faithfulness": faithfulness_results
        }


    def evaluate_all_metrics(
        self,
        model_name: str,
        dataset_name: str,
        method: str,
        retriever_type: str = "naive",
        resume: bool = True,
        skip_llm: bool = False,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Evaluate all metrics for a dataset and method

        Computes: semantic_f1, coverage, faithfulness_hard, faithfulness_soft, llm_label

        Args:
            model_name: Model name (e.g., "deepseek-chat")
            dataset_name: Dataset name
            method: One of "naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"
            retriever_type: For iterative_rag, the base retriever type ("naive" or "graph")
            resume: If True, skip already evaluated questions
            skip_llm: If True, skip LLM-based evaluation (faster)
            max_concurrent: Maximum concurrent LLM calls for parallel evaluation (default: 10)

        Returns:
            List of evaluation results with all metrics
        """
        print(f"Evaluating {method} on {dataset_name} (model: {model_name})")

        # Load data
        questions = self.load_questions(dataset_name)
        answers = self.load_answers(model_name, dataset_name, method, retriever_type)
        retrievals = self.load_retrievals(model_name, dataset_name, method, retriever_type)

        print(f"Loaded {len(questions)} questions, {len(answers)} answers, {len(retrievals)} retrievals")

        # Load existing results if resuming
        results = []
        evaluated_ids = set()

        if resume:
            results = EvaluationSaver.load(model_name, dataset_name, method, retriever_type)
            if results:
                evaluated_ids = {r['id'] for r in results}
                print(f"Resuming: Found {len(evaluated_ids)} already evaluated")

        # Get questions to evaluate
        to_evaluate = [qid for qid in questions.keys() if qid in answers and qid not in evaluated_ids]

        if not to_evaluate:
            print("All questions already evaluated!")
            return results

        print(f"Evaluating {len(to_evaluate)} questions...")

        # Ensure models are loaded for batch processing
        self._ensure_models_loaded()

        # Prepare batch data for BERTScore
        batch_ground_truths = []
        batch_predictions = []
        batch_qids = []

        for qid in to_evaluate:
            q_data = questions[qid]
            ground_truth = q_data['answer']
            predicted = answers.get(qid, '')
            batch_ground_truths.append(ground_truth)
            batch_predictions.append(predicted)
            batch_qids.append(qid)

        # Batch compute semantic F1
        print("\n[1/4] Computing Semantic F1 (BERTScore batch)...")
        semantic_f1_scores = self.evaluate_semantic_f1_batch(batch_ground_truths, batch_predictions)

        # Process each question for other metrics
        print("[2/4] Computing Coverage...")
        print("[3/4] Computing Faithfulness...")

        new_results = []
        for i, qid in enumerate(tqdm(batch_qids, desc="Coverage & Faithfulness")):
            q_data = questions[qid]
            ground_truth = q_data['answer']
            predicted = answers.get(qid, '')
            retrieval_texts = retrievals.get(qid, [])

            # Coverage
            coverage = self._compute_soft_coverage_fast(ground_truth, predicted)

            # Faithfulness (skip for LLM Direct which has no retrieval)
            if retrieval_texts:
                faith_hard, faith_soft = self._compute_faithfulness_fast(predicted, retrieval_texts)
            else:
                faith_hard, faith_soft = None, None

            result = {
                "id": qid,
                "semantic_f1": semantic_f1_scores[i] if i < len(semantic_f1_scores) else 0.0,
                "coverage": coverage,
                "faithfulness_hard": faith_hard,
                "faithfulness_soft": faith_soft,
            }
            new_results.append(result)

        # LLM evaluation (optional, with parallel processing)
        if not skip_llm:
            print("[4/4] Computing LLM Label (parallel)...")

            # Run parallel LLM evaluation
            llm_results = asyncio.run(self._evaluate_llm_labels_async(
                batch_qids, questions, answers, max_concurrent
            ))

            # Merge LLM results into new_results
            for i, (qid, label_result) in enumerate(zip(batch_qids, llm_results)):
                new_results[i]["llm_label"] = label_result.get("label", "incomplete")
                new_results[i]["llm_reason"] = label_result.get("reason", "")
        else:
            print("[4/4] Skipping LLM Label (--skip-llm)")
            for result in new_results:
                result["llm_label"] = None
                result["llm_reason"] = None

        # Combine results
        results.extend(new_results)

        # Final save
        EvaluationSaver.save(results, model_name, dataset_name, method, retriever_type)

        # Print summary
        self._print_evaluation_summary(results)

        return results

    async def _evaluate_llm_labels_async(
        self,
        batch_qids: List[int],
        questions: Dict[int, Dict[str, Any]],
        answers: Dict[int, str],
        max_concurrent: int = 10
    ) -> List[Dict[str, str]]:
        """Evaluate LLM labels in parallel

        Args:
            batch_qids: List of question IDs to evaluate
            questions: Dict mapping id -> question data
            answers: Dict mapping id -> predicted answer
            max_concurrent: Maximum concurrent LLM calls

        Returns:
            List of label results in the same order as batch_qids
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(batch_qids)

        async def evaluate_single(idx: int, qid: int):
            async with semaphore:
                q_data = questions[qid]
                question = q_data['question']
                ground_truth = q_data['answer']
                predicted = answers.get(qid, '')

                label_result = await self.classify_answer_async(question, ground_truth, predicted)
                results[idx] = label_result

        # Create tasks with progress bar
        tasks = [evaluate_single(i, qid) for i, qid in enumerate(batch_qids)]

        # Run with progress bar
        for f in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Label"):
            await f

        return results

    def _print_evaluation_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics for evaluation results"""
        if not results:
            return

        total = len(results)

        # Semantic F1
        f1_scores = [r['semantic_f1'] for r in results if r.get('semantic_f1') is not None]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        # Coverage
        cov_scores = [r['coverage'] for r in results if r.get('coverage') is not None]
        avg_cov = sum(cov_scores) / len(cov_scores) if cov_scores else 0

        # Faithfulness
        faith_hard = [r['faithfulness_hard'] for r in results if r.get('faithfulness_hard') is not None]
        faith_soft = [r['faithfulness_soft'] for r in results if r.get('faithfulness_soft') is not None]
        avg_faith_hard = sum(faith_hard) / len(faith_hard) if faith_hard else 0
        avg_faith_soft = sum(faith_soft) / len(faith_soft) if faith_soft else 0

        # LLM Label
        labels = [r.get('llm_label') for r in results if r.get('llm_label')]
        correct = sum(1 for l in labels if l == 'correct')
        incorrect = sum(1 for l in labels if l == 'incorrect')
        incomplete = sum(1 for l in labels if l == 'incomplete')

        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Total questions: {total}")
        print(f"\nSemantic F1 (BERTScore):  {avg_f1:.4f}")
        print(f"Coverage (Soft):          {avg_cov:.4f}")
        print(f"Faithfulness (Hard):      {avg_faith_hard:.4f}")
        print(f"Faithfulness (Soft):      {avg_faith_soft:.4f}")
        if labels:
            print(f"\nLLM Label:")
            print(f"  Correct:    {correct}/{len(labels)} ({correct/len(labels)*100:.1f}%)")
            print(f"  Incorrect:  {incorrect}/{len(labels)} ({incorrect/len(labels)*100:.1f}%)")
            print(f"  Incomplete: {incomplete}/{len(labels)} ({incomplete/len(labels)*100:.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--method", type=str, required=True,
                        choices=["naive_rag", "graph_rag", "hybrid_rag", "iterative_rag", "llm_direct"],
                        help="Method to evaluate")
    parser.add_argument("--retriever-type", type=str, default="naive",
                        choices=["naive", "graph"],
                        help="Base retriever for iterative_rag (default: naive)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-based evaluation (faster)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent LLM calls (default: 10)")
    args = parser.parse_args()

    evaluator = ResultEvaluator()
    results = evaluator.evaluate_all_metrics(
        model_name=args.model,
        dataset_name=args.dataset,
        method=args.method,
        retriever_type=args.retriever_type,
        resume=not args.no_resume,
        skip_llm=args.skip_llm,
        max_concurrent=args.max_concurrent
    )
