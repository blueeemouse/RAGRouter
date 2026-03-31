"""
Hidden States Extractor

This module extracts hidden states from LLM during the prefill phase
using vLLM's speculative decoding infrastructure.

Technical Details:
- Uses vLLM's extract_hidden_states speculative method
- Extracts hidden states from specified layers
- Aggregates results (mean, last, last_k_mean) to reduce storage
- Saves results in safetensors format with float16 precision
"""
import os
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from Config.HiddenStatesConfig import HiddenStatesConfig
from Config.PathConfig import PathConfig
from RAGCore.Prompt.PromptTemplate import PromptTemplate


@dataclass
class HiddenStatesResult:
    """Container for hidden states extraction result"""
    sample_id: str
    token_ids: torch.Tensor
    mean_hidden: torch.Tensor      # [num_layers, hidden_size]
    last_hidden: torch.Tensor      # [num_layers, hidden_size]
    last_5_mean: torch.Tensor      # [num_layers, hidden_size]
    seq_len: int


class HiddenStatesExtractor:
    """Extract hidden states from LLM during prefill phase

    This class uses vLLM's speculative decoding infrastructure to extract
    hidden states from specified layers during the prefill phase.

    Usage:
        extractor = HiddenStatesExtractor()
        results = extractor.extract_batch(questions)
        extractor.save_results(results, output_dir)
    """

    def __init__(
        self,
        model_path: str = None,
        layer_ids: List[int] = None,
        max_model_len: int = None,
        gpu_memory_utilization: float = None,
    ):
        """Initialize the hidden states extractor

        Args:
            model_path: Path to the model (default: from config)
            layer_ids: Layer indices to extract (default: from config)
            max_model_len: Maximum model sequence length (default: from config)
            gpu_memory_utilization: GPU memory utilization (default: from config)
        """
        self.model_path = model_path or HiddenStatesConfig.MODEL_PATH
        self.layer_ids = layer_ids or HiddenStatesConfig.LAYER_IDS
        self.max_model_len = max_model_len or HiddenStatesConfig.MAX_MODEL_LEN
        self.gpu_memory_utilization = gpu_memory_utilization or HiddenStatesConfig.GPU_MEMORY_UTILIZATION

        self.llm = None
        self.tokenizer = None
        self._initialized = False

    def _init_vllm(self):
        """Initialize vLLM engine with hidden states extraction config"""
        if self._initialized:
            return

        from vllm import LLM, SamplingParams

        # Create temporary directory for hidden states storage
        self._temp_dir = tempfile.mkdtemp()

        # Initialize vLLM with speculative config for hidden states extraction
        self.llm = LLM(
            model=self.model_path,
            quantization=HiddenStatesConfig.QUANTIZATION,
            dtype=HiddenStatesConfig.DTYPE,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            seed=HiddenStatesConfig.SEED,  # Fixed seed for reproducibility
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": self.layer_ids,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": self._temp_dir,
                },
            },
        )

        # Get tokenizer for text processing
        self.tokenizer = self.llm.get_tokenizer()

        self._initialized = True
        print(f"Initialized vLLM engine for hidden states extraction")
        print(f"  Model: {self.model_path}")
        print(f"  Layers: {self.layer_ids}")
        print(f"  Temp dir: {self._temp_dir}")

    def build_prompt(self, question: str) -> str:
        """Build full prompt from question using QA_SYSTEM prompt

        This uses the same system prompt as LLMDirect to simulate
        the scenario where LLM answers without external retrieval.

        Args:
            question: The question text

        Returns:
            Full prompt string (system + user)
        """
        messages = PromptTemplate.get_qa_messages(question)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def extract_single(self, question: str, sample_id: str) -> Optional[HiddenStatesResult]:
        """Extract hidden states for a single question

        Args:
            question: The question text
            sample_id: Sample identifier

        Returns:
            HiddenStatesResult or None if extraction failed
        """
        from vllm import SamplingParams

        self._init_vllm()

        try:
            # Build prompt
            prompt = self.build_prompt(question)

            # Generate with max_tokens=1 (minimal generation for prefill)
            # Set temperature=0 for greedy decoding (ensures reproducibility)
            sampling_params = SamplingParams(
                max_tokens=1,
                temperature=HiddenStatesConfig.TEMPERATURE
            )
            outputs = self.llm.generate([prompt], sampling_params)

            if not outputs:
                print(f"Warning: No output for sample {sample_id}")
                return None

            output = outputs[0]

            # Get hidden states path from output
            hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
            if hidden_states_path is None:
                print(f"Warning: No hidden states path for sample {sample_id}")
                return None

            # Load hidden states from safetensors
            with safe_open(hidden_states_path, framework="pt") as f:
                token_ids = f.get_tensor("token_ids")
                hidden_states = f.get_tensor("hidden_states")

            # hidden_states shape: [num_layers, seq_len, hidden_size]
            # Aggregate to reduce storage

            # Mean pooling over all tokens
            mean_hidden = hidden_states.mean(dim=1)  # [num_layers, hidden_size]

            # Last token
            last_hidden = hidden_states[:, -1, :]  # [num_layers, hidden_size]

            # Last 5 tokens mean (or all if seq_len < 5)
            seq_len = hidden_states.shape[1]
            last_k = min(5, seq_len)
            last_5_mean = hidden_states[:, -last_k:, :].mean(dim=1)  # [num_layers, hidden_size]

            return HiddenStatesResult(
                sample_id=sample_id,
                token_ids=token_ids,
                mean_hidden=mean_hidden,
                last_hidden=last_hidden,
                last_5_mean=last_5_mean,
                seq_len=seq_len,
            )

        except Exception as e:
            print(f"Error extracting hidden states for {sample_id}: {e}")
            return None

    def extract_batch(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[HiddenStatesResult]:
        """Extract hidden states for a batch of questions

        Args:
            questions: List of question dicts with 'id' and 'question' keys
            show_progress: Whether to show progress bar

        Returns:
            List of HiddenStatesResult
        """
        self._init_vllm()

        results = []
        iterator = tqdm(questions, desc="Extracting hidden states") if show_progress else questions

        for q_data in iterator:
            sample_id = q_data.get("id")
            question = q_data.get("question")

            if not sample_id or not question:
                continue

            result = self.extract_single(question, sample_id)
            if result:
                results.append(result)

        return results

    def save_result(self, result: HiddenStatesResult, output_path: str):
        """Save a single result to safetensors file

        Args:
            result: HiddenStatesResult to save
            output_path: Path to save the .safetensors file
        """
        # Convert to float16 for storage efficiency
        tensors = {
            "token_ids": result.token_ids.to(torch.int64),
            "mean_hidden": result.mean_hidden.to(torch.float16),
            "last_hidden": result.last_hidden.to(torch.float16),
            "last_5_mean": result.last_5_mean.to(torch.float16),
        }

        save_file(tensors, output_path)

    def save_results(
        self,
        results: List[HiddenStatesResult],
        output_dir: str,
        save_metadata: bool = True
    ):
        """Save all results to directory

        Args:
            results: List of HiddenStatesResult
            output_dir: Directory to save results
            save_metadata: Whether to save metadata.json
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save each result
        for result in results:
            output_path = os.path.join(output_dir, f"{result.sample_id}.safetensors")
            self.save_result(result, output_path)

        # Save metadata
        if save_metadata:
            metadata = {
                "model_path": self.model_path,
                "layer_ids": self.layer_ids,
                "hidden_size": HiddenStatesConfig.HIDDEN_SIZE,
                "dtype": HiddenStatesConfig.DTYPE,
                "aggregation_methods": HiddenStatesConfig.AGGREGATION_METHODS,
                "num_samples": len(results),
            }
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"Saved {len(results)} results to {output_dir}")

    def load_progress(self, progress_path: str) -> set:
        """Load processed sample IDs from progress file

        Args:
            progress_path: Path to progress.json

        Returns:
            Set of processed sample IDs
        """
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                data = json.load(f)
                return set(data.get("processed_ids", []))
        return set()

    def save_progress(self, processed_ids: set, progress_path: str):
        """Save progress to file

        Args:
            processed_ids: Set of processed sample IDs
            progress_path: Path to progress.json
        """
        with open(progress_path, "w") as f:
            json.dump({"processed_ids": list(processed_ids)}, f, indent=2)

    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)


def load_hidden_states(file_path: str) -> Dict[str, torch.Tensor]:
    """Load hidden states from a safetensors file

    Args:
        file_path: Path to .safetensors file

    Returns:
        Dictionary with tensors:
        - token_ids: [seq_len]
        - mean_hidden: [num_layers, hidden_size]
        - last_hidden: [num_layers, hidden_size]
        - last_5_mean: [num_layers, hidden_size]
    """
    tensors = {}
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


if __name__ == "__main__":
    # Test the extractor
    print("Testing HiddenStatesExtractor...")

    extractor = HiddenStatesExtractor()

    test_questions = [
        {"id": "test_001", "question": "What is the capital of France?"},
    ]

    results = extractor.extract_batch(test_questions)

    if results:
        print(f"\nExtracted {len(results)} results")
        print(f"Sample ID: {results[0].sample_id}")
        print(f"Token IDs shape: {results[0].token_ids.shape}")
        print(f"Mean hidden shape: {results[0].mean_hidden.shape}")
        print(f"Seq length: {results[0].seq_len}")
