"""
Hidden States Extraction Configuration

This module provides configuration for extracting hidden states from LLM
during the prefill phase. Used for training the RAG Router.
"""
import os


class HiddenStatesConfig:
    """Configuration for hidden states extraction"""

    # Model Configuration
    # Use the same model as LLMDirect (8B AWQ-INT4 on port 8001)
    MODEL_PATH = "/home/lhz/code/model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    MODEL_NAME = "llama-3.1-8b-awq-int4"  # For file paths

    # Quantization
    QUANTIZATION = "awq"

    # Layer Selection
    # Llama-3.1-8B has 32 layers (indices 0-31)
    # Select: shallow (8), middle (16), deep-middle (24), last (31)
    LAYER_IDS = [8, 16, 24, 31]
    NUM_LAYERS = 32  # Total layers in the model

    # Hidden States Dimensions
    HIDDEN_SIZE = 4096  # Llama-3.1-8B hidden dimension

    # Storage Precision (must match model runtime precision)
    # AWQ models use float16 for activations
    DTYPE = "float16"

    # vLLM Engine Configuration
    MAX_MODEL_LEN = 2048
    GPU_MEMORY_UTILIZATION = 0.8
    TENSOR_PARALLEL_SIZE = 1

    # Aggregation Methods
    # Save aggregated results to reduce storage
    AGGREGATION_METHODS = ["mean", "last", "last_5_mean"]

    # Batch Processing
    BATCH_SIZE = 8  # Number of samples to process in one batch

    # Path Configuration
    DATASET_ROOT = "/home/lhz/code/RAGRouter/Dataset"
    HIDDEN_STATES_DIR = os.path.join(DATASET_ROOT, "HiddenStates")

    @staticmethod
    def get_hidden_states_path(dataset_name: str, model_name: str = None) -> str:
        """Get the base path for hidden states storage

        Args:
            dataset_name: Name of the dataset
            model_name: Model name for file paths (default: use config)

        Returns:
            Path: /Dataset/HiddenStates/{dataset}/{model}/
        """
        if model_name is None:
            model_name = HiddenStatesConfig.MODEL_NAME
        return os.path.join(HiddenStatesConfig.HIDDEN_STATES_DIR, dataset_name, model_name)

    @staticmethod
    def get_sample_path(dataset_name: str, sample_id: str, model_name: str = None) -> str:
        """Get path for a single sample's hidden states file

        Args:
            dataset_name: Name of the dataset
            sample_id: Sample ID (e.g., "musique_0000")
            model_name: Model name for file paths

        Returns:
            Path: /Dataset/HiddenStates/{dataset}/{model}/{sample_id}.safetensors
        """
        base_path = HiddenStatesConfig.get_hidden_states_path(dataset_name, model_name)
        return os.path.join(base_path, f"{sample_id}.safetensors")

    @staticmethod
    def get_metadata_path(dataset_name: str, model_name: str = None) -> str:
        """Get path for metadata file

        Args:
            dataset_name: Name of the dataset
            model_name: Model name for file paths

        Returns:
            Path: /Dataset/HiddenStates/{dataset}/{model}/metadata.json
        """
        base_path = HiddenStatesConfig.get_hidden_states_path(dataset_name, model_name)
        return os.path.join(base_path, "metadata.json")

    @staticmethod
    def get_progress_path(dataset_name: str, model_name: str = None) -> str:
        """Get path for progress tracking file (for resume)

        Args:
            dataset_name: Name of the dataset
            model_name: Model name for file paths

        Returns:
            Path: /Dataset/HiddenStates/{dataset}/{model}/progress.json
        """
        base_path = HiddenStatesConfig.get_hidden_states_path(dataset_name, model_name)
        return os.path.join(base_path, "progress.json")

    @staticmethod
    def ensure_dir(dataset_name: str, model_name: str = None) -> str:
        """Create directory if it doesn't exist

        Args:
            dataset_name: Name of the dataset
            model_name: Model name for file paths

        Returns:
            The created directory path
        """
        path = HiddenStatesConfig.get_hidden_states_path(dataset_name, model_name)
        os.makedirs(path, exist_ok=True)
        return path


# Available datasets in the project
AVAILABLE_DATASETS = [
    "musique",
    "quality",
    "graphragBench_medical",
    "ultraDomain_legal",
]


if __name__ == "__main__":
    print(f"Model Path: {HiddenStatesConfig.MODEL_PATH}")
    print(f"Model Name: {HiddenStatesConfig.MODEL_NAME}")
    print(f"Layer IDs: {HiddenStatesConfig.LAYER_IDS}")
    print(f"Hidden Size: {HiddenStatesConfig.HIDDEN_SIZE}")
    print(f"Storage Dtype: {HiddenStatesConfig.DTYPE}")
    print(f"Aggregation Methods: {HiddenStatesConfig.AGGREGATION_METHODS}")
    print(f"\nSample Path: {HiddenStatesConfig.get_sample_path('musique', 'musique_0000')}")
    print(f"Metadata Path: {HiddenStatesConfig.get_metadata_path('musique')}")
