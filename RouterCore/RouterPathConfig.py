"""Router-specific path configuration.

This module owns all filesystem locations under Dataset/RouterTrainingData.
It is intentionally independent from the global Config/PathConfig.py.
"""

from pathlib import Path


class RouterPathConfig:
    """Path helpers for router training data and artifacts."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_ROOT = PROJECT_ROOT / "Dataset"
    ROUTER_DATA_ROOT = DATASET_ROOT / "RouterTrainingData"

    AGGREGATED_DIR = ROUTER_DATA_ROOT / "Aggregated"
    LABELS_DIR = ROUTER_DATA_ROOT / "Labels"
    SPLITS_DIR = ROUTER_DATA_ROOT / "Splits"
    MODELS_DIR = ROUTER_DATA_ROOT / "Models"
    EVALUATION_DIR = ROUTER_DATA_ROOT / "Evaluation"

    @staticmethod
    def get_aggregated_dir(dataset_name: str, result_model: str) -> Path:
        """Directory containing aggregated query-metrics files for a dataset/model pair."""
        return RouterPathConfig.AGGREGATED_DIR / dataset_name / result_model

    @staticmethod
    def get_aggregated_path(dataset_name: str, result_model: str, file_name: str = "query_metrics_v1.json") -> Path:
        """Aggregated query-level metrics file path."""
        return RouterPathConfig.get_aggregated_dir(dataset_name, result_model) / file_name

    @staticmethod
    def get_label_dir(dataset_name: str, result_model: str) -> Path:
        """Directory containing label files for a dataset/model pair."""
        return RouterPathConfig.LABELS_DIR / dataset_name / result_model

    @staticmethod
    def get_hard_label_path(dataset_name: str, result_model: str, label_name: str = "hard_llm_correct_rule_v1") -> Path:
        """Hard-label router supervision file path."""
        return RouterPathConfig.get_label_dir(dataset_name, result_model) / f"{label_name}.json"

    @staticmethod
    def get_soft_label_path(dataset_name: str, result_model: str, label_name: str = "soft_llm_correct_v1") -> Path:
        """Soft-label router supervision file path."""
        return RouterPathConfig.get_label_dir(dataset_name, result_model) / f"{label_name}.json"

    @staticmethod
    def get_split_dir(dataset_name: str) -> Path:
        """Directory containing split files for a dataset."""
        return RouterPathConfig.SPLITS_DIR / dataset_name

    @staticmethod
    def get_split_path(dataset_name: str, split_name: str = "split_v1") -> Path:
        """Train/val/test split metadata file path."""
        return RouterPathConfig.get_split_dir(dataset_name) / f"{split_name}.json"

    @staticmethod
    def get_model_dir(model_name: str, dataset_name: str) -> Path:
        """Directory for trained router checkpoints and artifacts."""
        return RouterPathConfig.MODELS_DIR / model_name / dataset_name

    @staticmethod
    def get_evaluation_dir(model_name: str, dataset_name: str) -> Path:
        """Directory for offline router evaluation outputs."""
        return RouterPathConfig.EVALUATION_DIR / model_name / dataset_name

    @staticmethod
    def get_evaluation_path(model_name: str, dataset_name: str, file_name: str = "evaluation.json") -> Path:
        """Offline router evaluation file path."""
        return RouterPathConfig.get_evaluation_dir(model_name, dataset_name) / file_name

    @staticmethod
    def ensure_dir(path: Path) -> None:
        """Create a directory if it does not exist."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def ensure_parent(path: Path) -> None:
        """Create a file path's parent directory if it does not exist."""
        RouterPathConfig.ensure_dir(path.parent)
