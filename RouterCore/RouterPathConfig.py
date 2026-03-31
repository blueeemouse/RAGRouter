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
    def get_aggregated_path(dataset_name: str, result_model: str, file_name: str = "query_metrics_v1.json") -> Path:
        return RouterPathConfig.AGGREGATED_DIR / dataset_name / result_model / file_name

    @staticmethod
    def get_hard_label_path(dataset_name: str, result_model: str, label_name: str = "hard_llm_correct_rule_v1") -> Path:
        return RouterPathConfig.LABELS_DIR / dataset_name / result_model / f"{label_name}.json"

    @staticmethod
    def get_soft_label_path(dataset_name: str, result_model: str, label_name: str = "soft_llm_correct_v1") -> Path:
        return RouterPathConfig.LABELS_DIR / dataset_name / result_model / f"{label_name}.json"

    @staticmethod
    def get_split_path(dataset_name: str, split_name: str = "split_v1") -> Path:
        return RouterPathConfig.SPLITS_DIR / dataset_name / f"{split_name}.json"

    @staticmethod
    def get_model_dir(model_name: str, dataset_name: str) -> Path:
        return RouterPathConfig.MODELS_DIR / model_name / dataset_name

    @staticmethod
    def get_evaluation_path(model_name: str, dataset_name: str, file_name: str = "evaluation.json") -> Path:
        return RouterPathConfig.EVALUATION_DIR / model_name / dataset_name / file_name

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
