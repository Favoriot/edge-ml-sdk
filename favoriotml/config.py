from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    model_path: str
    preprocessor_path: Optional[str] = None  # ONLY preprocessor path (no scaler_path)
    model_type: str = "joblib"
    task_type: str = "regression"
    feature_order: List[str] = None

    def __post_init__(self):
        if self.model_type not in ["joblib", "onnx"]:
            raise ValueError("model_type must be 'joblib' or 'onnx'")
        if self.task_type not in ["classification", "regression", "clustering"]:
            raise ValueError("task_type must be 'classification', 'regression', or 'clustering'")
        if self.feature_order is None:
            raise ValueError("feature_order must be provided")