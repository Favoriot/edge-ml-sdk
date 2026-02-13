import numpy as np
import time
from typing import Any, Optional, Dict
from sklearn.metrics.pairwise import pairwise_distances
from .preprocessor import Preprocessor

class EdgeInference:
    def __init__(
        self,
        model: Any,
        model_type: str,
        task_type: str,
        feature_order: list,
        scaler: Optional[Any] = None,          # Comes from preprocessor artifact
        label_encoder: Optional[Any] = None,
        categorical_encoders: Optional[Dict] = None,
        model_name: str = ""
    ):
        self.model = model
        self.model_type = model_type
        self.task_type = task_type
        self.feature_order = feature_order
        self.scaler = scaler                    # Scaler from preprocessor (not standalone file)
        self.label_encoder = label_encoder
        self.categorical_encoders = categorical_encoders or {}
        self.model_name = model_name.lower()
        self.preprocessor = Preprocessor(feature_order)
        self.last_inference_time_ms = 0.0
        self.last_preprocessing_time_ms = 0.0
        
        # Debug info
        if self.categorical_encoders:
            print(f"Intialized with {len(self.categorical_encoders)} categorical encoders: {list(self.categorical_encoders.keys())}")
        if self.scaler is not None:
            print(f"Intialized with scaler: {self.scaler.__class__.__name__}")

    def _dbscan_assign_label(self, model, new_data):
        """Custom DBSCAN label assignment for new samples."""
        if not hasattr(model, 'components_') or model.components_ is None:
            return np.full(new_data.shape[0], -1)
        
        core_samples = model.components_
        distances = pairwise_distances(new_data, core_samples)
        labels = np.full(new_data.shape[0], -1)
        
        for i, dist in enumerate(distances):
            if np.min(dist) <= model.eps:
                labels[i] = model.labels_[np.argmin(dist)]
        
        return labels

    def predict(self, x: np.ndarray) -> Any:
        """
        Run inference and return SINGLE prediction value (not list).
        Handles classification, regression, and clustering models.
        """
        if self.model_type == "joblib":
            start = time.perf_counter()
            
            # Clustering path
            if self.task_type == "clustering":
                if 'dbscan' in self.model_name:
                    preds = self._dbscan_assign_label(self.model, x)
                else:
                    preds = self.model.predict(x)
                pred = int(preds[0])
            
            # Classification path
            elif self.task_type == "classification":
                preds = self.model.predict(x)
                pred = preds[0]
                if self.label_encoder is not None:
                    pred = self.label_encoder.inverse_transform([pred])[0]
            
            # Regression path
            else:  # regression
                preds = self.model.predict(x)
                pred = float(preds[0])
            
            self.last_inference_time_ms = (time.perf_counter() - start) * 1000
            return pred

        elif self.model_type == "onnx":
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            
            start = time.perf_counter()
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: x})
            self.last_inference_time_ms = (time.perf_counter() - start) * 1000
            
            pred = outputs[0][0]
            
            # Handle bytes → string conversion for zipmap classifiers
            if isinstance(pred, bytes):
                pred = pred.decode('utf-8')
            # For clustering/regression, ensure numeric output
            elif self.task_type in ["clustering", "regression"] and not isinstance(pred, (int, float)):
                try:
                    pred = float(pred)
                    if self.task_type == "clustering":
                        pred = int(pred)
                except (ValueError, TypeError):
                    pass
            
            return pred

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def predict_from_payload(self, payload: dict) -> Any:
        """End-to-end inference with FULL preprocessing (encoding + scaling)."""
        prep_start = time.perf_counter()
        dtype = "float32" if self.model_type == "onnx" else "float64"
        
        # CRITICAL: Pass BOTH categorical_encoders AND scaler from preprocessor artifact
        x = self.preprocessor.transform(
            payload,
            categorical_encoders=self.categorical_encoders,
            scaler=self.scaler,  # Scaler comes from preprocessor artifact
            dtype=dtype
        )
        self.last_preprocessing_time_ms = (time.perf_counter() - prep_start) * 1000
        
        pred = self.predict(x)
        total_time = self.last_preprocessing_time_ms + self.last_inference_time_ms
        self.last_inference_time_ms = total_time
        
        return pred

    def get_last_inference_time(self) -> float:
        """Get last inference duration in milliseconds (preprocessing + model)."""
        return self.last_inference_time_ms

    def get_timing_breakdown(self) -> dict:
        """Get detailed timing breakdown."""
        model_time = self.last_inference_time_ms - self.last_preprocessing_time_ms
        return {
            "preprocessing_ms": self.last_preprocessing_time_ms,
            "model_inference_ms": max(0, model_time),
            "total_ms": self.last_inference_time_ms
        }