import json
import numpy as np
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class JSONStandardScaler(BaseEstimator, TransformerMixin):
    """Reconstruct StandardScaler from JSON params (no joblib dependency)"""
    def __init__(self, mean=None, scale=None):
        self.mean_ = np.array(mean) if mean is not None else None
        self.scale_ = np.array(scale) if scale is not None else None
        self.n_features_in_ = len(mean) if mean is not None else 0
    
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            return X
        return (X - self.mean_) / self.scale_
    
    def fit(self, X, y=None):
        return self  # Not needed for inference


class JSONLabelEncoder:
    """Lightweight label encoder from JSON mapping (implements .transform())"""
    def __init__(self, mapping: dict):
        # Clean whitespace from mapping keys/values
        self.mapping = {
            str(k).strip(): int(v) 
            for k, v in mapping.items() 
            if k.strip() not in ["MISSING", "UNKNOWN"]
        }
        # Default value for unknown categories
        self.default_value = len(self.mapping)
    
    def transform(self, values):
        result = []
        for val in values:
            key = str(val).strip() if val is not None else "MISSING"
            result.append(self.mapping.get(key, self.default_value))
        return np.array(result)


class Preprocessor:
    def __init__(self, feature_order: list):
        # Clean whitespace from feature names
        self.feature_order = [f.strip() for f in feature_order]
    
    def transform(
        self,
        payload: Dict[str, Any],
        categorical_encoders: Optional[Dict] = None,
        scaler: Optional[Any] = None,
        dtype: str = "float64"
    ) -> np.ndarray:
        """
        Convert payload dict to model-ready 2D array with FULL preprocessing support:
        1. Categorical encoding (LabelEncoder/OneHotEncoder)
        2. Numeric extraction in correct order
        3. Scaling (StandardScaler/MinMaxScaler)
        4. Dtype conversion (float32 for ONNX, float64 for joblib)
        
        Args:
            payload: Input features as dict (e.g., {"User_Type": "visitor", "temp": 25})
            categorical_encoders: Dict of {feature: {'type': 'label'/'onehot', 'encoder': obj}}
            scaler: Optional scaler from preprocessor artifact (applied AFTER encoding)
            dtype: "float32" for ONNX, "float64" for joblib
        
        Returns:
            2D numpy array of shape (1, n_features) with specified dtype
        """
        # WORK ON A COPY to avoid mutating original payload
        processed_payload = {}
        for key, value in payload.items():
            processed_payload[key.strip()] = value
        
        # STEP 1: Apply categorical encoding FIRST (before numeric extraction)
        if categorical_encoders:
            for feat, enc_info in categorical_encoders.items():
                clean_feat = feat.strip()
                if clean_feat not in processed_payload:
                    continue
                
                raw_value = processed_payload[clean_feat]
                
                # Skip if already numeric (allows mixed pre-encoded/raw inputs)
                if isinstance(raw_value, (int, float, np.number)):
                    continue
                
                try:
                    # Convert to string for encoding (handles None/NaN)
                    str_value = str(raw_value).strip() if raw_value is not None else "MISSING"
                    
                    if enc_info['type'] == 'label':
                        # LabelEncoder: single-column compact encoding
                        encoded_val = enc_info['encoder'].transform([str_value])[0]
                        processed_payload[clean_feat] = encoded_val
                    
                    elif enc_info['type'] == 'onehot':
                        # OneHotEncoder fallback: use argmax as compact representation
                        try:
                            ohe_result = enc_info['encoder'].transform([[str_value]])
                            processed_payload[clean_feat] = int(np.argmax(ohe_result[0]))
                        except Exception:
                            processed_payload[clean_feat] = 0
                            print(f"⚠ Unknown category '{str_value}' for feature '{clean_feat}', using default encoding")
                
                except Exception as e:
                    raise ValueError(
                        f"Failed to encode categorical feature '{clean_feat}' with value '{raw_value}': {e}"
                    ) from e

        # STEP 2: Extract values in correct order (now all should be numeric)
        try:
            vector = [processed_payload[f] for f in self.feature_order]
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(
                f"Missing required feature '{missing}'. "
                f"Expected features: {self.feature_order}"
            ) from e

        # STEP 3: Convert to numpy array
        try:
            x = np.array([vector], dtype=np.float64)
        except (ValueError, TypeError) as e:
            non_numeric = [
                (f, processed_payload[f]) 
                for f in self.feature_order 
                if not isinstance(processed_payload[f], (int, float, np.number))
            ]
            raise ValueError(
                f"Non-numeric values after encoding: {non_numeric}\n"
                f"All inputs must be numeric or have a valid categorical encoder."
            ) from e

        # STEP 4: Apply scaler if provided (from preprocessor artifact)
        if scaler is not None:
            x = scaler.transform(x)

        # STEP 5: Convert dtype for ONNX compatibility
        target_dtype = np.float32 if dtype.strip() == "float32" else np.float64
        return x.astype(target_dtype)