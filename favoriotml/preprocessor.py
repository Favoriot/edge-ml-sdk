import numpy as np
from typing import Dict, Any, Optional

class Preprocessor:
    def __init__(self, feature_order: list):
        self.feature_order = feature_order

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
        processed_payload = payload.copy()
        
        # STEP 1: Apply categorical encoding FIRST (before numeric extraction)
        if categorical_encoders:
            for feat, enc_info in categorical_encoders.items():
                if feat not in processed_payload:
                    continue
                
                raw_value = processed_payload[feat]
                
                # Skip if already numeric (allows mixed pre-encoded/raw inputs)
                if isinstance(raw_value, (int, float, np.number)):
                    continue
                
                try:
                    # Convert to string for encoding (handles None/NaN)
                    str_value = str(raw_value) if raw_value is not None else "MISSING"
                    
                    if enc_info['type'] == 'label':
                        # LabelEncoder: single-column compact encoding
                        encoded_val = enc_info['encoder'].transform([str_value])[0]
                        processed_payload[feat] = encoded_val
                    
                    elif enc_info['type'] == 'onehot':
                        # OneHotEncoder fallback: use argmax as compact representation
                        try:
                            ohe_result = enc_info['encoder'].transform([[str_value]])
                            processed_payload[feat] = int(np.argmax(ohe_result[0]))
                        except Exception:
                            processed_payload[feat] = 0
                            print(f"⚠ Unknown category '{str_value}' for feature '{feat}', using default encoding")
                
                except Exception as e:
                    known_cats = list(enc_info['encoder'].classes_)
                    raise ValueError(
                        f"Failed to encode categorical feature '{feat}' with value '{raw_value}': {e}\n"
                        f"Known categories: {known_cats}"
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
        target_dtype = np.float32 if dtype == "float32" else np.float64
        return x.astype(target_dtype)