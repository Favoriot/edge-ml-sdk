import json
import joblib
import onnxruntime as ort
from typing import Any, Optional, Tuple, Dict
from .preprocessor import JSONStandardScaler, JSONLabelEncoder


class ModelLoader:
    def __init__(self, model_path: str, model_type: str, preprocessor_path: Optional[str] = None):
        self.model_path = model_path
        self.model_type = model_type
        self.preprocessor_path = preprocessor_path  # ONLY preprocessor path (no scaler_path)
    
    def _load_json_preprocessor(self, path: str) -> Tuple[Dict, Any, Dict, list]:
        """Load and clean JSON preprocessor artifact"""
        with open(path, 'r') as f:
            raw_artifact = json.load(f)
        
        # Clean whitespace from ALL keys/values recursively
        def clean_dict(d):
            if isinstance(d, dict):
                return {k.strip(): clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            elif isinstance(d, str):
                return d.strip()
            return d
        
        artifact = clean_dict(raw_artifact)
        
        # Extract feature order
        feature_order = artifact.get("feature_order", [])
        
        # Build categorical encoders
        categorical_encoders = {}
        cat_encs_raw = artifact.get("categorical_encoders", {})
        for feat_name, enc_info in cat_encs_raw.items():
            clean_feat = feat_name.strip()
            if enc_info.get("type") == "label" and "mapping" in enc_info:
                categorical_encoders[clean_feat] = {
                    "type": "label",
                    "encoder": JSONLabelEncoder(enc_info["mapping"])
                }
        
        # Reconstruct scaler from params
        scaler = None
        scaler_info = artifact.get("scaler", {})
        if scaler_info.get("type") == "standard":
            scaler = JSONStandardScaler(
                mean=scaler_info.get("mean"),
                scale=scaler_info.get("scale")
            )
        
        feature_config = {"feature_order": feature_order}
        
        print(f"✓ Loaded JSON preprocessor: {len(categorical_encoders)} categorical encoders")
        print(f"  - Features: {feature_order}")
        if scaler:
            print(f"  - Scaler: StandardScaler (reconstructed from params)")
        
        return categorical_encoders, scaler, feature_config, feature_order

    def load(self) -> Tuple[Any, Optional[Any], Optional[Any], str, Dict, Optional[Any]]:
        """
        Load model and ALL preprocessing artifacts from preprocessor artifact.
        Returns:
            (model, scaler, label_encoder, model_name, categorical_encoders, feature_config)
        """
        scaler = None
        label_encoder = None
        model_name = ""
        categorical_encoders = {}
        feature_config = {}
        feature_order = []

        # Load preprocessor artifact FIRST (contains encoders + scaler + metadata)
        if self.preprocessor_path:
            try:
                if self.preprocessor_path.endswith('.json'):
                    # LOAD FROM JSON (Python 3.12 safe)
                    categorical_encoders, scaler, feature_config, feature_order = \
                        self._load_json_preprocessor(self.preprocessor_path)
                else:
                    # Fallback to joblib (legacy support)
                    artifact = joblib.load(self.preprocessor_path)
                    categorical_encoders = artifact.get('categorical_encoders', {})
                    scaler = artifact.get('scaler')   # Scaler comes ONLY from preprocessor
                    feature_config = artifact.get('feature_config', {})
                    feature_order = feature_config.get('feature_order', [])
                    
                    print(f"✓ Loaded joblib preprocessor artifact: {len(categorical_encoders)} categorical encoders")
                    if categorical_encoders:
                        cat_feats = list(categorical_encoders.keys())
                        print(f"  - Categorical features: {cat_feats}")
                    if scaler is not None:
                        print(f"  - Scaler: {scaler.__class__.__name__}")
            except Exception as e:
                raise RuntimeError(f"Failed to load preprocessor artifact '{self.preprocessor_path}': {e}")
        else:
            print("⚠ No preprocessor_path provided - inference will require pre-encoded numeric inputs only")

        # Load model
        if self.model_type == "joblib":
            loaded_obj = joblib.load(self.model_path)
            
            if isinstance(loaded_obj, dict):
                if 'classifier' in loaded_obj:
                    model_obj = loaded_obj['classifier']
                    label_encoder = loaded_obj.get('label_encoder')
                    model_name = model_obj.__class__.__name__
                elif 'regressor' in loaded_obj:
                    model_obj = loaded_obj['regressor']
                    model_name = model_obj.__class__.__name__
                elif 'cluster' in loaded_obj:
                    model_obj = loaded_obj['cluster']
                    model_name = model_obj.__class__.__name__
                else:
                    model_obj = next(iter(loaded_obj.values()))
                    model_name = model_obj.__class__.__name__
            else:
                model_obj = loaded_obj
                model_name = model_obj.__class__.__name__
                label_encoder = None
            
            return model_obj, scaler, label_encoder, model_name, categorical_encoders, feature_config

        elif self.model_type == "onnx":
            sess = ort.InferenceSession(self.model_path)
            model_name = sess.get_modelmeta().producer_name or "ONNXModel"
            # ONNX models don't use sklearn scalers internally, but we still need preprocessor for categorical encoding
            return sess, scaler, None, model_name, categorical_encoders, feature_config

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")