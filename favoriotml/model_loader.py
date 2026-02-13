import joblib
import onnxruntime as ort
from typing import Any, Optional, Tuple, Dict

class ModelLoader:
    def __init__(self, model_path: str, model_type: str, preprocessor_path: Optional[str] = None):
        self.model_path = model_path
        self.model_type = model_type
        self.preprocessor_path = preprocessor_path  # ONLY preprocessor path (no scaler_path)

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

        # Load preprocessor artifact FIRST (contains encoders + scaler + metadata)
        if self.preprocessor_path:
            try:
                artifact = joblib.load(self.preprocessor_path)
                categorical_encoders = artifact.get('categorical_encoders', {})
                scaler = artifact.get('scaler')  # Scaler comes ONLY from preprocessor
                feature_config = artifact.get('feature_config', {})
                
                print(f"✓ Loaded preprocessor artifact: {len(categorical_encoders)} categorical encoders")
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