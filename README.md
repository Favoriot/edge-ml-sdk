# Favoriot Edge ML SDK

The Favoriot Edge ML SDK is a lightweight, high-performance Python library designed for edge deployments on gateways, industrial PCs, and IoT devices. It enables seamless **offline inference** for Scikit-Learn (Joblib) and ONNX models with built-in preprocessing—now fully compatible with **Python 3.12+** using secure JSON artifacts instead of pickle-based formats.

> ✅ **Production-ready**: No unsafe pickle dependencies • Whitespace-robust • ONNX-optimized • Backward compatible

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| **Python 3.12+ Compatible** | Uses JSON preprocessor artifacts instead of pickle/joblib for secure cross-version deployment |
| **Dual Format Support** | Loads preprocessing artifacts from **both `.json` (recommended)** and `.joblib` formats |
| **Whitespace-Robust Parsing** | Automatically strips inconsistent spacing in feature names/values (e.g., `"User_Type "` → `"User_Type"`) |
| **ONNX-Optimized** | Native float32 handling, automatic dtype conversion, and production-grade latency tracking |
| **Unified Preprocessing** | Single artifact contains scalers, categorical encoders, and feature metadata |
| **DBSCAN Support** | Custom label assignment logic for models without standard `.predict()` methods |
| **Zero-Config Encoding** | Raw categorical strings (`"Visitor"`, `"High"`) accepted directly—no manual encoding required |

## 📦 Installation

### Requirements
- Python 3.9+ (fully tested on 3.12+)
- Core: `numpy`, `scikit-learn`, `joblib`
- ONNX (optional but recommended): `onnxruntime`

### Setup
```bash
# Create virtual environment (recommended)
python -m venv favoriot-ml-env
source favoriot-ml-env/bin/activate  # Linux/macOS
# favoriot-ml-env\Scripts\activate   # Windows

# Install SDK
git clone https://github.com/Favoriot/edge-ml-sdk.git
cd edge-ml-env
pip install -e .
```

  3. Verify Installation
  - After installation, you can verify that the favoriotml package is available by listing your installed packages:
  ```bash
        pip list | grep favoriotml
  ```

## ⚙️ Model Configuration
The ModelConfig class is used to validate your settings. Note: You must follow the exact configuration used during your Favoriot ML training.

- **model_path**: Path to your .pkl, .joblib, or .onnx file.
- **preprocessor_path**: Path to the preprocessor.json or joblib(support python3.12++) artifact (this contains all scalers and encoders; no separate scaler path is needed).
- **model_type**: Either "joblib" or "onnx".
- **task_type**: Must be one of "classification", "regression", or "clustering".
- **feature_order**: A list of features (e.g., ["temp", "humidity"]) in the exact order used during Favoriot training.

## 🛠️ Quick Start
This example shows how to perform clustering inference using a Joblib model and raw categorical strings.

```python
from favoriotml.config import ModelConfig
from favoriotml.model_loader import ModelLoader
from favoriotml.inference import EdgeInference

# 1. Setup Configuration (must match Favoriot training configuration)
config = ModelConfig(
    model_path=r"path/to/your/model.pkl",
    preprocessor_path=r"path/to/your/preprocessor.json", # Consolidated artifact
    model_type="joblib",
    task_type="clustering",
    feature_order=["User_Type", "Nearby_Traffic_Level"]
)

# 2. Load model and ALL preprocessing artifacts from the preprocessor file
model, scaler, label_encoder, model_name, cat_encoders, _ = ModelLoader(
    config.model_path,
    config.model_type,
    config.preprocessor_path
).load()

# 3. Initialize Inference Engine
inference = EdgeInference(
    model=model,
    model_type=config.model_type,
    task_type=config.task_type,
    feature_order=config.feature_order,
    scaler=scaler,                    # Automatically extracted from preprocessor
    label_encoder=label_encoder,
    categorical_encoders=cat_encoders  # Automatically extracted from preprocessor
)

# 4. Predict from raw Favoriot sensor payload
payload = {"User_Type": "Visitor", "Nearby_Traffic_Level": "Low"}
prediction = inference.predict_from_payload(payload)

print(f"Prediction: {prediction}")
print(f"Latency: {inference.get_timing_breakdown()}")

```

## 🏗️ Architecture

| Component       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| ModelLoader     | Extracts models and metadata (scalers, encoders) from local disk storage. |
| Preprocessor    | Handles encoding, feature reordering, and scaling of raw sensor data.     |
| EdgeInference   | The main execution engine that routes data to the correct prediction backend. |
| ModelConfig     | Validates configuration parameters against Favoriot-supported types.      |

## 📈 Monitoring
Monitor your edge device performance with built-in metrics:
- get_last_inference_time(): Returns total duration (preprocessing + inference) in ms.
- get_timing_breakdown(): Provides a detailed dictionary of preprocessing_ms and model_inference_ms.

## 📄 License
- Distributed under the MIT License.

## 🤝 Support

- For Favoriot-specific integration, visit the **Favoriot Developer Documentations**:  
  https://platform.favoriot.com/tutorial/v2/

- 🌐 Official Website:  
  https://favoriot.com/

- 💻 GitHub Repository:  
  https://github.com/Favoriot
