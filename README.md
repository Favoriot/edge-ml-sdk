## Favoriot Edge ML SDK

The **Favoriot Edge ML SDK** is a lightweight, high-performance Python library specifically designed for Favoriot platform users. It enables seamless local machine learning inference on edge gateways, industrial PCs, and IoT devices by providing a unified interface for **Scikit-Learn (Joblib)** and **ONNX** models.

This SDK allows you to run "offline intelligence," processing raw sensor payloads directly at the edge without requiring constant cloud connectivity.

## 🚀 Features
- Unified Inference API: Simple predict() and predict_from_payload() methods support Classification, Regression, and Clustering tasks.
- Built-in Preprocessing: Automatically handles StandardScaler, MinMaxScaler, LabelEncoder, and OneHotEncoder through a dedicated preprocessor artifact.
- Favoriot Optimized: Designed to map raw sensor JSON payloads directly to model-ready tensors based on your Favoriot training configuration.
- DBSCAN Support: Includes custom label assignment logic for DBSCAN models, which typically lack a standard .predict() method for new samples.
- Dual-Engine Support: Run legacy .joblib models or high-performance .onnx runtimes.
- Performance Tracking: Built-in timing breakdown to distinguish between preprocessing latency and model execution time.

## 📦 Installation

### Requirements:
 - Python 3.9+
 - numpy, scikit-learn, joblib, onnxruntime

 1. Prepare the Environment
  - It is highly recommended to use a virtual environment to avoid dependency conflicts with other projects.
  - Step 1: Create and activate a virtual environment
  ```bash
        # Create a virtual environment
        python -m venv favoriot-ml-inference

        # Activate the environment
        # On Windows:
        favoriot-ml-inference\Scripts\activate
        # On Linux/macOS:
        source favoriot-ml-inference/bin/activate
  ```
  - Step 2: Pull from GitHub

  ```bash
        git clone https://github.com/Favoriot/edge-ml-sdk.git
        cd edge-ml-sdk

  ```
  2. Install from Source
  - Navigate to the root directory where the setup.py file is located and run the following command:
  ```bash
        # Install in regular mode
        pip install .
        # OR install in editable mode (useful if you plan to modify the SDK code)
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
- **preprocessor_path**: Path to the preprocessor.joblib artifact (this contains all scalers and encoders; no separate scaler path is needed).
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
    preprocessor_path=r"path/to/your/preprocessor.joblib", # Consolidated artifact
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

