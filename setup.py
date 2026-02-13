from setuptools import setup, find_packages

setup(
    name="favoriotml",
    version="0.1.0",
    description="Favoriot ML Inference SDK",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "joblib",
        "scikit-learn",
        "onnxruntime"
    ]
)

