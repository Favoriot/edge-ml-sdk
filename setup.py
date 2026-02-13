from setuptools import setup, find_packages

setup(
    name="favoriotml",
    version="1.0.0",
    author="mnazrinnapiah",
    description="Edge ML SDK for Favoriot local inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/favoriot/favoriot-ml-sdk",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "joblib",
        "onnxruntime",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
