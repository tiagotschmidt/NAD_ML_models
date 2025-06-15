# EPPNAD: Energy and Performance Profiler for Network Anomaly Detection

[![Continuous Integration]](https://github.com/tiagotschmidt/NAD_ML_models/blob/main/.github/workflows/ci.yml)

EPPNAD is a Python framework for studying the energy and performance trade-offs of Deep Learning models applied to network anomaly detection. It provides a profiling mechanism to systematically measure and analyze how different model architectures and hyperparameter configurations impact performance metrics (like accuracy, precision, recall) and energy consumption. This allows researchers and practitioners to make informed decisions when designing and deploying deep learning-based intrusion detection systems.

# 🚀 Installation

You can install and use EPPNAD in two ways, depending on your goal.

## For Users (via Pip)

_Note: This will be available once the package is published to the Python Package Index (PyPI)._

```bash
pip install eppnad
```

## For Developers (from source)

To set up the project for development, which allows you to modify the source code, clone the repository and use the Conda environment.
Bash

### 1. Clone the repository
```bash
git clone https://github.com/tiagotschmidt/NAD_ML_models.git
cd <NAD_ML_models>
```

### 2. Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate eppnad
```

### 3. Install the project in editable mode
```bash
pip install -e .
```

# ⚡ Quick Start

The main entry point for running experiments is the manager.py script. It uses a configuration file to define the models to run, their hyperparameters, and other experiment parameters.

The framework will then execute the defined experiments, train the models, measure performance and energy, and save the results for analysis.

Check the examples directory for reference.