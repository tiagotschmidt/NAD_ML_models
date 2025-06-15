# Project Evolution: From Graduation Work to Published Library

This document outlines the revised roadmap for extending the project. The primary goals are to extend the research paper for journal submission and to refactor the associated Python framework into a high-quality, reusable library.

---

## 🚀 Phase 1: Foundational Cleanup and Reorganization

The first step is to clean up the repository, establish a professional structure, and automate the development environment setup.

### 1.1. New Repository Structure
- [x] Reorganize the project files and directories to follow a standard Python project layout.

### 1.2. Automated Development Environment
- [x] Create `environment.yml` for reproducible Conda environments.
- [x] Create/update a `Makefile` to automate common tasks like `setup`, `install`, and `test`.

### 1.3. Establish the Pip Package
- [x] Create `pyproject.toml` to define the package metadata and dependencies, making it installable via `pip`.

---

## 🛠️ Phase 2: A Practical Refactoring Plan

Refactor the existing, scattered scripts into a modular and cohesive framework. The goal is to decouple data processing, model definition, and experiment execution.

### 2.1. Refactor the Core Execution Engine
- [x] **Goal:** Make the `ExecutionEngine` a better generic orchestrator that runs experiments using the (keras) standardized model interface.
- [x] **Action:**
    - [x] Move `execution_engine.py` to `src/eppnad/core/`.
    - [x] Update execution logic to enable intermitent execution.
    - [x] Update result saving logic to provide more insightfull data.
    - [x] Update profile saving logic to use long term storage periodically (csv?). 
    - [x] Create a runtime checkpoint logic.
    - [x] Create multiple profiles functions (full, intermitent).
    - [x] Add unit tests for all functions.
    
### 2.2. Refactor the Logger(EnergyMonitor)
- [x] **Goal:** Make the `EnergyMonitor` a better energy watchdog process.
- [x] **Action:**
    - [x] Move `logger.py` to `src/eppnad/core/`.    
    - [x] Add unit tests for all functions.
    
### 2.3. Refactor the Manager
- [x] **Goal:** Make the `Manager` a better entrypoint process for the framework..
- [x] **Action:**
    - [x] Move `manager.py` to `src/eppnad/core/`.    
    - [x] Add unit tests for all functions.
    - [x] Adapt the manager for the different profile styles (and new data structures).
    
### 2.4. Refactor the Plotter
- [x] **Goal:** Write a better plotting engine. 
- [x] **Action:**
    - [x] Move `plotter.py` to `src/eppnad/core/`.    
    - [x] Adapt the plotter to new intermediate way of storing profile data (csv?)

---

## ✅ Phase 3: Incremental Testing and CI

Build confidence in the framework's reliability through a structured, incremental testing strategy and automated checks.

### 3.1. Set Up the Testing Infrastructure
- [x] Add `pytest` and `pytest-cov` to the development environment file.
- [x] Create a `pytest.ini` or configure `pyproject.toml` to automatically discover tests in the `tests/` directory.

### 3.2. Implement Unit Tests Incrementally
- [x] **Target:** Core framework components.
    - [x] Write unit tests for the `EnergyMonitor` to verify message formatting and output.
    - [x] Write unit tests for the `Manager` to ensure it correctly parses configurations.
    - [x] Write unit tests for the `ExecutionEngine` using a "mock" model object to test the execution flow without training a real model.

### 3.3. Measure and Improve Test Coverage
- [x] **Goal:** Use test coverage as a guide to identify untested parts of the codebase.
- [x] **Action:**
    - [x] Configure `pytest-cov` to generate a coverage report in the terminal after running tests. Add the following to your `pytest.ini`:
     ``ini
      [pytest]
      addopts = --cov=src/nad_framework --cov-report=term-missing
      ```
    - [x] Set an initial, achievable coverage target (e.g., 60%).
    - [x] Write new tests specifically for the files/lines that the coverage report shows are being missed.

### 3.4. Set Up Continuous Integration (CI)
- [x] Create a GitHub Actions workflow (`.github/workflows/ci.yml`).
- [x] Configure the CI to:
    - [x] Check out the code.
    - [x] Install the Conda environment from `environment.yml`.
    - [x] Run the full `pytest` suite with coverage reporting on every push and pull request.

---

## 🔬 Phase 4: Experimentation and Documentation

Focus on the research extension and prepare the project for public use by creating a flexible experiment client and comprehensive documentation.

### 4.1. Develop a Flexible Experiment Client
- [ ] **Goal:** Create a simple command-line tool for running experiments without needing to write new Python code for each run.
- [ ] **Technology:** Use Python's built-in `argparse` library for simplicity.
- [ ] **Functionality:**
    - [ ] Create `examples/run_experiment.py`.
    - [ ] Add command-line arguments to specify the model type (`--model mlp`), dataset path (`--data ...`), and output file for results (`--output results.json`).
    - [ ] The script will use the refactored `nad-framework` library to execute the experiment.
    - [ ] Results should be saved in a structured format (JSON is a good choice) for easy analysis later.

### 4.2. Write a Comprehensive `README.md`
- [x] **Goal:** Make the `README.md` the definitive guide for a new user.
- [x] **Structure:**
    - [x] **Project Title & CI Badge:** Add the title and the auto-updating CI badge from your GitHub Action.
    - [x] **Project Description:** A clear, one-paragraph summary of what the project does and the problem it solves.
    - [x] **Installation:** Provide two clear, copy-pasteable blocks of code for installing the framework:
        1. For users: `pip install nad-framework` (once it's on PyPI).
        2. For developers: `conda env create -f environment.yml` followed by `pip install -e .`.
    - [x] **Quick Start:** A minimal, working example showing how to use the `run_experiment.py` client.
---

## 🏆 Phase 5: Finalization

Perform the final tasks before a stable `1.0` release.

### 5.1. Choose a Final Repository Name
- [ ] **Goal:** Select a permanent, descriptive name for the framework now that its structure and purpose are clear.
- [ ] **Action:**
    - [ ] Brainstorm names that reflect the project's focus (e.g., `NetAnomalyDetector`, `EnergyProfilerDL`, `NADKit`).
    - [ ] Check PyPI and GitHub to ensure the name is available.
    - [ ] Rename the repository and update the `pyproject.toml` file.
