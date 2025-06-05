# Project Evolution: From a Graduation Work to Published Library

This document outlines the roadmap for extending the project. The primary goals are to extend the research paper for journal submission and to refactor the associated Python framework into a high-quality, reusable library.

---

## 🚀 Phase 1: Foundational Cleanup and Reorganization

The first step is to clean up the repository, establish a professional structure, and automate the development environment setup.

### 1.1. New Repository Name

- [ ] Choose a new, descriptive name for the repository.
  - Suggestions: `nad-framework`, `eppnad`, `neuradetect`.
  - *Decision*: `nad-framework` (placeholder)

### 1.2. New Repository Structure

- [x] Reorganize the project files and directories. The new structure will follow a standard Python project layout to improve clarity and maintainability, as shown below.

    ```markdown
    nad-framework/
    ├── .github/workflows/
    ├── data/
    ├── examples/
    ├── src/
    │   └── nad_framework/
    │       ├── core/
    │       ├── models/
    │       └── utils/
    ├── tests/
    ├── .gitignore
    ├── LICENSE
    ├── Makefile
    ├── pyproject.toml
    ├── README.md
    └── environment.yml
    ```

### 1.3. Automated Development Environment

- [x] Create `environment.yml` for reproducible Conda environments.
- [x] Create/update `Makefile` to automate common tasks like `setup`, `install`, and `test`.

---

## 🛠️ Phase 2: Code Refactoring and Framework Development

Refactor the existing code into a robust, modular, and extensible framework with the goal of publishing it as a library.

### 2.1. Establish the Pip Package

- [ ] Create `pyproject.toml` to define the package metadata and dependencies, making it installable via `pip`.

### 2.2. Refactor Core Components

- [ ] **Refactor `eppnad` -> `src/nad_framework/core`**:
    - [ ] Move `manager.py`, `execution_engine.py`, etc., to the new `core` directory.
    - [ ] Apply dependency injection to decouple components (e.g., pass an `Engine` instance to the `Manager`).
- [ ] **Refactor Models -> `src/nad_framework/models`**:
    - [ ] Consolidate scattered model scripts (`mlp_*.py`, `cnn_*.py`, etc.).
    - [ ] Create a `BaseModel` abstract class to define a common interface (`build`, `train`, `predict`).
    - [ ] Make `MLP`, `CNN`, and `LSTM` classes inherit from `BaseModel`.

---

## ✅ Phase 3: Testing and Continuous Integration

Build confidence in the framework's reliability through robust testing and automation.

### 3.1. Improve the Test Suite

- [ ] Write comprehensive unit tests for all core components.
- [ ] Write integration tests for key user workflows (e.g., configuring and running a full experiment).
- [ ] Use `pytest` and `pytest-cov` to measure test coverage.

### 3.2. Set Up Continuous Integration (CI)

- [ ] Create a GitHub Actions workflow (`.github/workflows/ci.yml`).
- [ ] Configure the CI to automatically install dependencies and run the full test suite on every push and pull request.

---

## 🔬 Phase 4: Experimentation and Documentation

Focus on the research extension and prepare the project for public use.

### 4.1. Develop the Experiment Client

- [ ] Create `examples/experiment_client.py` as a CLI tool.
- [ ] Implement command-line arguments (using `argparse`) to configure experiments (model, hyperparameters, dataset).
- [ ] Ensure the client uses the refactored `nad-framework` as a library.
- [ ] Save experiment results to a structured format (e.g., CSV, JSON) for analysis.

### 4.2. Update the `README.md`

- [ ] Rewrite the `README.md` to be the definitive entry point for new users.
- [ ] Include:
    - [ ] Project title and CI badge.
    - [ ] Clear, concise project description.
    - [ ] "Installation" and "Quick Start" sections.
    - [ ] "How to Cite" section for academic use.