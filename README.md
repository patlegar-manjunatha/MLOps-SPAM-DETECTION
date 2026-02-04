
# Reproducible MLOps Pipeline with DVC

**Growing from writing single-use notebook scripts to building reproducible, industry-grade machine learning systems.**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/DVC-Data_Versioning-9cf?style=flat&logo=dvc&logoColor=white" alt="DVC" />
  <img src="https://img.shields.io/badge/Git-Version_Control-orange?logo=git&logoColor=white" />
  <img src="https://img.shields.io/badge/DVCLive-Experiment_Tracking-green" />
  
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=black" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/NumPy-Computation-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-Dataframes-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/spaCy-NLP-09a3d5?style=flat&logo=spacy&logoColor=white" alt="spaCy" />
</p>
This repository demonstrates a disciplined approach to machine learning engineering. It shifts focus from simple model building to creating a robust, reproducible, and observable MLOps pipeline using DVC (Data Version Control) and modular Python architecture.



## Project Overview

**The Problem:** Traditional notebooks are often messy, non-reproducible, and hard to collaborate on.\
**The Solution:** A decoupled, configuration-driven pipeline where code, data, and parameters are versioned separately.

### Key Features
* **Reproducibility:** The pipeline executes deterministically via explicit dependency graphs.
* **Modularity:** Distinct stages for ingestion, processing, and training.
* **Data Versioning:** Data and models are tracked alongside code using DVC.
* **Configuration:** All hyperparameters are controlled via a centralized configuration file.



## Pipeline Architecture

The system follows a strict linear dependency graph. DVC ensures that only stages with changed dependencies are re-executed.


```mermaid
graph LR
    A[Raw Data] -->|ingest| B[Data Ingestion]
    B -->|clean| C[Data Preprocessing]
    C -->|vectorize| D[Feature Engineering]
    D -->|train| E[Model Building]
    E -->|validate| F[Model Evaluation]

    %% Node styling (dark background, white text)
    style A fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#ffffff
    style B fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#ffffff
    style C fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#ffffff
    style D fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#ffffff
    style E fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#ffffff
    style F fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#ffffff
```

## Model Performance

The current pipeline implements a **Random Forest Classifier** with TF-IDF vectorization. Focus was placed on pipeline stability over model complexity.

| Metric | Score | Description |
| --- | --- | --- |
| **Accuracy** | **94.6%** | High overall classification correctness. |
| **Precision** | **84.4%** | Minimizes false positives. |
| **Recall** | **100.0%** | Captures all relevant positive cases perfectly. |
| **AUC Score** | **0.995** | Excellent separation capability between classes. |

*Metrics generated via `src/model_evaluation.py` and tracked in `reports/metrics.json`.*


## Repository Structure

The directory structure mimics industry standards for maintainability.

```text
.
├── .dvc/                   # DVC internal metadata & cache
├── data/                   # Data registry (Git-ignored, DVC-tracked)
│   ├── raw/                # Immutable original data
│   ├── interim/            # Cleaned/Normalized data
│   └── processed/          # Model-ready features (TF-IDF matrices)
│
├── logs/                   # Execution logs per stage
├── model/                  # Serialized model artifacts (.pkl)
├── reports/                # JSON metrics and evaluation plots
├── src/                    # Modular source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
│
├── dvc.yaml                # The Pipeline Definition (DAG)
├── params.yaml             # Centralized Configuration
└── requirements.txt        # Python dependencies

```

## Execution Guide

### 1. Clone the repository:

```bash
git clone https://github.com/patlegar-manjunatha/NextWord-Predictor.git
```

### 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Initialize DVC

```bash
dvc init
```

### 4. Reproduce Pipeline

Run the end-to-end workflow. DVC will check dependencies and run necessary stages.

```bash
dvc repro
```

## Design Decisions

Every file in this repository serves a specific engineering purpose:

1. **DVC over Makefiles:** DVC offers data-aware versioning, meaning it knows if your *dataset* changed, not just your code.
2. **Configuration Management:** Hard-coding parameters (like `n_estimators` or `test_size`) is avoided. Parameters are centralized in `params.yaml` to allow easy experimentation.
3. **Modular Architecture:** Unlike a monolithic notebook, this structure allows individual components to be tested, linted, and reused.
4. **Logging Strategy:** Each stage emits standard logs to `logs/`, essential for debugging remote training runs.

## Why This Project Qualifies as Industry-Grade (Level 1)

This repository demonstrates the baseline expectations for a professional machine learning project.

* Modular and maintainable codebase
* Configuration-driven execution
* Fully reproducible pipelines
* Versioned data and models
* Logged and traceable experiments
* Clean repository hygiene
---
### Author: [Manjunatha Patlegar](https://www.linkedin.com/in/patlegar-manjunatha/)
*Building robust AI systems.*
