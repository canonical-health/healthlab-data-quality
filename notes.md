# Folder Structure
```
health-dq-lab/
├── data/
│   ├── raw/                  # Unmodified source data (e.g. FHIR bundles, CSVs)
│   ├── processed/            # Normalized or flattened data
│   ├── synthetic/            # Simulated examples for training/testing
│   └── mappings/             # Code system maps, value sets, terminology files

├── notebooks/
│   ├── 01_explore_fhir.ipynb         # FHIR exploration + visualization
│   ├── 02_feature_engineering.ipynb  # Feature extraction and transformation
│   └── 03_train_models.ipynb         # Model training workflows

├── src/
│   ├── config/               # YAML/JSON config for models, datasets
│   ├── pipelines/            # ETL, feature pipelines, FHIR to tabular conversion
│   ├── models/
│   │   ├── baseline_ml/      # LogisticRegression, RandomForest, etc.
│   │   ├── deep/             # Autoencoders, TabTransformer, etc.
│   │   ├── gnn/              # Graph neural networks
│   │   └── transformer/      # BERT-style FHIR models
│   ├── evaluation/           # DQ metrics, plausibility tests, explainability
│   └── utils/                # Shared helpers: date math, code lookup, etc.

├── tests/                    # Unit + integration tests

├── scripts/
│   ├── run_pipeline.py       # Main entrypoint for preprocessing
│   ├── train_model.py        # CLI for training models
│   └── evaluate.py           # Evaluate model on DQ dimensions

├── examples/                 # Sample inputs + expected outputs (e.g., flagged FHIR bundles)

├── docs/                     # Project writeups, model architectures, results

├── requirements.txt          # Python dependencies
├── pyproject.toml            # If using Poetry
├── README.md
└── .gitignore
```

# Ideas
- Equate ML Classification Threshold Metrics to DQ Applications
    - Recall: probability of detection (True Positive Rate)
    - probability of false alarm (False Positive Rate)
    - Accuracy, Precision, Recall, False Positive Rate
    - 