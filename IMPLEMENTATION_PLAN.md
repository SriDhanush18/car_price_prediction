# Implementation Plan - MLOps Car Price Prediction Project

Develop a production-ready, modular Machine Learning Operations (MLOps) pipeline and REST API service for predicting car prices based on automobile technical specifications and market features.

## Project Scope & Workflow

The solution follows a complete 10-stage MLOps lifecycle:
1. **Problem Definition & Dataset Ingestion**: Load and validate the car dataset (205 records, 26 features).
2. **Exploratory Data Analysis (EDA)**: Statistical profiling, distribution analysis, correlation inspection, duplicate & outlier detection, and automated EDA report generation.
3. **Data Preprocessing & Cleaning**:
   - Extract and standardize car brand/manufacturer names (cleaning typos like `vokswagen` $\rightarrow$ `volkswagen`, `porcshce` $\rightarrow$ `porsche`, `toyouta` $\rightarrow$ `toyota`, `maxda` $\rightarrow$ `mazda`).
   - Standardize ordinal/worded numbers (e.g. `doornumber`, `cylindernumber`).
   - Imputation/missing value checks and outlier treatment.
4. **Feature Engineering & Feature Selection**:
   - Create domain features: Power-to-Weight ratio, total vehicle volume ($L \times W \times H$), combined weighted MPG, engine performance index, luxury brand tier flag.
   - Robust column transformation pipeline (`OneHotEncoder`, `StandardScaler`, `FunctionTransformer`) preventing data leakage.
   - Feature importance & collinearity analysis.
5. **Model Building (Diverse Algorithms)**:
   - Baseline Linear Regression
   - Regularized Linear Models: Ridge ($L_2$), Lasso ($L_1$), ElasticNet ($L_1 + L_2$)
   - Non-linear Models: Decision Tree, Support Vector Regressor (SVR), K-Nearest Neighbors (KNN)
   - Ensemble & Boosting Models: Random Forest, Gradient Boosting Regressor, HistGradientBoostingRegressor
6. **Cross-Validation & Hyperparameter Tuning**:
   - 5-Fold Cross-Validation on training split.
   - `GridSearchCV` / `RandomizedSearchCV` for best candidate model optimization.
7. **Model Evaluation & Best Model Selection**:
   - Metrics: $R^2$, Adjusted $R^2$, MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error).
   - Residual analysis, error distributions, and feature importance visual plots.
   - Multi-metric model comparison table with formal selection justification.
8. **Pipeline Serialization & Artifact Storage**:
   - Serialize production pipeline (`preprocessor + model`) into `artifacts/models/best_model.joblib`.
   - Store evaluation metrics in `artifacts/metrics/metrics.json`.
9. **Production REST API (FastAPI) & Serving**:
   - Pydantic schema validation for request payloads.
   - Endpoints: `GET /health`, `GET /model_info`, `POST /predict`, `POST /batch_predict`.
10. **Sample Inference & Comprehensive Final Report**:
    - Test single & batch unseen realistic vehicle specifications.
    - Comprehensive conclusion detailing primary price drivers and business recommendations.

---

## Proposed System Architecture & Directory Structure

```
car_price/
├── data/
│   ├── raw/
│   │   └── car_price.csv              # Raw original dataset
│   └── processed/
│       ├── train.csv                  # Preprocessed train split
│       └── test.csv                   # Preprocessed test split
├── artifacts/
│   ├── models/
│   │   └── best_model.joblib          # Serialized production pipeline
│   ├── metrics/
│   │   └── model_comparison.json      # Benchmark metrics across all models
│   └── plots/                         # Visual artifacts (EDA, residuals, feature importance)
├── src/
│   ├── __init__.py
│   ├── config.py                      # Centralized configuration & hyperparameter grids
│   ├── data_loader.py                 # Ingestion & data split modules
│   ├── preprocessor.py                # Scikit-learn Pipeline & feature transformer definitions
│   ├── train.py                       # Multi-model training, CV, and tuning engine
│   ├── evaluate.py                    # Metric calculation & plotting utilities
│   └── predict.py                     # Standalone CLI / Python inference engine
├── app/
│   ├── __init__.py
│   ├── schemas.py                     # Pydantic input/output validation models
│   └── main.py                        # FastAPI production web service
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py               # Unit tests for preprocessing & model inference
│   └── test_api.py                    # API endpoint integration tests
├── run_pipeline.py                    # Master orchestrator script (End-to-End execution)
├── requirements.txt                   # Project dependencies
└── README.md                          # Full documentation & usage guide
```

---

## User Review Required

> [!IMPORTANT]
> - The dataset contains 205 records. We will use an 80/20 train-test split with 5-fold cross-validation during training to ensure high statistical reliability and prevent overfitting.
> - The entire preprocessing and model pipeline will be saved as a single unified `joblib` object so that inference requests directly accept raw automotive parameters without requiring manual feature transformations.

---

## Verification Plan

### Automated Tests
1. **Pipeline & Model Integrity Tests**:
   - `pytest tests/test_pipeline.py` testing data loading, custom transformer correctness, shape compatibility, non-negative predictions, and model serialization.
2. **API Endpoint Tests**:
   - `pytest tests/test_api.py` testing FastAPI `/health`, `/predict`, and `/batch_predict` with valid and invalid payloads.
3. **End-to-End Pipeline Execution**:
   - Execute `python run_pipeline.py` which runs the full lifecycle from data validation $\rightarrow$ EDA $\rightarrow$ preprocessing $\rightarrow$ training 8+ models $\rightarrow$ tuning $\rightarrow$ model selection $\rightarrow$ evaluation charts $\rightarrow$ unseen test prediction.

### Manual Verification
- Validate the generated comparison table and plots in `artifacts/plots/`.
- Verify interactive Swagger UI documentation at `http://127.0.0.1:8000/docs`.
