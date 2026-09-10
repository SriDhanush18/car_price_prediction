# End-to-End MLOps Car Price Prediction Project

A production-grade, reproducible Machine Learning Operations (MLOps) project designed to predict automobile prices based on technical specifications and market attributes.

---

## Architecture & Data Layer

```
                    DATA LAYER
                        │
                        ▼
             CarPrice_Assignment.csv
                        │
                        ▼
                    data/raw/
                        │
                        ▼
                  ingestion.py
                        │
                        ▼
                 pandas DataFrame
                        │
                        ▼
                 Data Validation
```

---

## Key Features & Highlights

- **Data Layer Ingestion & Automated Validation**: Full schema verification, duplicate detection, null checking, numerical range verification, and categorical domain auditing in `src/ingestion.py`.
- **Exploratory Data Analysis (EDA)**: Statistical profiling, IQR outlier detection, brand distribution, and auto-generated high-resolution visualization artifacts.
- **Leak-Free Preprocessing Pipeline**: Custom Scikit-Learn `Transformer` handling brand typo correction, worded count conversion, domain feature engineering, and robust scaling.
- **Multi-Model Benchmark**: Evaluates 10 algorithms:
  - *Baseline*: Linear Regression
  - *Regularized Models*: Ridge ($L_2$), Lasso ($L_1$), ElasticNet ($L_1 + L_2$)
  - *Non-Linear & Tree Models*: Decision Tree, Random Forest, Support Vector Regressor (SVR), K-Nearest Neighbors (KNN)
  - *Ensembles & Boosting*: Gradient Boosting, HistGradientBoosting
- **5-Fold Cross-Validation & Hyperparameter Tuning**: Comprehensive `GridSearchCV` optimization on candidate models.
- **Evaluation & Diagnostics**: Multi-metric evaluation ($R^2$, Adjusted $R^2$, MAE, RMSE, MAPE) with residual distribution and feature importance plots.
- **Production REST API**: High-throughput FastAPI application with Pydantic validation, `/predict`, `/batch_predict`, `/model_info`, `/validation_report`, and `/health` endpoints.
- **Automated Test Suite**: Full `pytest` unit and integration test coverage ($11/11$ passing).

---

## Directory Structure

```
car_price/
├── data/
│   ├── raw/
│   │   └── CarPrice_Assignment.csv    # Raw original dataset (205 records)
│   └── processed/
│       ├── train.csv                  # 80% train split
│       └── test.csv                   # 20% test split
├── artifacts/
│   ├── models/
│   │   └── best_model.joblib          # Serialized production pipeline
│   ├── metrics/
│   │   ├── data_validation_report.json# Data Layer automated validation report
│   │   ├── eda_summary.json           # EDA statistical summary
│   │   └── model_comparison.json      # Benchmark metrics across all models
│   └── plots/                         # Evaluation & EDA visualization charts
├── src/
│   ├── __init__.py
│   ├── config.py                      # Centralized configuration & parameters
│   ├── ingestion.py                   # Data Layer ingestion & validation engine
│   ├── data_loader.py                 # Split & ingestion integration module
│   ├── eda.py                         # EDA profiling & plot generation
│   ├── preprocessor.py                # Custom Scikit-Learn transformers & pipeline
│   ├── train.py                       # Multi-model training, CV, & tuning engine
│   ├── evaluate.py                    # Metric calculation & residual plots
│   └── predict.py                     # CLI & Python inference engine
├── app/
│   ├── __init__.py
│   ├── schemas.py                     # Pydantic schemas for request/response
│   └── main.py                        # FastAPI web application
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py               # Unit tests for ingestion, preprocessing & models
│   └── test_api.py                    # API endpoint integration tests
├── run_pipeline.py                    # Master End-to-End Orchestrator
├── streamlit_app.py                   # Interactive Streamlit Web UI Dashboard
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## Quickstart & Execution

### 1. Launch the Interactive Streamlit Web Dashboard
```bash
streamlit run streamlit_app.py
```
*Visual browser dashboard with sliders, dropdowns, quick vehicle presets, real-time price estimation, model benchmarks, and batch CSV valuation.*

### 2. Run the Complete End-to-End MLOps Pipeline
```bash
python run_pipeline.py
```
*Executes Data Layer Ingestion $\rightarrow$ 6-point Data Validation $\rightarrow$ EDA $\rightarrow$ 10-Model 5-Fold CV Benchmark $\rightarrow$ GridSearchCV Tuning $\rightarrow$ Artifact Serialization $\rightarrow$ Unseen Inference.*

### 3. Run Automated Test Suite (11/11 Tests Passing)
```bash
python -m pytest tests/ -v
```

### 4. Launch the Production FastAPI Service
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger API documentation available at `http://127.0.0.1:8000/docs`.*

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and model readiness |
| `GET` | `/model_info` | Active production model metadata and test performance |
| `GET` | `/metrics` | Full comparison metrics across all 10 trained models |
| `POST` | `/predict` | Predict price for a single vehicle |
| `POST` | `/batch_predict` | High-throughput batch price prediction |

### Sample Prediction Request (`POST /predict`)
```json
{
  "CarName": "toyota corolla",
  "symboling": 0,
  "fueltype": "gas",
  "aspiration": "std",
  "doornumber": "four",
  "carbody": "sedan",
  "drivewheel": "fwd",
  "enginelocation": "front",
  "wheelbase": 98.8,
  "carlength": 175.6,
  "carwidth": 65.5,
  "carheight": 53.9,
  "curbweight": 2200.0,
  "enginetype": "ohc",
  "cylindernumber": "four",
  "enginesize": 110.0,
  "fuelsystem": "2bbl",
  "boreratio": 3.19,
  "stroke": 3.03,
  "compressionratio": 9.0,
  "horsepower": 75.0,
  "peakrpm": 4800.0,
  "citympg": 32.0,
  "highwaympg": 38.0
}
```

### Sample Response
```json
{
  "car_identifier": "toyota corolla",
  "predicted_price": 7824.50,
  "predicted_price_formatted": "$7,824.50",
  "status": "success"
}
```

---

## Domain Features Engineered

1. **Power-to-Weight Ratio**: $\frac{\text{horsepower}}{\text{curbweight}}$
2. **Total Vehicle Volume**: $\text{carlength} \times \text{carwidth} \times \text{carheight}$
3. **Combined Weighted MPG**: $0.55 \times \text{citympg} + 0.45 \times \text{highwaympg}$
4. **Engine Displacement Ratio**: $\frac{\text{enginesize}}{\text{cylindernumber}}$
5. **Brand Luxury Tier Flag**: Boolean indicator derived from manufacturer luxury classification.
