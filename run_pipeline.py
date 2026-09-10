"""
Master Enterprise MLOps Pipeline Orchestrator for Car Price Prediction.
Full-Scale Lifecycle:
DVC -> Ingestion -> Validation -> EDA -> Feature Eng -> 10-Model Benchmark -> MLflow Registry -> Drift Detection -> Multi-Agent AI -> Unseen Inference
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import logging
import pandas as pd

from notebooks.src.config import (
    RAW_DATA_PATH,
    MODEL_SAVE_PATH,
    METRICS_SAVE_PATH,
    EDA_SUMMARY_PATH,
    DATA_VALIDATION_REPORT_PATH
)
from notebooks.src.ingestion import DataIngestion
from notebooks.src.data_loader import load_raw_data, split_and_save_data
from notebooks.src.eda import run_eda
from notebooks.src.train import run_training_pipeline
from notebooks.src.predict import CarPricePredictor
from notebooks.src.drift_detector import run_drift_audit
from notebooks.src.agents.orchestrator import MLOpsMultiAgentOrchestrator
from app.db import record_inference_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MLOpsMasterPipeline")


def format_table(df: pd.DataFrame) -> str:
    """Helper to format pretty ascii tables."""
    return df.to_string(index=False)


def run_full_pipeline():
    logger.info("=" * 85)
    logger.info("       STARTING FULL-LENGTH ENTERPRISE MLOPS PIPELINE")
    logger.info("=" * 85)

    # STEP 1: Understand Dataset & Define Problem
    logger.info("\n>>> STEP 1: UNDERSTAND THE DATASET & DEFINE PROBLEM")
    logger.info("Objective: Supervised regression for vehicle valuation from 26 mechanical & market attributes.")

    # STEP 2: Data Layer Ingestion & Automated 6-Point Validation
    logger.info("\n>>> STEP 2: DATA LAYER INGESTION & 6-POINT DATA QUALITY GATE")
    ingestion = DataIngestion(RAW_DATA_PATH)
    df = ingestion.get_validated_data()
    val_report = ingestion.validation_report
    logger.info(f"Loaded raw canonical file: {ingestion.locate_raw_file().name} (Shape: {df.shape})")
    logger.info(f"Validation Status: {val_report['validation_status']} ({val_report['checks_summary']['passed_checks']}/{val_report['checks_summary']['total_checks']} quality gates passed)")

    # STEP 3: Exploratory Data Analysis & Statistical Profiling
    logger.info("\n>>> STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL PROFILING")
    eda_summary = run_eda(df)
    logger.info(f"Target Variable: {eda_summary['target_variable']} | Mean: ${eda_summary['price_statistics']['mean']:,.2f} | Median: ${eda_summary['price_statistics']['median']:,.2f}")

    # STEP 4 & 5: Preprocessing, Data Cleaning, and Feature Engineering
    logger.info("\n>>> STEP 4 & 5: DATA CLEANING & DOMAIN FEATURE ENGINEERING")
    logger.info("Automated typo correction (maxda->mazda, vokswagen->volkswagen, porcshce->porsche, toyouta->toyota).")
    logger.info("Engineered: power_to_weight, car_volume, avg_mpg, engine_displacement_ratio, is_luxury_brand.")

    # STEP 6 & 7: Multi-Model Benchmark, Regularization, 5-Fold CV & MLflow Tracking
    logger.info("\n>>> STEP 6 & 7: 10-MODEL BENCHMARK, REGULARIZATION, 5-FOLD CV & MLFLOW REGISTRY")
    training_summary = run_training_pipeline()
    
    benchmark_df = pd.DataFrame(training_summary["all_models_benchmark"])
    print("\n" + "=" * 95)
    print("                      ALL CANDIDATE MODELS BENCHMARK COMPARISON")
    print("=" * 95)
    print(format_table(benchmark_df))
    print("=" * 95)

    # STEP 8: Winning Model Selection & Justification
    best_name = training_summary["best_model_name"]
    best_metrics = training_summary["best_model_test_metrics"]
    
    logger.info("\n>>> STEP 8: PRODUCTION MODEL SELECTION & METRICS")
    print(f"\n WINNING PRODUCTION MODEL: {best_name}")
    print(f"   • Test R² Score:                    {best_metrics['r2_score']:.4f}")
    print(f"   • Adjusted R²:                      {best_metrics['adjusted_r2']:.4f}")
    print(f"   • Mean Absolute Error (MAE):        ${best_metrics['mae']:,.2f}")
    print(f"   • Root Mean Squared Error (RMSE):   ${best_metrics['rmse']:,.2f}")
    print(f"   • Mean Abs Percentage Error (MAPE): {best_metrics['mape_percent']:.2f}%")

    # STEP 9: Statistical Data Drift Detection (KS-Test & PSI)
    logger.info("\n>>> STEP 9: REAL-TIME STATISTICAL DATA DRIFT AUDIT")
    drift_report = run_drift_audit()
    print(f"\n DRIFT DETECTION STATUS: {drift_report['overall_drift_status']}")
    print(f"   • Drifted Features Count: {drift_report['drifted_features_count']}")
    print(f"   • Retraining Trigger Recommended: {drift_report['retraining_recommended']}")

    # STEP 10: Multi-Agent AI System & Unseen Inference
    logger.info("\n>>> STEP 10: MULTI-AGENT AI VALUATION & UNSEEN PREDICTIONS")
    predictor = CarPricePredictor(MODEL_SAVE_PATH)
    multi_agent_orchestrator = MLOpsMultiAgentOrchestrator()
    
    unseen_vehicles = [
        {
            "CarName": "toyota corolla le", "symboling": 0, "fueltype": "gas", "aspiration": "std",
            "doornumber": "four", "carbody": "sedan", "drivewheel": "fwd", "enginelocation": "front",
            "wheelbase": 98.8, "carlength": 175.6, "carwidth": 65.5, "carheight": 53.9,
            "curbweight": 2200, "enginetype": "ohc", "cylindernumber": "four", "enginesize": 110,
            "fuelsystem": "2bbl", "boreratio": 3.19, "stroke": 3.03, "compressionratio": 9.0,
            "horsepower": 75, "peakrpm": 4800, "citympg": 32, "highwaympg": 38
        },
        {
            "CarName": "porsche 911 carrera turbo", "symboling": 3, "fueltype": "gas", "aspiration": "turbo",
            "doornumber": "two", "carbody": "hardtop", "drivewheel": "rwd", "enginelocation": "rear",
            "wheelbase": 92.0, "carlength": 172.0, "carwidth": 68.5, "carheight": 51.0,
            "curbweight": 3100, "enginetype": "ohcf", "cylindernumber": "six", "enginesize": 220,
            "fuelsystem": "mpfi", "boreratio": 3.80, "stroke": 3.10, "compressionratio": 9.5,
            "horsepower": 240, "peakrpm": 6000, "citympg": 16, "highwaympg": 24
        },
        {
            "CarName": "bmw 530i luxury sedan", "symboling": 0, "fueltype": "gas", "aspiration": "turbo",
            "doornumber": "four", "carbody": "sedan", "drivewheel": "rwd", "enginelocation": "front",
            "wheelbase": 108.0, "carlength": 192.0, "carwidth": 69.5, "carheight": 55.0,
            "curbweight": 3450, "enginetype": "dohc", "cylindernumber": "six", "enginesize": 195,
            "fuelsystem": "mpfi", "boreratio": 3.55, "stroke": 3.30, "compressionratio": 9.2,
            "horsepower": 190, "peakrpm": 5500, "citympg": 19, "highwaympg": 26
        },
        {
            "CarName": "volkswagen golf tdi diesel", "symboling": 1, "fueltype": "diesel", "aspiration": "turbo",
            "doornumber": "four", "carbody": "hatchback", "drivewheel": "fwd", "enginelocation": "front",
            "wheelbase": 97.5, "carlength": 168.0, "carwidth": 65.5, "carheight": 54.5,
            "curbweight": 2400, "enginetype": "ohc", "cylindernumber": "four", "enginesize": 115,
            "fuelsystem": "idi", "boreratio": 3.10, "stroke": 3.40, "compressionratio": 22.0,
            "horsepower": 80, "peakrpm": 4500, "citympg": 36, "highwaympg": 44
        }
    ]
    
    sample_preds = predictor.predict_formatted(unseen_vehicles)
    print("\n" + "=" * 65)
    print("             SAMPLE VALUATION PREDICTIONS ON UNSEEN CARS")
    print("=" * 65)
    for p, car in zip(sample_preds, unseen_vehicles):
        # Log to Database
        record_inference_log(car_features=car, predicted_price=p["predicted_price"])
        print(f"Vehicle: {p['car_identifier']:<32} -> {p['predicted_price_formatted']:>12}")
    print("=" * 65)

    # Multi-Agent Sample Audit
    agent_evaluation = multi_agent_orchestrator.process_vehicle_valuation(unseen_vehicles[0], sample_preds[0]["predicted_price"])
    print("\n" + "=" * 80)
    print("                      MULTI-AGENT AI EVALUATION REPORT")
    print("=" * 80)
    print(f"Consensus Verdict: {agent_evaluation['consensus_verdict']}")
    print(f"Data Quality:      {agent_evaluation['data_quality_agent']['status']} ({agent_evaluation['data_quality_agent']['recommendation']})")
    print(f"Model Governance:  {agent_evaluation['model_governance_agent']['compliance_status']}")
    print(f"Market Segment:    {agent_evaluation['pricing_intelligence_agent']['market_segment']}")
    print(f"Pricing Summary:   {agent_evaluation['pricing_intelligence_agent']['market_summary']}")
    print("=" * 80)

    print("\n" + "=" * 85)
    print("                          FINAL MLOPS PROJECT CONCLUSION")
    print("=" * 85)
    print("1. Data & Pipeline Versioning (DVC): Complete stage DAG configured in dvc.yaml.")
    print("2. Experiment Tracking & Registry (MLflow): 10 models tracked, champion registered.")
    print("3. Database Persistence (FastAPI + SQLite/Postgres): Inference & feedback logged.")
    print("4. Statistical Drift Monitoring: Continuous KS-Test and PSI distribution checks.")
    print("5. Automated Retraining Engine: Champion-Challenger evaluation with auto-promotion.")
    print("6. Multi-Agent AI System: Collaborative DataQuality, Governance, DriftOps & Pricing Agents.")
    print("7. Deployment Ready: Containerized via Docker Compose & automated with GitHub Actions.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_full_pipeline()
