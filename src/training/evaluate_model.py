"""
Model evaluation module.
Evaluates both the historical persistence baseline and the trained Random Forest
on the held-out chronological test set (June 10 to June 27, 2012).
Computes regression metrics (MAE, RMSE, MAPE, R2) and congestion classification metrics.
Generates interactive Plotly evaluation figures and saves JSON evaluation report.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.common.config import config, PROJECT_ROOT
from src.common.logging_config import setup_logger
from src.common.schemas import get_congestion_level
from src.training.train_model import FEATURE_COLS, TARGET_COL, compute_metrics

logger = setup_logger("evaluate_model")

def evaluate():
    test_path = PROJECT_ROOT / "data/processed/test_features.parquet"
    model_path = PROJECT_ROOT / config["modeling"]["model_path"]
    report_path = PROJECT_ROOT / config["modeling"]["evaluation_report_path"]
    figures_dir = PROJECT_ROOT / "outputs/figures"

    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading test set from {test_path}...")
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loading trained Random Forest model from {model_path}...")
    rf_model = joblib.load(model_path)

    X_test = test_df[FEATURE_COLS]
    y_true = test_df[TARGET_COL].values

    # 1. Baseline Predictions
    y_pred_baseline = test_df["current_speed"].values
    baseline_metrics = compute_metrics(y_true, y_pred_baseline)

    # 2. Random Forest Predictions
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = compute_metrics(y_true, y_pred_rf)

    logger.info("=== HELD-OUT CHRONOLOGICAL TEST EVALUATION ===")
    logger.info(f"Baseline:      MAE={baseline_metrics['mae']:.3f} mph, RMSE={baseline_metrics['rmse']:.3f} mph, MAPE={baseline_metrics['mape_percent']:.2f}%, R2={baseline_metrics['r2']:.4f}")
    logger.info(f"Random Forest: MAE={rf_metrics['mae']:.3f} mph, RMSE={rf_metrics['rmse']:.3f} mph, MAPE={rf_metrics['mape_percent']:.2f}%, R2={rf_metrics['r2']:.4f}")

    # 3. Congestion Level Evaluation
    logger.info("Evaluating derived congestion classification performance...")
    true_congestion = [get_congestion_level(s) for s in y_true]
    pred_rf_congestion = [get_congestion_level(s) for s in y_pred_rf]
    pred_base_congestion = [get_congestion_level(s) for s in y_pred_baseline]

    labels = ["LOW", "MODERATE", "HIGH", "SEVERE"]
    clf_report_rf = classification_report(true_congestion, pred_rf_congestion, labels=labels, output_dict=True, zero_division=0)
    cm_rf = confusion_matrix(true_congestion, pred_rf_congestion, labels=labels).tolist()

    # Breakdown of errors by speed regime
    test_df["true_speed"] = y_true
    test_df["rf_pred"] = y_pred_rf
    test_df["baseline_pred"] = y_pred_baseline
    test_df["rf_abs_error"] = np.abs(y_true - y_pred_rf)
    test_df["base_abs_error"] = np.abs(y_true - y_pred_baseline)
    test_df["true_congestion"] = true_congestion

    regime_breakdown = {}
    for cong in labels:
        subset = test_df[test_df["true_congestion"] == cong]
        if len(subset) > 0:
            regime_breakdown[cong] = {
                "count": int(len(subset)),
                "baseline_mae": round(float(subset["base_abs_error"].mean()), 3),
                "rf_mae": round(float(subset["rf_abs_error"].mean()), 3),
                "rf_improvement_mph": round(float(subset["base_abs_error"].mean() - subset["rf_abs_error"].mean()), 3)
            }

    # 4. Generate Interactive Plotly Figures
    logger.info("Generating evaluation figures...")

    # Figure 1: Actual vs Predicted Scatter (sampled for responsive rendering)
    sample_eval = test_df.sample(n=min(3000, len(test_df)), random_state=42)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=sample_eval["true_speed"],
        y=sample_eval["rf_pred"],
        mode="markers",
        marker=dict(size=4, opacity=0.5, color=sample_eval["rf_abs_error"], colorscale="Viridis", showscale=True),
        name="Random Forest"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=[0, 80], y=[0, 80],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Ideal 1:1 Line"
    ))
    fig_scatter.update_layout(
        title="Actual vs Predicted Speed (Held-Out Test Set)",
        xaxis_title="Actual Speed (mph)",
        yaxis_title="Predicted Speed (mph)",
        template="plotly_white"
    )
    fig_scatter.write_html(str(figures_dir / "actual_vs_predicted.html"))

    # Figure 2: Residuals Distribution
    residuals = y_true - y_pred_rf
    fig_res = px.histogram(
        residuals, nbins=80,
        title="Prediction Residuals Distribution (Actual - Predicted mph)",
        labels={"value": "Residual Error (mph)"},
        template="plotly_white",
        marginal="box"
    )
    fig_res.write_html(str(figures_dir / "residuals_distribution.html"))

    # Figure 3: Hourly Error Profile
    hourly_perf = test_df.groupby("hour_of_day")[["base_abs_error", "rf_abs_error"]].mean().reset_index()
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_perf["hour_of_day"], y=hourly_perf["base_abs_error"],
        name="Baseline MAE", marker_color="#94a3b8"
    ))
    fig_hourly.add_trace(go.Bar(
        x=hourly_perf["hour_of_day"], y=hourly_perf["rf_abs_error"],
        name="Random Forest MAE", marker_color="#2563eb"
    ))
    fig_hourly.update_layout(
        barmode="group",
        title="Mean Absolute Error by Hour of Day (Rush Hour vs Night)",
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="MAE (mph)",
        template="plotly_white"
    )
    fig_hourly.write_html(str(figures_dir / "hourly_error_profile.html"))

    # 5. Compile Complete Evaluation Report
    report = {
        "evaluation_dataset": "METR-LA Held-Out Chronological Test Split",
        "test_period": {
            "start": str(test_df["timestamp"].min()),
            "end": str(test_df["timestamp"].max()),
            "total_test_records": int(len(test_df))
        },
        "metrics_summary": {
            "baseline": baseline_metrics,
            "random_forest": rf_metrics,
            "improvements": {
                "mae_reduction_mph": round(baseline_metrics["mae"] - rf_metrics["mae"], 3),
                "mae_reduction_percent": round(((baseline_metrics["mae"] - rf_metrics["mae"]) / baseline_metrics["mae"]) * 100, 2),
                "rmse_reduction_mph": round(baseline_metrics["rmse"] - rf_metrics["rmse"], 3),
                "rmse_reduction_percent": round(((baseline_metrics["rmse"] - rf_metrics["rmse"]) / baseline_metrics["rmse"]) * 100, 2),
                "r2_gain": round(rf_metrics["r2"] - baseline_metrics["r2"], 4)
            }
        },
        "congestion_classification_metrics": {
            "labels": labels,
            "confusion_matrix": cm_rf,
            "overall_accuracy": round(float(clf_report_rf["accuracy"]), 4),
            "per_class_report": clf_report_rf
        },
        "regime_error_breakdown": regime_breakdown
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {report_path}")
    logger.info(f"Interactive figures saved to {figures_dir}")
    return report

if __name__ == "__main__":
    evaluate()
