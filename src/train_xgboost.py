"""
XGBoost training module.
"""

from __future__ import annotations

from pathlib import Path
import time
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

try:
    from .data_pipeline import PROCESSED_DATA_DIR
except ImportError:
    from data_pipeline import PROCESSED_DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = PROCESSED_DATA_DIR / "train_selected.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_selected.csv"


def train_xgboost() -> None:
    print("Loading datasets...")
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    target = "Attack_Category"
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("\nTraining XGBoost...")
    start_time = time.time()
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_encoded)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    accuracy = float(accuracy_score(y_test_encoded, y_pred))
    precision = float(precision_score(y_test_encoded, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_test_encoded, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test_encoded, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importance_file = REPORT_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": list(X_train.columns),
        "model_name": "XGBoost",
        "metrics": {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
    }

    model_file = MODEL_DIR / "xgboost_model.joblib"
    joblib.dump(artifact, model_file)

    report_file = REPORT_DIR / "xgboost_report.txt"
    report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("XGBOOST RESULTS\n========================================\n\n")
        f.write(f"Accuracy:        {accuracy:.6f}\n")
        f.write(f"Macro Precision: {precision:.6f}\n")
        f.write(f"Macro Recall:    {recall:.6f}\n")
        f.write(f"Macro F1:        {macro_f1:.6f}\n")
        f.write(f"Weighted F1:     {weighted_f1:.6f}\n")
        f.write(f"Training Time:   {training_time:.4f} sec\n")
        f.write(f"Prediction Time: {prediction_time:.4f} sec\n\nClassification Report:\n")
        f.write(report)
        f.write("\n\nTop 15 Features:\n")
        f.write(importance_df.head(15).to_string(index=False))

    print(f"\nSaved XGBoost Model: {model_file}")


if __name__ == "__main__":
    train_xgboost()
