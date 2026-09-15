"""
Random Forest training module.
"""

from __future__ import annotations

from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


def train_random_forest() -> None:
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

    print("\nTraining Random Forest...")
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
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

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": list(X_train.columns),
        "model_name": "Random Forest",
        "metrics": {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
    }

    model_file = MODEL_DIR / "random_forest_model.joblib"
    joblib.dump(artifact, model_file)

    report_file = REPORT_DIR / "random_forest_report.txt"
    report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST RESULTS\n========================================\n\n")
        f.write(f"Accuracy:        {accuracy:.6f}\n")
        f.write(f"Macro Precision: {precision:.6f}\n")
        f.write(f"Macro Recall:    {recall:.6f}\n")
        f.write(f"Macro F1:        {macro_f1:.6f}\n")
        f.write(f"Weighted F1:     {weighted_f1:.6f}\n")
        f.write(f"Training Time:   {training_time:.4f} sec\n")
        f.write(f"Prediction Time: {prediction_time:.4f} sec\n\nClassification Report:\n")
        f.write(report)

    print(f"\nSaved Random Forest Model: {model_file}")


if __name__ == "__main__":
    train_random_forest()
