"""
Evaluation CLI and module for trained IDS models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .inference import predict_dataframe
except ImportError:
    from inference import predict_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "test_selected.csv"


def evaluate_model(
    model_path: Path,
    test_file: Path = TEST_FILE,
    report_dir: Path = REPORT_DIR,
) -> Dict[str, Any]:
    """
    Evaluate a saved model artifact against test data and save detailed metrics.
    """
    model_path = Path(model_path)
    test_file = Path(test_file)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test dataset file not found: {test_file}")

    artifact = joblib.load(model_path)
    model_name = artifact.get("model_name", model_path.stem)

    test_df = pd.read_csv(test_file)
    target_col = "Attack_Category"
    if target_col not in test_df.columns:
        raise ValueError(f"Test dataset must contain '{target_col}' column.")

    y_true = test_df[target_col].values
    X_test = test_df.drop(columns=[target_col])

    results = predict_dataframe(artifact, X_test)
    if isinstance(results, tuple):
        results = results[0]
    y_pred = results["Predicted_Category"].values

    classes = sorted(list(set(y_true) | set(y_pred)))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    class_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_report_str = classification_report(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    results = {
        "model_name": model_name,
        "model_file": str(model_path.name),
        "test_samples": len(y_true),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": class_report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }

    # Save TXT report
    safe_stem = model_path.stem.replace("_model", "")
    txt_report_path = report_dir / f"{safe_stem}_eval_report.txt"
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name.upper()} EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:        {accuracy:.6f}\n")
        f.write(f"Macro Precision: {macro_precision:.6f}\n")
        f.write(f"Macro Recall:    {macro_recall:.6f}\n")
        f.write(f"Macro F1:        {macro_f1:.6f}\n")
        f.write(f"Weighted F1:     {weighted_f1:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report_str + "\n\n")
        f.write("Confusion Matrix (Rows=True, Cols=Pred):\n")
        f.write(f"Classes: {classes}\n")
        f.write(np.array2string(cm) + "\n")

    # Save JSON metrics
    json_report_path = report_dir / f"{safe_stem}_metrics.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Confusion Matrix CSV
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    cm_path = report_dir / f"{safe_stem}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    print(f"\nEvaluated {model_name}: Accuracy={accuracy:.4f}, Macro_F1={macro_f1:.4f}")
    print(f"Saved reports to:\n  - {txt_report_path}\n  - {json_report_path}\n  - {cm_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained IDS models.")
    parser.add_argument("--model-path", type=str, help="Path to a specific model .joblib file.")
    parser.add_argument("--all", action="store_true", help="Evaluate all models in the models directory.")
    args = parser.parse_args()

    if args.model_path:
        evaluate_model(Path(args.model_path))
    elif args.all or not args.model_path:
        model_files = sorted(list(MODEL_DIR.glob("*.joblib")))
        if not model_files:
            print("No model files found in models directory.")
            return
        for m in model_files:
            evaluate_model(m)


if __name__ == "__main__":
    main()
