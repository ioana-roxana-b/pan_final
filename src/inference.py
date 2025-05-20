import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def run_inference(test_df, output_dir, model, problem_type):

    # Step 1: Clean up features
    X_test = test_df.drop(columns=["problem_id", "sentence_index", "label"], errors='ignore')
    X_test = X_test.select_dtypes(include=[np.number])
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if X_test.shape[0] == 0 or X_test.shape[1] == 0:
        print("[WARNING] No features available for inference. Skipping.")
        return

    # Step 2: Apply MinMaxScaler for easy/medium only
    if problem_type in ["easy", "medium"]:
        print(f"[INFO] Loading MinMaxScaler for problem type: {problem_type}")
        if problem_type == "easy":
            scaler_path = f"models/minmax_scaler_C3_easy.pkl"
        else:
            scaler_path = f"models/minmax_scaler_C14_medium.pkl"

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)

    # Step 3: Predict
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.35).astype(int)
    else:
        y_pred = model.predict(X_test)

    test_df = pd.concat([test_df, pd.Series(y_pred, name="y_pred")], axis=1)

    # Step 5: Write predictions
    for pid, group in test_df.groupby("problem_id"):
        solution = {
            "changes": group["y_pred"].tolist()
        }
        with open(os.path.join(output_dir, f"solution-{pid}.json"), "w") as f:
            json.dump(solution, f, indent=4, separators=(",", ": "))

    print(f"[DONE] Predictions saved to: {output_dir}")
