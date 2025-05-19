import os
import json

import joblib
import pandas as pd

from src import inference, pipeline_pan25

def load_features(directory, wan_config):
    """
    Loads and concatenates all CSV feature files from the specified directory.

    - Validates that the directory exists and contains `.csv` files
    - Ensures that training data includes a 'label' column
    - Merges all CSVs into a single DataFrame

    Params:
        directory (str): Path to the directory containing the CSV feature files.

    Returns:
        pd.DataFrame: A concatenated DataFrame of all features in the directory.
    """
    path = f'{directory}/{wan_config}'

    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature directory '{path}' not found.")

    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError(f"No CSV files found in directory '{path}'.")

    # Load all CSVs and concatenate
    dataframes = []
    for f in csv_files:
        file_path = os.path.join(path, f)
        df = pd.read_csv(file_path)

        # Ensure 'label' column exists for training data
        if 'label' not in df.columns and 'train' in path.lower():
            raise ValueError(f"Missing 'label' column in {file_path}")

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

def detect_problem_type(input_dir):
    parts = os.path.normpath(input_dir).split(os.sep)
    for part in ["easy", "medium", "hard"]:
        if part in parts:
            return part
    raise ValueError("Could not determine problem type (easy/medium/hard) from path: " + input_dir)

def test(args):
    print("\n==== STARTING TEST PIPELINE ====\n")
    problem_type = detect_problem_type(args.input)
    print(f"[INFO] Detected problem type: {problem_type}")

    if problem_type == "easy":
        model_path = "models/grad_boost_C14_medium.pkl"
        wan_config = "C3"
    elif problem_type == "medium":
        model_path = "models/grad_boost_C14_medium.pkl"
        wan_config = "C14"
    else:
        model_path = "models/grad_boost_C9_hard.pkl"
        wan_config = "C9"


    features_path = f"PAN25/{problem_type}"
    os.makedirs(features_path, exist_ok=True)
    pipeline_pan25.pipeline_pan(test_dir=args.input,
                                output_test_dir=features_path,
                                wan_config=wan_config)

    print("\n>> Loading extracted features...")
    test_features = load_features(directory=features_path, wan_config=wan_config)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    inference.run_inference(
        test_df=test_features,
        output_dir=args.output,
        model=model,
        problem_type=problem_type
    )

    print("\n==== TEST PIPELINE COMPLETED SUCCESSFULLY ====\n")
