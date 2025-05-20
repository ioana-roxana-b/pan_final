import os
import joblib
from src import inference, pipeline_pan25

def detect_problem_type(input_dir):
    parts = os.path.normpath(input_dir).split(os.sep)
    for part in ["easy", "medium", "hard"]:
        if part in parts:
            return part
    raise ValueError("Could not determine problem type (easy/medium/hard) from path: " + input_dir)

def test(args):
    print("\n==== STARTING TEST PIPELINE ====\n")
    base_input_dir = args.input
    base_output_dir = args.output
    base_dir = os.path.dirname(os.path.abspath(__file__))

    problem_types = ["easy", "medium", "hard"]

    for problem_type in problem_types:
        problem_input_path = os.path.join(base_input_dir, problem_type)

        if not os.path.exists(problem_input_path):
            print(f"[SKIP] Input subfolder not found: {problem_input_path}")
            continue

        print(f"\n--- Processing problem type: {problem_type} ---")

        if problem_type == "easy":
            model_path = os.path.join(base_dir, "models", "grad_boost_C3_easy.pkl")
            wan_config = "C3"
        elif problem_type == "medium":
            model_path = os.path.join(base_dir, "models", "grad_boost_C14_medium.pkl")
            wan_config = "C14"
        else:
            model_path = os.path.join(base_dir, "models", "grad_boost_C9_hard.pkl")
            wan_config = "C9"

        print("\n>> Extracting features in-memory...")
        test_features = pipeline_pan25.pipeline_pan(
            test_dir=problem_input_path,
            output_test_dir=None,
            wan_config=wan_config
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)

        output_dir = os.path.join(base_output_dir, problem_type)
        os.makedirs(output_dir, exist_ok=True)

        inference.run_inference(
            test_df=test_features,
            output_dir=output_dir,
            model=model,
            problem_type=problem_type
        )

    print("\n==== ALL TESTS COMPLETED SUCCESSFULLY ====\n")
