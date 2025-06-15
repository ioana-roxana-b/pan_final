# Multi-Author Writing Style Analysis - PAN 2025

This project is developed for the PAN 2025 competition, focusing on the task of multi-author writing style analysis. The goal is to detect changes in authorship within a given text by analyzing stylistic features and using machine learning models to predict where these changes occur.

## Overview

The system processes input texts, extracts a variety of linguistic and stylistic features, and applies pre-trained models to infer authorship changes. It supports different difficulty levels (easy, medium, hard) with tailored models and configurations for each.

## Project Structure

- **src/**: Contains the core source code for the project.
  - **preprocessing.py**: Handles text preprocessing tasks such as tokenization, lemmatization, and stopword removal.
  - **features.py**: Extracts various features from text, including lexical, syntactic, and semantic features.
  - **pipeline_pan25.py**: Defines the main pipeline for processing texts and extracting features for the PAN 2025 task.
  - **inference.py**: Runs inference using pre-trained models to predict authorship changes.
  - **test.py**: Manages the testing process across different problem types.
- **models/**: Stores pre-trained machine learning models and scalers for different difficulty levels.
- **main.py**: Entry point for running the analysis on input data.
- **requirements.txt**: Lists the dependencies required for the project.
- **Dockerfile**: Configuration for containerizing the application.
- **Makefile**: Contains build and run commands for the project.
- **eval.sh** and **run_test.sh**: Scripts for evaluation and testing.

## Features

- **Text Preprocessing**: Customizable preprocessing with options for handling punctuation, stopwords, lemmatization, and POS tagging.
- **Feature Extraction**: Comprehensive feature set including:
  - Lexical and syntactic features
  - Semantic features using SBERT embeddings
  - Contextual and deep style features
- **Difficulty Levels**: Separate models and configurations for easy, medium, and hard problem sets.
- **Inference**: Predicts authorship changes using gradient boosting models with tailored thresholds.

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the analysis on a dataset:

```bash
python main.py -i <input_directory> -o <output_directory>
```

- `<input_directory>`: Path to the directory containing the input text files, organized by difficulty level (easy, medium, hard).
- `<output_directory>`: Path where the output predictions will be saved as JSON files.

The input directory should have subdirectories for each difficulty level (e.g., `input/easy`, `input/medium`, `input/hard`), containing text files named `problem-<id>.txt` and corresponding truth files `truth-<id>.json` if available.

## Output Format

For each processed problem, a JSON file named `solution-<id>.json` will be created in the output directory under the respective difficulty folder. The JSON file contains a list of binary values indicating predicted authorship changes between consecutive sentences.

Example:
```json
{
    "changes": [0, 1, 0, ...]
}
```

## Models

The project uses pre-trained gradient boosting models for each difficulty level:
- Easy: `grad_boost_C3_easy.pkl` with configuration C3
- Medium: `grad_boost_C14_medium.pkl` with configuration C14
- Hard: `grad_boost_C9_hard.pkl` with configuration C9

Additionally, MinMax scalers are applied for easy and medium levels to normalize features before inference.
