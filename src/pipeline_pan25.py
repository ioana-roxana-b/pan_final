import os
import json
import numpy as np
import pandas as pd
import torch

from collections import Counter
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction import DictVectorizer
from sentence_transformers import SentenceTransformer
from src import preprocessing, features
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Global SBERT model for worker processes
sbert_model = None

def select_device():
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        if mem > 6000:
            return "cuda"
    return "cpu"

def init_worker():
    global sbert_model
    device = select_device()
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def compute_sbert_features_batch(sentences, include_embeddings=True, reduce_dim=None):
    global sbert_model
    embeddings = sbert_model.encode(sentences, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    
    # Optional dimensionality reduction
    if reduce_dim and isinstance(reduce_dim, int):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=reduce_dim)
        embeddings = pca.fit_transform(embeddings)

    features = []
    for i in range(len(embeddings) - 1):
        emb1, emb2 = embeddings[i], embeddings[i + 1]

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        denom = norm1 * norm2
        cosine_sim = float(np.dot(emb1, emb2) / denom) if denom != 0 else 0.0
        cosine_dist = 1 - cosine_sim

        euclid = np.linalg.norm(emb1 - emb2)
        manhattan = np.sum(np.abs(emb1 - emb2))
        norm_ratio = norm2 / norm1 if norm1 != 0 else 0.0

        pair_features = {
            "sbert_cosine_similarity": cosine_sim,
            "sbert_cosine_distance": cosine_dist,
            "sbert_euclidean": euclid,
            "sbert_manhattan": manhattan,
            "sbert_norm_ratio": norm_ratio
        }

        if include_embeddings:
            # Add raw embeddings
            for j, val in enumerate(emb1):
                pair_features[f"sbert_emb1_{j}"] = val
            for j, val in enumerate(emb2):
                pair_features[f"sbert_emb2_{j}"] = val

            # Optionally, include difference vector
            delta = emb2 - emb1
            for j, val in enumerate(delta):
                pair_features[f"sbert_delta_emb_{j}"] = val

        features.append(pair_features)

    return features

def compute_multi_skip_semantic_drift(embeddings, skips=[1, 2, 3]):
    drift_features = []
    num_sentences = len(embeddings)
    for i in range(num_sentences - 1):
        features = {}
        for skip in skips:
            if i + skip < num_sentences:
                emb_i = embeddings[i]
                emb_j = embeddings[i + skip]
                diff_vec = emb_j - emb_i
                distance = np.linalg.norm(diff_vec)
                cosine = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)) if np.linalg.norm(emb_i) * np.linalg.norm(emb_j) > 0 else 0.0
                features[f"semantic_drift_distance_skip{skip}"] = distance
                features[f"semantic_drift_cosine_skip{skip}"] = cosine
        drift_features.append(features)
    return drift_features


def smooth_features(paired_features, feature_keys, window_size=3):
    """
    Smooths selected feature columns using a moving average.

    Args:
        paired_features (List[Dict]): List of feature dicts for sentence pairs.
        feature_keys (List[str]): List of feature names to smooth.
        window_size (int): Size of moving window (must be odd).

    Returns:
        List[Dict]: Updated paired_features with smoothed feature values.
    """
    assert window_size % 2 == 1, "Window size must be odd!"

    half_window = window_size // 2
    num_samples = len(paired_features)

    for key in feature_keys:
        values = np.array([pf.get(key, 0) for pf in paired_features])

        smoothed_values = []
        for i in range(num_samples):
            start = max(0, i - half_window)
            end = min(num_samples, i + half_window + 1)
            smoothed = np.mean(values[start:end])
            smoothed_values.append(smoothed)

        # Store smoothed version under new name
        for i in range(num_samples):
            paired_features[i][f"{key}_smooth"] = smoothed_values[i]

    return paired_features


def collect_problem_ids(directory):
    entries = os.listdir(directory)
    files = [f for f in entries if f.endswith(".txt")]

    if files:
        print(f"[INFO] Found {len(files)} .txt files in: {directory}")
        return [f.replace(".txt", "") for f in files if f.startswith("problem-")], directory

    subdirs = [d for d in entries if os.path.isdir(os.path.join(directory, d))]
    if len(subdirs) == 1:
        inner_dir = os.path.join(directory, subdirs[0])
        inner_files = os.listdir(inner_dir)
        txts = [f for f in inner_files if f.endswith(".txt") and f.startswith("problem-")]
        print(f"[INFO] Found {len(txts)} .txt files in subdirectory: {subdirs[0]}")
        return [f.replace(".txt", "") for f in txts], inner_dir

    print(f"[WARNING] No valid .txt files found in: {directory}")
    return [], directory


def load_wan_config(config_name):
    all_configs = {
        "C1": {"punctuations": "True", "stopwords": "True", "lemmatizer": "True", "include_pos": "True"},
        "C2": {"punctuations": "True", "stopwords": "True", "lemmatizer": "False", "include_pos": "True"},
        "C3": {"punctuations": "True", "stopwords": "False", "lemmatizer": "False", "include_pos": "True"},
        "C4": {"punctuations": "True", "stopwords": "False", "lemmatizer": "True", "include_pos": "True"},
        "C5": {"punctuations": "False", "stopwords": "True", "lemmatizer": "True", "include_pos": "True"},
        "C6": {"punctuations": "False", "stopwords": "True", "lemmatizer": "False", "include_pos": "True"},
        "C7": {"punctuations": "False", "stopwords": "False", "lemmatizer": "False", "include_pos": "True"},
        "C8": {"punctuations": "False", "stopwords": "False", "lemmatizer": "True", "include_pos": "True"},
        "C9": {"punctuations": "True", "stopwords": "True", "lemmatizer": "True", "include_pos": "False"},
        "C10": {"punctuations": "True", "stopwords": "True", "lemmatizer": "False", "include_pos": "False"},
        "C11": {"punctuations": "True", "stopwords": "False", "lemmatizer": "False", "include_pos": "False"},
        "C12": {"punctuations": "True", "stopwords": "False", "lemmatizer": "True", "include_pos": "False"},
        "C13": {"punctuations": "False", "stopwords": "True", "lemmatizer": "True", "include_pos": "False"},
        "C14": {"punctuations": "False", "stopwords": "True", "lemmatizer": "False", "include_pos": "False"},
        "C15": {"punctuations": "False", "stopwords": "False", "lemmatizer": "False", "include_pos": "False"},
        "C16": {"punctuations": "False", "stopwords": "False", "lemmatizer": "True", "include_pos": "False"},
    }

    config = all_configs.get(config_name, {})
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Invalid WAN config: {config_name}")

    # Convert strings to booleans
    for key in ["punctuations", "stopwords", "lemmatizer", "include_pos"]:
        config[key] = str(config.get(key, "False")).lower() == "true"

    config['name'] = config_name
    return config

def process_problem(problem_id, data_dir, output_dir, config):
    global sbert_model
    try:
        print(f"\n>> [START] Processing {problem_id}...")  
        text_file = os.path.join(data_dir, f"{problem_id}.txt")
        truth_file = os.path.join(data_dir, f"truth-{problem_id}.json")

        if not os.path.exists(text_file):
            print(f"ERROR: {text_file} not found.")
            return None

        with open(text_file, "r") as f:
            sentences = f.read().strip().split('\n')

        if len(sentences) < 2:
            print(f"Insufficient sentences in {problem_id}.")
            return None

        labels = None
        if os.path.exists(truth_file):
            with open(truth_file, 'r') as f:
                labels = json.load(f).get("changes", [])

        preprocessed = preprocessing.preprocessing(
            text=sentences,
            stopwords=config['stopwords'],
            lemmatizer=config['lemmatizer'],
            punctuations=config['punctuations']
        )

        wans = preprocessing.construct_wans(preprocessed, output_dir=output_dir, include_pos=config['include_pos'])
        sentence_features = features.extract_features(wans, include_pos=config['include_pos'])
        sentence_features = features.extract_lexical_syntactic_features(sentence_features, include_pos=config['include_pos'])
    
        for i, sentence in enumerate(sentences):
            sentence_features[i].update(features.extract_lexical_features(sentence))
            sentence_features[i].update(features.extract_deep_style_features(sentence, index=i, all_sentences=sentences))

        contextual_feats = features.extract_contextual_features(sentences)
        for i in range(len(sentences)):
            sentence_features[i].update(contextual_feats[i])

        # Existing
        sbert_features = compute_sbert_features_batch(sentences, include_embeddings=False)

        # --- SBERT encoding ---
        embeddings = sbert_model.encode(sentences, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

        # --- Compute multi-skip semantic drift ---
        multi_skip_feats = compute_multi_skip_semantic_drift(embeddings, skips=[1, 2, 3])

        # --- Directional cosine drift ---
        directional_cosines = []
        for i in range(len(embeddings) - 2):
            delta1 = embeddings[i + 1] - embeddings[i]
            delta2 = embeddings[i + 2] - embeddings[i + 1]
            norm1 = np.linalg.norm(delta1)
            norm2 = np.linalg.norm(delta2)
            denom = norm1 * norm2
            cos_angle = float(np.dot(delta1, delta2) / denom) if denom != 0 else 0.0
            directional_cosines.append(cos_angle)

        # NEW: Compute burstiness and variance
        punctuation_burstiness = features.compute_punctuation_burstiness(sentences, window_size=3)
        length_variance = features.compute_length_variance(sentences, window_size=3)

        pos_tfidf_features = features.extract_pos_tfidf_features(sentences)
        pos_entropies = [features.compute_pos_entropy(sent) for sent in sentences]

        wan_drift_feats = features.compute_wan_window_drift(sentences, window_size=3, step=1)

        paired_features = []

        deep_keys = ["formality_score", "clause_complexity"]

        for i in range(len(sentences) - 1):
            f1, f2 = sentence_features[i], sentence_features[i + 1]
            common_keys = [k for k in f1 if k in f2 and isinstance(f1[k], (int, float)) and isinstance(f2[k], (int, float))]

            f1_vec = np.nan_to_num([f1[k] for k in common_keys], nan=0.0)
            f2_vec = np.nan_to_num([f2[k] for k in common_keys], nan=0.0)

            denom = np.linalg.norm(f1_vec) * np.linalg.norm(f2_vec)
            cos_sim = float(np.dot(f1_vec, f2_vec) / denom) if denom != 0 else 0.0
            euc_dist = euclidean(f1_vec, f2_vec)

            trig1, trig2 = f1.get("pos_trigram_counter", Counter()), f2.get("pos_trigram_counter", Counter())
            pos_cosine, pos_jaccard = 0.0, 0.0
            if trig1 and trig2:
                all_keys = set(trig1) | set(trig2)
                vecs = DictVectorizer(sparse=False).fit_transform([trig1, trig2])
                denom = np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
                pos_cosine = float(np.dot(vecs[0], vecs[1]) / denom) if denom != 0 else 0.0
                intersection = len(set(trig1) & set(trig2))
                pos_jaccard = intersection / len(all_keys) if all_keys else 0.0

            f1_deep = np.array([f1.get(k, 0) for k in deep_keys])
            f2_deep = np.array([f2.get(k, 0) for k in deep_keys])
            denom = np.linalg.norm(f1_deep) * np.linalg.norm(f2_deep)
            deep_style_cosine = float(np.dot(f1_deep, f2_deep) / denom) if denom != 0 else 0.0
            pos_entropy_diff = pos_entropies[i+1] - pos_entropies[i]
            pos_entropy_abs_diff = abs(pos_entropies[i+1] - pos_entropies[i])

            combined = {
                "problem_id": problem_id,
                "sentence_index": i,
                "cosine_similarity": cos_sim,
                "euclidean_distance": euc_dist,
                "pos_trigram_cosine": pos_cosine,
                "pos_trigram_jaccard": pos_jaccard,
                "deep_style_cosine": deep_style_cosine,
                "pos_tfidf_cosine": pos_tfidf_features[i],
                "pos_entropy_diff": pos_entropy_diff,
                "pos_entropy_abs_diff": pos_entropy_abs_diff,
                "delta_punctuation_burstiness": punctuation_burstiness[i+1] - punctuation_burstiness[i],
                "delta_length_variance": length_variance[i+1] - length_variance[i],
                "position_ratio": i / max(1, len(sentences) - 1),
                **sbert_features[i],
                **{f"f1_{k}": v for k, v in f1.items() if isinstance(v, (int, float))},
                **{f"f2_{k}": v for k, v in f2.items() if isinstance(v, (int, float))},
                **{f"delta_{k}": f2.get(k, 0.0) - f1.get(k, 0.0) for k in common_keys},
                **{f"diff_{k}": abs(f1[k] - f2[k]) for k in common_keys},
                **{f"{k}_increased": int(f2[k] > f1[k]) for k in ["average_clustering", "average_degree", "pos_density", "pos_avg_degree"] if k in f1 and k in f2},
                "label": labels[i] if labels and i < len(labels) else None
            }

            wan1, wan2 = wans.get(i), wans.get(i + 1)
            if wan1 and wan2:
                combined.update(features.extract_wan_pairwise_features(
                    wan1, wan2, compute_edit_distance=False, compute_spectral=False
                ))

            if i < len(multi_skip_feats):
                combined.update(multi_skip_feats[i])

            match = next((w for w in wan_drift_feats if w["wan_window_index"] == i), {})
            if match:
                combined.update({
                    "wan_cosine_drift": match.get("wan_window_cosine_sim", 0.0),
                    "wan_euclidean_drift": match.get("wan_window_euclidean", 0.0)
                })

            if i < len(directional_cosines):
                combined["sbert_directional_cosine"] = directional_cosines[i]
            else:
                combined["sbert_directional_cosine"] = 0.0

            readability_keys = [
                "flesch_reading_ease",
                "gunning_fog",
                "automated_readability_index",
                "syllable_count",
                "dale_chall_score"
            ]
            for key in readability_keys:
                combined[f"delta_{key}"] = f2.get(key, 0.0) - f1.get(key, 0.0)

            paired_features.append(combined)

        # --- Smooth important features ---
        paired_features = smooth_features(
            paired_features,
            feature_keys=[
                "sbert_cosine_similarity",
                "cosine_similarity",
                "sbert_cosine_distance",
                "sbert_directional_cosine",
                "wan_cosine_drift",
                "deep_style_cosine",
                "pos_trigram_cosine",
                "wan_euclidean_drift"
            ],
            window_size=3
        )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(paired_features).fillna(0)
        if 'label' in df.columns:
            cols = [c for c in df.columns if c != 'label'] + ['label']
            df = df[cols]

        print(f"[DONE] Features extracted for {problem_id}")
        return df

    except Exception as e:
        print(f"[ERROR] Failed {problem_id}: {e}")
        return None

def pipeline_pan(test_dir, output_test_dir, wan_config):
    print("==== Starting Pipeline ====")
    config = load_wan_config(wan_config)
    test_ids, actual_data_dir = collect_problem_ids(test_dir)

    # print(f"[DEBUG] Looking in: {test_dir}")
    # print("Files in dir:", os.listdir(test_dir))
    #
    # print(f"[DEBUG] Found problem files: {test_ids}")

    global sbert_model
    device = select_device()
    sbert_model = SentenceTransformer("models/all-MiniLM-L6-v2", device=device)

    results = []
    for pid in test_ids:
        df = process_problem(pid, actual_data_dir, output_test_dir, config)
        if df is not None:
            print(f"[INFO] Feature shape for {pid}: {df.shape}")
            results.append(df)
        else:
            print(f"[WARNING] Skipped {pid} (no features or error).")

    print(f"[SUMMARY] Successfully processed {len(results)} / {len(test_ids)} problems.")

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()
