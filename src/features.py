from collections import Counter

import networkx as nx
import numpy as np

from scipy.stats import entropy
import re
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy
from spacy.lang.en.stop_words import STOP_WORDS
import textstat
from scipy.spatial.distance import cosine

import emoji
from collections import Counter
import numpy as np
from src import preprocessing
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

WH_WORDS = {"who", "what", "when", "where", "why", "how"}
FUNCTION_POS_TAGS = {"PRON", "ADP", "DET", "CCONJ", "SCONJ", "PART", "AUX"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours", "u", "ur"}
THIRD_PERSON = {"he", "him", "his", "she", "her", "they", "them", "their", "theirs"}

MODAL_WORDS = {
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "ought", "probably", "possibly", "certainly", "definitely", "undoubtedly",
    "clearly", "evidently"
}

DISCOURSE_MARKERS = {
    "however", "moreover", "therefore", "thus", "meanwhile", "in fact", "furthermore",
    "nevertheless", "anyway", "nonetheless", "on the other hand", "by contrast",
    "similarly", "also", "instead", "for example", "for instance"
}

SARCASM_MARKERS = [
    "yeah right", "sure jan", "as if", "oh great", "lucky me", "just perfect", "awesome", "love that for you",
    "i'm so sure", "can't wait", "what a surprise", "wow", "fantastic", "brilliant", "amazing"
]

LIWC_ANALYTIC = {"therefore", "thus", "consequently", "in conclusion", "resulted", "hence"}
LIWC_AFFECTIVE = {"happy", "sad", "angry", "fear", "hate", "love", "joy", "cry", "laugh", "terrible", "amazing"}
LIWC_AUTHENTICITY = FIRST_PERSON
LIWC_CERTAINTY = {"always", "never", "definitely", "certainly", "must", "undeniable"}
LIWC_TENTATIVENESS = {"maybe", "perhaps", "possibly", "seems", "guess", "kind of", "sort of", "i think"}

def get_pattern_type(text):
    """
    Classifies a sentence into one of several pattern types based on common moderator or bot language.

    Args:
        text (str): Input text string.

    Returns:
        int: Pattern type identifier (0 if no pattern matched).
    """

    s = text.strip().lower()

    # Rule 1: Clearly mod-automated content
    if s.startswith("r/") or "moderator applications" in s or "consider !" in s:
        return 1

    # Rule 2: Matches common mod speech blocks (strong anchor phrases)
    rule_phrases = [
        r"\bin general, be courteous\b",
        r"\bpermanent ban\b",
        r"\bviolations of rule[s]?\b",
        r"\bquestions regarding.*media outlets\b",
        r"\bapproved domains list\b",
        r"\bplease review our rules\b"
    ]
    for pattern in rule_phrases:
        if re.search(pattern, s):
            return 2

    # Rule 3: Bot speech
    if "i am a bot" in s or s.startswith("please if you have any questions"):
        return 3

    # Rule 4: Edit markers
    if s.startswith("edit") or s.startswith("update:"):
        return 4

    return 0



def extract_deep_style_features(sentence, index=None, all_sentences=None):
    """
    Extracts a rich set of deep style features from a sentence, including formality, discourse markers,
    LIWC-style categories, emoji usage, sentiment, and pattern-based flags.

    Args:
        sentence (str): Input sentence.
        index (int, optional): Index of the sentence in a list.
        all_sentences (list, optional): List of surrounding sentences.

    Returns:
        dict: Dictionary of extracted style features.
    """

    doc = nlp(sentence)
    tokens = [token for token in doc if not token.is_space]
    token_len = len(tokens)
    pos_tags = [token.pos_ for token in tokens]

    # --- Formality
    pos_counts = Counter(pos_tags)
    formality = (
        pos_counts.get("NOUN", 0) + pos_counts.get("ADJ", 0) +
        pos_counts.get("ADP", 0) + pos_counts.get("DET", 0)
        - pos_counts.get("PRON", 0) - pos_counts.get("VERB", 0)
        - pos_counts.get("ADV", 0) - pos_counts.get("INTJ", 0)
    )
    formality_score = 50 + 0.5 * formality / token_len if token_len else 50

    # --- NER Density
    ner_count = len([ent for ent in doc.ents])
    ner_density = ner_count / token_len if token_len else 0

    # --- Modality
    modal_count = sum(1 for token in tokens if token.lemma_.lower() in MODAL_WORDS)
    modal_word_ratio = modal_count / token_len if token_len else 0

    # --- Discourse markers
    text_lower = sentence.lower()
    discourse_count = sum(1 for marker in DISCOURSE_MARKERS if marker in text_lower)
    discourse_marker_ratio = discourse_count / token_len if token_len else 0

    # --- Clause complexity
    clause_deps = {"relcl", "advcl", "ccomp", "xcomp"}
    clause_complexity = sum(1 for token in tokens if token.dep_ in clause_deps) / token_len if token_len else 0

    # --- Pattern type and flags
    pattern_type = get_pattern_type(sentence)
    pattern_flags = {
        "pattern_type": pattern_type,
        "flag_pattern_1": int(pattern_type == 1),
        "flag_pattern_2": int(pattern_type == 2),
        "flag_pattern_3": int(pattern_type == 3),
        "flag_pattern_4": int(pattern_type == 4),
        "flag_special_meta": int(pattern_type > 0)
    }

    # --- Pairwise comparison with next sentence (flat logic)
    if index is not None and all_sentences is not None:
        prev_type = get_pattern_type(all_sentences[index - 1]) if index > 0 else 0
        next_type = get_pattern_type(all_sentences[index + 1]) if index < len(all_sentences) - 1 else 0

        pattern_flags.update({
            "pattern_same_block": int(pattern_type != 0 and pattern_type == next_type),
            "pattern_boundary_cross": int(pattern_type != next_type),
            "pattern_block_start": int(pattern_type != 0 and pattern_type != prev_type),
            "pattern_block_end": int(pattern_type != 0 and pattern_type != next_type)
        })


    # --- Emoji & formatting
    has_emoji = any(char in emoji.EMOJI_DATA for char in sentence)
    has_non_ascii = any(ord(c) > 127 for c in sentence)
    has_quoted_text = bool(re.search(r"[\"“”‘’].+?[\"“”‘’]", sentence))

    # --- Meta-pragmatic
    blob = TextBlob(sentence)
    sentiment_polarity = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    rhetorical_question = int(sentence.strip().endswith("?") and sentence.strip().split()[0].lower() in {
        "what", "why", "how", "is", "are", "do", "can", "should"
    })
    all_caps_words = sum(1 for word in sentence.split() if word.isupper() and len(word) > 2)
    sarcasm_detected = any(marker in text_lower for marker in SARCASM_MARKERS)

    # --- LIWC-style
    words = text_lower.split()
    def count_match(word_set): return sum(1 for w in words if w in word_set)
    liwc_features = {
        "liwc_analytic_count": count_match(LIWC_ANALYTIC),
        "liwc_affective_count": count_match(LIWC_AFFECTIVE),
        "liwc_authenticity_count": count_match(LIWC_AUTHENTICITY),
        "liwc_certainty_count": count_match(LIWC_CERTAINTY),
        "liwc_tentativeness_count": count_match(LIWC_TENTATIVENESS),
    }

    return {
        "formality_score": formality_score,
        "ner_density": ner_density,
        "modal_word_ratio": modal_word_ratio,
        "discourse_marker_ratio": discourse_marker_ratio,
        "clause_complexity": clause_complexity,
        "has_emoji": int(has_emoji),
        "has_non_ascii": int(has_non_ascii),
        "has_quoted_text": int(has_quoted_text),
        "sentiment_polarity": sentiment_polarity,
        "subjectivity_score": subjectivity_score,
        "rhetorical_question": rhetorical_question,
        "has_all_caps": int(all_caps_words > 0),
        "sarcasm_marker": int(sarcasm_detected),
        **pattern_flags,
        **liwc_features
    }

def compute_wan_window_drift(sentences, window_size=3, step=1, include_pos=True):
    """
    Computes drift in WAN-based lexical-syntactic features across sliding windows of sentences.

    Args:
        sentences (list): List of sentences.
        window_size (int): Number of sentences in each window.
        step (int): Step size for moving the window.
        include_pos (bool): Whether to include POS-specific WAN features.

    Returns:
        list: List of dictionaries with cosine similarity and Euclidean distance for adjacent windows.
    """

    if len(sentences) < window_size + 1:
        return []

    window_features = []
    indices = []

    # Build WAN + extract features per window
    for start in range(0, len(sentences) - window_size + 1, step):
        window_text = sentences[start:start + window_size]
        joined = " ".join(window_text)
        preprocessed = preprocessing.preprocessing([joined], stopwords=False, lemmatizer=False, punctuations=False)
        wans = preprocessing.construct_wans(preprocessed, include_pos=include_pos)
        features = extract_features(wans, include_pos=include_pos)
        flat = extract_lexical_syntactic_features(features, include_pos=include_pos)
        window_features.append(flat[0])
        indices.append(start)

    # Compute pairwise drift between adjacent windows
    drifts = []
    for i in range(len(window_features) - 1):
        f1 = window_features[i]
        f2 = window_features[i + 1]

        shared_keys = [k for k in f1 if k in f2 and isinstance(f1[k], (float, int)) and isinstance(f2[k], (float, int))]
        vec1 = np.array([f1[k] for k in shared_keys])
        vec2 = np.array([f2[k] for k in shared_keys])

        cosine_sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) if np.linalg.norm(vec1) * np.linalg.norm(vec2) > 0 else 0.0
        euclidean = float(np.linalg.norm(vec1 - vec2))

        drifts.append({
            "wan_window_cosine_sim": cosine_sim,
            "wan_window_euclidean": euclidean,
            "wan_window_index": indices[i],
            "wan_window_size": window_size
        })

    return drifts

def extract_contextual_features(sentences):
    """
    Extracts contextual features for each sentence, including similarity to neighbors, NER and lemma overlaps,
    discourse cues, and subject continuity.

    Args:
        sentences (list): List of sentences.

    Returns:
        list: List of dictionaries containing contextual features per sentence.
    """

    features = []
    docs = list(nlp.pipe(sentences))

    for i, doc in enumerate(docs):
        sent_features = {}
        cur_vec = doc.vector

        # Sentence-to-sentence SBERT cosine
        if 0 < i < len(docs) - 1:
            prev_vec = docs[i - 1].vector
            next_vec = docs[i + 1].vector
            avg_cosine = (
                cosine_similarity([cur_vec], [prev_vec])[0][0] +
                cosine_similarity([cur_vec], [next_vec])[0][0]
            ) / 2
            sent_features["avg_sbert_cosine_neighbors"] = avg_cosine
        else:
            sent_features["avg_sbert_cosine_neighbors"] = 0.0

        # NER entity overlap with next
        if i < len(docs) - 1:
            ents1 = {ent.text.lower() for ent in doc.ents}
            ents2 = {ent.text.lower() for ent in docs[i + 1].ents}
            intersect = ents1 & ents2
            union = ents1 | ents2
            ner_overlap = len(intersect) / len(union) if union else 0.0
            sent_features["ner_overlap_next"] = ner_overlap
        else:
            sent_features["ner_overlap_next"] = 0.0

        # Lemma overlap with next
        if i < len(docs) - 1:
            lemmas1 = {token.lemma_ for token in doc if token.is_alpha}
            lemmas2 = {token.lemma_ for token in docs[i + 1] if token.is_alpha}
            intersect = lemmas1 & lemmas2
            union = lemmas1 | lemmas2
            lemma_overlap = len(intersect) / len(union) if union else 0.0
            sent_features["lemma_overlap_next"] = lemma_overlap
        else:
            sent_features["lemma_overlap_next"] = 0.0

        # Discourse cue starter
        first_token = doc[0].text.lower() if len(doc) > 0 else ""
        sent_features["starts_with_discourse"] = int(first_token in DISCOURSE_MARKERS)

        # Subject continuity with previous (heuristic)
        if i > 0:
            subj1 = [tok.text.lower() for tok in doc if tok.dep_ == "nsubj"]
            subj0 = [tok.text.lower() for tok in docs[i - 1] if tok.dep_ == "nsubj"]
            sent_features["subject_overlap_prev"] = int(bool(set(subj0) & set(subj1)))
        else:
            sent_features["subject_overlap_prev"] = 0

        features.append(sent_features)

    return features

def jaccard_similarity(set1, set2):
    """
    Computes Jaccard similarity between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard similarity score.
    """

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def cosine_similarity_safe(v1, v2):
    """
    Computes cosine similarity between two vectors with zero-vector check.

    Args:
        v1 (np.array): First vector.
        v2 (np.array): Second vector.

    Returns:
        float: Cosine similarity score.
    """

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return 1 - cosine(v1, v2)

def extract_wan_pairwise_features(wan1, wan2):
    """
    Computes similarity features between two WANs based on structure and centrality metrics.

    Args:
        wan1 (nx.DiGraph): First WAN graph.
        wan2 (nx.DiGraph): Second WAN graph.

    Returns:
        dict: Dictionary of similarity features.
    """
    features = {}

    # --- Jaccard similarities ---
    features["edge_jaccard"] = jaccard_similarity(set(wan1.edges()), set(wan2.edges()))
    features["node_jaccard"] = jaccard_similarity(set(wan1.nodes()), set(wan2.nodes()))

    # --- Degree Centrality Cosine Similarity ---
    deg1 = nx.degree_centrality(wan1)
    deg2 = nx.degree_centrality(wan2)
    all_nodes = list(set(deg1) | set(deg2))
    vec_deg1 = np.array([deg1.get(n, 0) for n in all_nodes])
    vec_deg2 = np.array([deg2.get(n, 0) for n in all_nodes])
    features["centrality_cosine"] = cosine_similarity_safe(vec_deg1, vec_deg2)

    # --- Closeness Centrality Cosine Similarity ---
    try:
        closeness1 = nx.closeness_centrality(wan1)
        closeness2 = nx.closeness_centrality(wan2)
        vec_close1 = np.array([closeness1.get(n, 0) for n in all_nodes])
        vec_close2 = np.array([closeness2.get(n, 0) for n in all_nodes])
        features["closeness_cosine"] = cosine_similarity_safe(vec_close1, vec_close2)
    except:
        features["closeness_cosine"] = 0.0

    # --- Eigenvector Centrality Cosine Similarity ---
    try:
        eig1 = nx.eigenvector_centrality(wan1, max_iter=5000)
        eig2 = nx.eigenvector_centrality(wan2, max_iter=5000)
        vec_eig1 = np.array([eig1.get(n, 0) for n in all_nodes])
        vec_eig2 = np.array([eig2.get(n, 0) for n in all_nodes])
        features["eigenvector_cosine"] = cosine_similarity_safe(vec_eig1, vec_eig2)
    except:
        features["eigenvector_cosine"] = 0.0

    return features


def degree_entropy(wan):
    """
    Computes entropy of the degree distribution of a WAN.

    Args:
        wan (nx.Graph): Word adjacency network.

    Returns:
        float: Degree entropy value.
    """
    degrees = [d for _, d in wan.degree()]
    if not degrees:
        return 0.0
    counts = np.bincount(degrees)
    probs = counts / counts.sum() if counts.sum() > 0 else [1]
    return entropy(probs)

def pos_transition_entropy(wan):
    """
    Computes entropy of POS tag transitions in a WAN.

    Args:
        wan (nx.Graph): Word adjacency network.

    Returns:
        float: Transition entropy score.
    """
    pos_edges = [(u, v) for u, v in wan.edges() if "POS_" in u and "POS_" in v]
    transitions = Counter(pos_edges)
    total = sum(transitions.values())
    probs = np.array([v / total for v in transitions.values()])
    return entropy(probs) if probs.size > 0 else 0.0

def count_triangles(wan):
    """
    Counts the number of triangles in the WAN.

    Args:
        wan (nx.Graph): Word adjacency network.

    Returns:
        int: Triangle count.
    """
    try:
        return sum(nx.triangles(wan.to_undirected()).values()) // 3
    except:
        return 0

def star_ratio(wan, min_degree=3):
    """
    Computes the ratio of nodes with degree greater than or equal to a threshold in a WAN.

    Args:
        wan (nx.Graph): Word adjacency network.
        min_degree (int): Degree threshold.

    Returns:
        float: Star structure ratio.
    """
    if wan.number_of_nodes() == 0:
        return 0.0
    high_deg = sum(1 for _, d in wan.degree() if d >= min_degree)
    return high_deg / wan.number_of_nodes()


def extract_features(wans, include_pos):
    """
    Extracts a variety of graph-based features from Word Adjacency Networks (WANs).

    For each WAN (one per sentence or scene), it computes:
    - Graph-level metrics: average degree, density, clustering coefficient, assortativity
    - Node-level metrics: degree centrality, in/out degree, betweenness, closeness, eigenvector centrality
    - POS-level metrics: statistics specific to nodes labeled as POS (e.g., "POS_NN")

    Params:
        wans (dict): A dictionary where keys are sentence or scene identifiers and values are
                     NetworkX directed graphs (DiGraph) representing Word Adjacency Networks.

    Returns:
        dict: A dictionary mapping each sentence/scene ID to its corresponding feature dictionary.
              Each feature dictionary contains scalar graph metrics and dictionaries of node-level
              or edge-level features.
    """
    features = {}

    for scene, wan in wans.items():
        if not isinstance(wan, nx.DiGraph):
            print(f"ERROR: `wans[{scene}]` is NOT a `DiGraph`. Skipping...")
            continue

        scene_features = {}

        # Graph-Level Metrics
        scene_features['average_degree'] = sum(dict(wan.degree()).values()) / wan.number_of_nodes() if wan.number_of_nodes() > 0 else 0
        scene_features['density'] = nx.density(wan)
        scene_features['average_clustering'] = nx.average_clustering(wan.to_undirected()) if wan.number_of_nodes() > 1 else 0
        scene_features['assortativity'] = nx.degree_assortativity_coefficient(wan) if wan.number_of_nodes() > 1 else 0

        # Node-Level Metrics
        scene_features['degree_centrality'] = nx.degree_centrality(wan)
        scene_features['in_degree'] = dict(wan.in_degree())
        scene_features['out_degree'] = dict(wan.out_degree())
        scene_features['betweenness_centrality'] = nx.betweenness_centrality(wan) if wan.number_of_nodes() > 1 else {}
        scene_features['closeness_centrality'] = nx.closeness_centrality(wan) if wan.number_of_nodes() > 1 else {}
        scene_features['eigenvector_centrality'] = nx.eigenvector_centrality(wan, max_iter=50000000) if wan.number_of_nodes() > 1 else {}

        # POS-level Features
        if include_pos:
            pos_nodes = [n for n, d in wan.nodes(data=True) if str(n).startswith("POS_")]
            pos_subgraph = wan.subgraph(pos_nodes).copy()

            scene_features['pos_node_count'] = len(pos_nodes)
            scene_features['pos_avg_degree'] = sum(dict(pos_subgraph.degree()).values()) / len(pos_nodes) if pos_nodes else 0
            scene_features['pos_density'] = nx.density(pos_subgraph) if len(pos_nodes) > 1 else 0

            pos_centrality = nx.degree_centrality(pos_subgraph) if pos_nodes else {}
            scene_features['pos_centrality'] = pos_centrality
            scene_features['pos_transition_entropy'] = pos_transition_entropy(wan)

        scene_features['degree_entropy'] = degree_entropy(wan)
        # scene_features['triangle_count'] = count_triangles(wan)
        # scene_features['triangle_density'] = scene_features[
        #                                          'triangle_count'] / wan.number_of_nodes() if wan.number_of_nodes() > 0 else 0
        # scene_features['star_ratio'] = star_ratio(wan)  # already implemented earlier


        features[scene] = scene_features

    return features


def extract_lexical_syntactic_features(features, include_pos):
    """
    Transforms raw graph features into scalar lexical-syntactic features for each sentence.
    Aggregates node-, edge-, and POS-level metrics (centralities, degrees, etc.).

    Params:
        features (dict): Raw features from extract_features().

    Returns:
        dict: Flattened, scalar-only feature dicts per sentence.
    """
    feature_vectors = {}

    for scene_id, scene_features in features.items():
        vector = {}

        # --- Graph-Level Features ---
        vector['average_degree'] = scene_features.get('average_degree', 0)
        vector['density'] = scene_features.get('density', 0)
        vector['average_clustering'] = scene_features.get('average_clustering', 0)
        vector['assortativity'] = scene_features.get('assortativity', 0)
        vector['degree_entropy'] = scene_features.get('degree_entropy', 0)

        # --- Node-Level Features ---
        for name in ['degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality']:
            values = list(scene_features.get(name, {}).values())
            if values:
                vector[f'{name}_mean'] = np.mean(values)
                vector[f'{name}_std'] = np.std(values)
                vector[f'{name}_max'] = np.max(values)
                vector[f'{name}_min'] = np.min(values)
            else:
                vector[f'{name}_mean'] = 0
                vector[f'{name}_std'] = 0
                vector[f'{name}_max'] = 0
                vector[f'{name}_min'] = 0

        # --- Degree-based Features ---
        for degree_type in ['in_degree', 'out_degree']:
            values = list(scene_features.get(degree_type, {}).values())
            if values:
                vector[f'{degree_type}_mean'] = np.mean(values)
                vector[f'{degree_type}_std'] = np.std(values)
            else:
                vector[f'{degree_type}_mean'] = 0
                vector[f'{degree_type}_std'] = 0

        # --- POS-Level Features ---
        if include_pos:
            vector['pos_node_count'] = scene_features.get('pos_node_count', 0)
            vector['pos_avg_degree'] = scene_features.get('pos_avg_degree', 0)
            vector['pos_density'] = scene_features.get('pos_density', 0)

            pos_values = list(scene_features.get('pos_centrality', {}).values())
            if pos_values:
                vector['pos_centrality_mean'] = np.mean(pos_values)
                vector['pos_centrality_std'] = np.std(pos_values)
            else:
                vector['pos_centrality_mean'] = 0
                vector['pos_centrality_std'] = 0

        feature_vectors[scene_id] = vector

    return feature_vectors


def extract_pos_ngrams(sentence, n=2, skip=0):
    """
    Extracts POS n-grams with optional skips from a sentence.

    Args:
        sentence (str): Input sentence.
        n (int): Length of n-grams.
        skip (int): Number of tokens to skip.

    Returns:
        Counter: POS n-gram counts.
    """
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc if not token.is_space]
    ngrams = []
    for i in range(len(pos_tags) - n + 1 - skip):
        ngram = [pos_tags[i + j*(skip+1)] for j in range(n)]
        ngrams.append('_'.join(ngram))
    return Counter(ngrams)



def yules_k(doc):
    """
    Computes Yule's K measure for lexical diversity.

    Args:
        doc (spacy.tokens.Doc): spaCy processed document.

    Returns:
        float: Yule's K value.
    """
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return 0
    freqs = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    N = sum(freqs.values())
    M1 = sum(v**2 for v in freqs.values())
    return 10_000 * (M1 - N) / (N**2) if N else 0

def dependency_avg_depth(sentence):
    """
    Computes the average depth of dependency parse trees in a sentence.

    Args:
        sentence (str): Input sentence.

    Returns:
        float: Average dependency depth.
    """
    doc = nlp(sentence)
    depths = []

    for token in doc:
        depth = 0
        current = token
        while current.head != current:
            current = current.head
            depth += 1
        depths.append(depth)

    return sum(depths) / len(depths) if depths else 0.0

def passive_ratio(sentence):
    """
    Computes the ratio of passive voice verbs in a sentence.

    Args:
        sentence (str): Input sentence.

    Returns:
        float: Passive voice ratio.
    """

    doc = nlp(sentence)
    passive_count = 0
    verb_count = 0

    for token in doc:
        if token.pos_ == "VERB":
            verb_count += 1
            if any(child.dep_ == "nsubjpass" for child in token.children):
                passive_count += 1

    return passive_count / verb_count if verb_count else 0.0


def compute_pos_entropy(sentence, ngram_range=(2, 3)):
    """
    Computes entropy of POS n-grams for a single sentence.

    Args:
        sentence (str): Input sentence.
        ngram_range (tuple): (min_n, max_n) n-gram range.

    Returns:
        float: Entropy score.
    """
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc]

    ngrams = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams += [" ".join(pos_tags[i:i+n]) for i in range(len(pos_tags)-n+1)]

    counter = Counter(ngrams)
    if not counter:
        return 0.0

    total = sum(counter.values())
    probs = np.array([v/total for v in counter.values()])
    return scipy_entropy(probs)


def extract_pos_tfidf_features(sentences):
    """
    Extract POS TF-IDF cosine similarity features for consecutive sentence pairs.

    Args:
        sentences (List[str]): List of sentences.

    Returns:
        List[float]: List of cosine similarities between POS TF-IDF vectors for sentence pairs.
    """
    pos_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        pos_tags = [token.pos_ for token in doc]
        pos_sequence = " ".join(pos_tags)
        pos_sentences.append(pos_sequence)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(pos_sentences)

    pos_tfidf_cosine = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1])[0][0]
        pos_tfidf_cosine.append(sim)

    return pos_tfidf_cosine

def compute_punctuation_burstiness(sentences, window_size=3):
    """
    Computes punctuation burstiness in local windows of sentences.

    Args:
        sentences (list): List of sentences.
        window_size (int): Window size for local analysis.

    Returns:
        list: Burstiness score for each sentence.
    """

    burstiness = []
    half_window = window_size // 2
    num_sentences = len(sentences)
    for i in range(num_sentences):
        start = max(0, i - half_window)
        end = min(num_sentences, i + half_window + 1)
        window_sentences = sentences[start:end]
        punctuations = []
        for sent in window_sentences:
            punctuations.extend(re.findall(r'[!?.]', sent))
        if len(punctuations) > 1:
            unique = len(set(punctuations))
            burst = unique / len(punctuations)
        else:
            burst = 0.0
        burstiness.append(burst)
    return burstiness

def compute_length_variance(sentences, window_size=3):
    """
    Computes variance in sentence lengths over a sliding window.

    Args:
        sentences (list): List of sentences.
        window_size (int): Number of sentences to include in each window.

    Returns:
        list: Variance values per sentence window.
    """

    variances = []
    half_window = window_size // 2
    num_sentences = len(sentences)
    for i in range(num_sentences):
        start = max(0, i - half_window)
        end = min(num_sentences, i + half_window + 1)
        lengths = [len(sent) for sent in sentences[start:end]]
        if len(lengths) > 1:
            var = np.var(lengths)
        else:
            var = 0.0
        variances.append(var)
    return variances


def extract_lexical_features(sentence):
    """
    Extracts lexical, syntactic, and stylometric features from a sentence.

    Args:
        sentence (str): Input sentence to analyze.

    Returns:
        dict: A dictionary containing features such as type-token ratio, average word length,
              character and token statistics, POS ratios, punctuation patterns, function word usage,
              pronoun perspective ratios, syntactic depth, passive voice ratio, POS n-gram counts,
              stopword and uppercase character ratios, and multiple readability metrics.
    """
    sentence = sentence.strip()
    doc = nlp(sentence)

    tokens = [token for token in doc if not token.is_space]
    token_len = len(tokens)
    char_len = len(sentence)

    words = [token.text for token in tokens]
    unique_words = set(token.text.lower() for token in tokens if token.is_alpha)

    features = {
        "ttr": len(unique_words) / token_len if token_len else 0,
        "yules_k": yules_k(doc),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "sentence_char_len": char_len,
        "sentence_token_len": token_len,
    }

    # Character ratios
    alpha = sum(c.isalpha() for c in sentence)
    digits = sum(c.isdigit() for c in sentence)
    features["char_alpha_ratio"] = alpha / char_len if char_len else 0
    features["char_per_token"] = char_len / token_len if token_len else 0

    features["repeated_punct"] = int(bool(re.search(r"(!{2,}|\?{2,}|\.{3,})", sentence)))
    features["markdown_emphasis"] = int("*" in sentence or "_" in sentence or "**" in sentence)
    features["all_caps_words"] = sum(1 for token in tokens if token.is_upper and len(token) > 1)

    # Function word ratio using spaCy POS tags
    function_word_count = sum(1 for token in tokens if token.pos_ in FUNCTION_POS_TAGS)
    features["function_word_ratio"] = function_word_count / token_len if token_len else 0

    # Specific function POS ratios
    pos_counts = {}
    for token in tokens:
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    features["pronoun_ratio"] = pos_counts.get("PRON", 0) / token_len if token_len else 0
    features["conjunction_ratio"] = pos_counts.get("CCONJ", 0) / token_len if token_len else 0
    features["preposition_ratio"] = pos_counts.get("ADP", 0) / token_len if token_len else 0

    # POS trigram counter (raw, for later similarity use)
    features["pos_bigram_counter"] = extract_pos_ngrams(sentence, n=2)
    features["pos_skipgram_counter"] = extract_pos_ngrams(sentence, n=2, skip=1)

    # --- Syntax-Level Features ---
    features["dependency_avg_depth"] = dependency_avg_depth(sentence)
    features["passive_ratio"] = passive_ratio(sentence)


    # Perspective-based pronoun features
    first_person_count = 0
    second_person_count = 0
    third_person_count = 0

    for token in tokens:
        if not token.is_alpha:
            continue
        word = token.text.lower()
        if word in FIRST_PERSON:
            first_person_count += 1
        elif word in SECOND_PERSON:
            second_person_count += 1
        elif word in THIRD_PERSON:
            third_person_count += 1

    features["first_person_ratio"] = first_person_count / token_len if token_len else 0
    features["second_person_ratio"] = second_person_count / token_len if token_len else 0
    features["third_person_ratio"] = third_person_count / token_len if token_len else 0

    # --- Stopword Ratio ---
    stopwords = set(STOP_WORDS)
    num_stopwords = sum(1 for token in tokens if token.text.lower() in stopwords)
    features["stopword_ratio"] = num_stopwords / token_len if token_len else 0

    # --- Uppercase Character Ratio ---
    num_upper = sum(1 for c in sentence if c.isupper())
    features["char_upper_ratio"] = num_upper / char_len if char_len else 0

    # --- Stylometric Readability/Rhythm Features ---
    try:
        features["flesch_reading_ease"] = textstat.flesch_reading_ease(sentence)
        features["gunning_fog"] = textstat.gunning_fog(sentence)
        features["automated_readability_index"] = textstat.automated_readability_index(sentence)
        features["syllable_count"] = textstat.syllable_count(sentence)
        features["dale_chall_score"] = textstat.dale_chall_readability_score(sentence)
    except Exception:
        features["flesch_reading_ease"] = 0.0
        features["gunning_fog"] = 0.0
        features["automated_readability_index"] = 0.0
        features["syllable_count"] = 0
        features["dale_chall_score"] = 0.0

    return features

