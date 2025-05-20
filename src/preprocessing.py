import os
import networkx as nx
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")

def process_sentence(sentence, stop_words=False, lemmatizer_instance=False, remove_punctuations=False):
    """
    Process a single sentence by applying common text preprocessing steps using spaCy.

    Params:
        sentence (str): The input sentence to be processed.
        stop_words (bool): Remove stopwords if True.
        lemmatizer_instance (bool): Apply lemmatization if True.
        remove_punctuations (bool): If True, removes all punctuations from the sentence.

    Returns:
        str: The processed sentence as a single string.
    """
    doc = nlp(sentence.casefold())

    tokens = []
    for token in doc:
        if token.is_space:
            continue
        if stop_words and token.is_stop:
            continue
        if remove_punctuations and token.is_punct:
            continue
        if lemmatizer_instance:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)

    return " ".join(tokens)


def preprocessing(text=None, stopwords=False, lemmatizer=False, punctuations=False):
    """
    Preprocess text data using spaCy.

    Params:
        text (list[str]): List of sentences.
        stopwords (bool): Remove stopwords if True.
        lemmatizer (bool): Apply lemmatization if True.
        punctuations (bool): Remove punctuations if True.

    Returns:
        list[str]: Preprocessed sentences.
    """
    if not text:
        raise ValueError("Input text cannot be None.")

    processed_text = [
        process_sentence(
            sentence=sentence,
            stop_words=stopwords,
            lemmatizer_instance=lemmatizer,
            remove_punctuations=punctuations
        )
        for sentence in text
    ]

    return processed_text

def construct_wans(sentences, output_dir=None, include_pos=False):
    """
    Constructs Word Adjacency Networks (WANs) from a list of sentences.
    Optionally saves WANs as pickle files if output_dir is provided.
    """

    # Only create directory if output_dir is explicitly given
    if output_dir is not None:
        output_dir = f"wans/{output_dir}/"
        os.makedirs(output_dir, exist_ok=True)

    wans = {}

    for i, sentence in enumerate(sentences):
        wan = nx.DiGraph()
        doc = nlp(sentence.lower())
        words = [token.text for token in doc if not token.is_space]

        for j in range(len(words) - 1):
            word1, word2 = words[j], words[j + 1]
            if wan.has_edge(word1, word2):
                wan[word1][word2]['weight'] += 1
            else:
                wan.add_edge(word1, word2, weight=1)

        if include_pos:
            pos_tags = [(token.text, token.tag_) for token in doc if not token.is_space]

            for word, pos in pos_tags:
                if word in wan.nodes:
                    wan.nodes[word]['pos'] = pos

            for j in range(len(pos_tags) - 1):
                pos1 = f"POS_{pos_tags[j][1]}"
                pos2 = f"POS_{pos_tags[j + 1][1]}"
                if not wan.has_node(pos1):
                    wan.add_node(pos1, type='POS')
                if not wan.has_node(pos2):
                    wan.add_node(pos2, type='POS')
                if wan.has_edge(pos1, pos2):
                    wan[pos1][pos2]['weight'] += 1
                else:
                    wan.add_edge(pos1, pos2, weight=1, edge_type='pos_pos')

        wan.remove_edges_from(nx.selfloop_edges(wan))
        wans[i] = wan

        # Only save if output_dir is valid
        if output_dir is not None:
            filepath = os.path.join(output_dir, f"wan_{i}.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(wan, f)

    return wans



