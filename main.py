"""
The purpose of this file is to cluster embeddings using the k-means algorithm 
using user-defined representative embedding centroids.
"""
import os

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

os.environ['OMP_NUM_THREADS'] = '1'

embedder = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def cluster_text(texts: list[str], representative_texts: list[str]) -> list[list[str]]:
    """
    Cluster texts using the k-means algorithm with user-defined 
    representative texts.

    Returns a list of lists of text, where each sub list represents a cluster.
    """
    text_embeddings = embedder.encode(texts)
    representative_embeddings = embedder.encode(representative_texts)

    kmeans = KMeans(n_clusters=len(representative_texts), init=representative_embeddings)
    kmeans.fit(text_embeddings)

    output = [[] for _ in range(len(representative_texts))]

    for i in range(len(texts)):
        output[kmeans.labels_[i]].append(texts[i])

    return output


if __name__ == "__main__":
    import json
    from pprint import pprint

    with open("notes.json") as f_notes:
        notes: list[str] = [note["note"] for note in json.load(f_notes)]
    
    representatives = [
        "Patient is experiencing cough.",
        "Patient is not experiencing cough."
    ]

    for cluster_idx, cluster in enumerate(cluster_text(notes, representatives)):
        print(f"Cluster {cluster_idx+1}: {representatives[cluster_idx]!r}")
        pprint(cluster)
        print()
