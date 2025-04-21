"""
The purpose of this file is to cluster embeddings using the k-means algorithm 
using user-defined representative embedding centroids.
"""
import os
import time

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from tqdm import tqdm

os.environ['OMP_NUM_THREADS'] = '1'

embedder = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def cluster_text_kmeans(texts: list[str], representative_texts: list[str] = None) -> list[list[str]]:
    """
    Cluster texts using the k-means algorithm with user-defined 
    representative texts. Best for low-dimensional embeddings.

    Returns a list of lists of text, where each sub list represents a cluster.
    """
    text_embeddings = embedder.encode(texts)

    if representative_texts is None:
        kmeans = KMeans()
        kmeans.fit(text_embeddings)
    else:
        representative_embeddings = embedder.encode(representative_texts)
        kmeans = KMeans(init=representative_embeddings, n_clusters=len(representative_texts))
        kmeans.fit(text_embeddings)

    output = [[] for _ in range(max(kmeans.labels_)+1)]

    for i in range(len(texts)):
        output[kmeans.labels_[i]].append(texts[i])

    return output

def cluster_text_optics(texts: list[str]) -> list[list[str]]:
    """
    Cluster texts using the OPTICS algorithm. Best for high-dimensional embeddings, but slow.

    Returns a list of lists of text, where each sub list represents a cluster.
    """

    text_embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    optics = OPTICS(min_samples=2, metric="cosine")
    labels = optics.fit_predict(tqdm(text_embeddings))

    output = [[] for _ in range(max(labels)+1)]

    for i in range(len(texts)):
        if labels[i] == -1:
            continue
        output[labels[i]].append(texts[i])

    return output

def cluster_text_fast(texts: list[str]) -> list[list[str]]:
    """
    Cluster texts using sentence_transformers.util.community_detection. Much faster than OPTICS and k-means.

    Returns a list of lists of text, where each sub list represents a cluster.
    """

    text_embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    clusters = util.community_detection(text_embeddings, min_community_size=2, show_progress_bar=True, threshold=0.75)
    output = [[] for _ in range(len(clusters))]

    for cluster_idx, cluster in enumerate(clusters):
        output[cluster_idx] = [texts[idx] for idx in cluster]
    
    return output


if __name__ == "__main__":
    import json
    from pprint import pprint

    with open("notes.json") as f_notes:
        notes: list[str] = [note["note"] for note in json.load(f_notes)]

    start = time.time()
    for cluster_idx, cluster in enumerate(cluster_text_fast(notes)):
        print(f"Cluster {cluster_idx+1}: ({len(cluster)} sentences)")
        pprint(cluster[:10])
        if len(cluster) > 10:
            print("...")
        print()
    print(f"Time: {time.time() - start}")
