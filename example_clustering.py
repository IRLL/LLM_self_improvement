"""
Author: Qianxi Li
Date: June 13, 2024
Description: This script performs clustering on BERT-encoded text data. It includes methods for
batch encoding, dimensionality reduction, clustering, and visualizing results. The script identifies
data points closest to cluster centers and saves them for further analysis.
"""

import torch
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import logging
from utils import log_method, load_bert, read_json, write_json, ClearCache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def batch_encode_strings(bert, tokenizer, strings, batch_size=16):
    """
    Encode a list of strings into vectors using a BERT model.

    Args:
        bert: BERT model for encoding.
        tokenizer: Tokenizer associated with the BERT model.
        strings: List of strings to encode.
        batch_size: Number of strings to process in each batch.

    Returns:
        Tensor containing encoded vectors.
    """
    model = bert
    model.eval()  # Set model to evaluation mode
    vectors = []  # Store vectors

    for i in tqdm(range(0, len(strings), batch_size), desc="Encoding Batches"):
        batch = strings[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = inputs['input_ids'].to("cuda:0")
        attention_mask = inputs['attention_mask'].to("cuda:0")

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # Use last hidden states

        batch_vectors = hidden_states.mean(dim=1)  # Mean pooling
        vectors.extend(batch_vectors)

    return torch.stack(vectors).cpu()

def apply_dim_reduction(vectors, n_components=2):
    """
    Reduce the dimensionality of vectors using PCA.

    Args:
        vectors: High-dimensional vectors to reduce.
        n_components: Target number of dimensions.

    Returns:
        Reduced-dimensionality vectors.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)

def apply_clustering(k, reduced_vectors):
    """
    Cluster vectors into k clusters using KMeans.

    Args:
        k: Number of clusters.
        reduced_vectors: Reduced-dimensionality vectors to cluster.

    Returns:
        Cluster centers.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_vectors)
    return kmeans.cluster_centers_

def find_most_centered_data(centers, reduced_vectors):
    """
    Identify data points closest to each cluster center.

    Args:
        centers: Cluster centers.
        reduced_vectors: Reduced-dimensionality vectors.

    Returns:
        List of indices of the closest data points.
    """
    closest_data = []
    for center in centers:
        distances = np.linalg.norm(reduced_vectors - center, axis=1)
        closest_data.append(np.argmin(distances))
    return closest_data

def find_center_examples(bert, tokenizer, task_json, k):
    """
    Process text data to identify examples closest to cluster centers.

    Args:
        bert: BERT model for encoding.
        tokenizer: Tokenizer associated with the BERT model.
        task_json: JSON object containing text data.
        k: Number of clusters.

    Returns:
        Reduced vectors, cluster centers, closest data indices, and examples closest to centers.
    """
    strings = task_json['Full clustering context']
    vectors = batch_encode_strings(bert, tokenizer, strings, batch_size=16)
    reduced_vectors = apply_dim_reduction(vectors, n_components=2)
    centers = apply_clustering(k, reduced_vectors)
    closest_data = find_most_centered_data(centers, reduced_vectors)

    examples = []
    for each in closest_data:
        question = task_json['Instances'][each]['input']
        answer = task_json['Instances'][each]['answer_prediction']
        reason = task_json['Instances'][each]['fb_pred']
        examples.append({"input": question, "output": answer, "reason": reason})

    return reduced_vectors, centers, closest_data, examples

def visualize_clustering(reduced_vectors, centers, closest_data):
    """
    Visualize clusters and their centers in 2D space.

    Args:
        reduced_vectors: Reduced-dimensionality vectors.
        centers: Cluster centers.
        closest_data: Indices of closest data points to centers.

    Saves:
        A plot visualizing the clustering.
    """
    plt.figure()
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5, label="Data Points")
    plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x', s=100, label="Centers")
    for index in closest_data:
        plt.scatter(
            reduced_vectors[index, 0], reduced_vectors[index, 1],
            color='green', marker='o', s=100, edgecolor='black', label="Closest to Center" if index == closest_data[0] else ""
        )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Clustering Visualization")
    plt.legend()
    plt.savefig("clustering_visualization.png")
    logging.info("Clustering visualization saved as clustering_visualization.png")

@log_method
def get_new_examples():
    """
    Generate new examples closest to cluster centers for feedback analysis.
    """
    with ClearCache():
        arguments = json.loads(sys.argv[1])
        experiment_root_path = arguments['experiment_root_path']
        fb_dataset_path = arguments['feedback_dataset_path']
        k = arguments['k']
        new_example_dict_path = arguments['prompt_example_dict_path']

        feedback_dataset = read_json(fb_dataset_path)
        bert, tokenizer = load_bert()

        all_new_examples = {}
        for key in feedback_dataset.keys():
            if key.endswith('.json'):
                reduced_vectors, centers, closest_data, examples = find_center_examples(bert, tokenizer, feedback_dataset[key], k)
                task_result_path = os.path.join(experiment_root_path, "prompt_example_clustering", key.split('.json')[0])
                os.makedirs(task_result_path, exist_ok=True)
                torch.save(reduced_vectors, os.path.join(task_result_path, "reduced_vectors.pt"))
                torch.save(centers, os.path.join(task_result_path, "centers.pt"))
                torch.save(closest_data, os.path.join(task_result_path, "closest_data.pt"))
                all_new_examples[key] = examples

        write_json(new_example_dict_path, all_new_examples)
        logging.info("New examples saved to %s", new_example_dict_path)
        del bert, tokenizer, all_new_examples, feedback_dataset

if __name__ == "__main__":
    get_new_examples()
