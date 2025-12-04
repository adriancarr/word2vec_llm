import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from generate_embeddings import WordEmbeddingGenerator

def load_categories():
    try:
        with open("output/categories.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: output/categories.json not found.")
        return {}

def calculate_distance_ratio(embeddings, words, category_words, sample_size=1000):
    """
    Calculates the ratio of average intra-group distance to average inter-group distance.
    Lower is better (means tight clusters separated from others).
    """
    word_to_idx = {w: i for i, w in enumerate(words)}
    valid_category_words = [w for w in category_words if w in word_to_idx]
    
    if len(valid_category_words) < 2:
        return float('nan')
        
    category_indices = [word_to_idx[w] for w in valid_category_words]
    
    # Intra-group distance
    cat_embeddings = embeddings[category_indices]
    intra_dists = pairwise_distances(cat_embeddings, metric='cosine')
    # We only care about upper triangle (excluding diagonal) to avoid double counting and self-distance
    intra_mean = intra_dists[np.triu_indices(len(cat_embeddings), k=1)].mean()
    
    # Inter-group distance
    # Sample non-category words to save time
    all_indices = set(range(len(words)))
    non_cat_indices = list(all_indices - set(category_indices))
    
    if len(non_cat_indices) > sample_size:
        non_cat_indices = np.random.choice(non_cat_indices, sample_size, replace=False)
        
    non_cat_embeddings = embeddings[non_cat_indices]
    
    inter_dists = pairwise_distances(cat_embeddings, non_cat_embeddings, metric='cosine')
    inter_mean = inter_dists.mean()
    
    if inter_mean == 0:
        return float('inf')
        
    return intra_mean / inter_mean

def calculate_category_density(embeddings, words, category_words, k=20):
    """
    Calculates the average percentage of neighbors that are in the same category.
    """
    word_to_idx = {w: i for i, w in enumerate(words)}
    # Filter category words to those present in the dataset
    valid_category_words = [w for w in category_words if w in word_to_idx]
    category_indices = [word_to_idx[w] for w in valid_category_words]
    
    if not category_indices:
        return 0.0
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings[category_indices])
    
    total_density = 0
    count = 0
    
    category_set = set(valid_category_words)
    
    for i, neighbors in enumerate(indices):
        # neighbors[0] is the word itself, so skip it
        neighbor_words = [words[idx] for idx in neighbors[1:]]
        matches = sum(1 for w in neighbor_words if w in category_set)
        density = matches / k
        total_density += density
        count += 1
        
    return total_density / count if count > 0 else 0.0

def scan_layers():
    # Load words
    input_file = "google-10000-english.txt"
    with open(input_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    
    # Load categories
    categories = load_categories()
    print(f"Loaded {len(categories)} categories.")
    for cat, items in categories.items():
        print(f"  - {cat}: {len(items)} words")
    
    # Initialize generator
    token_freqs_path = "output/token_frequencies.csv"
    generator = WordEmbeddingGenerator(token_freqs_path=token_freqs_path)
    
    num_layers = generator.model.config.num_hidden_layers
    print(f"Model has {num_layers} hidden layers.")
    
    results = []
    
    # Scan layers
    # Scan layers
    print(f"\nGenerating embeddings for all layers...")
    all_layers_data, _ = generator.generate_all_layers(
        words, 
        pooling_method='rarest', 
        batch_size=64,
        save_layers_dir="output/layers"
    )
    
    for layer_idx in range(num_layers + 1):
        print(f"Processing Layer {layer_idx}...")
        embeddings = all_layers_data[layer_idx]
        
        layer_result = {'layer': layer_idx}
        densities = []
        ratios = []
        
        for cat_name, cat_words in categories.items():
            # Density
            density = calculate_category_density(embeddings, words, cat_words)
            layer_result[f"{cat_name}_density"] = density
            densities.append(density)
            
            # Ratio
            ratio = calculate_distance_ratio(embeddings, words, cat_words)
            layer_result[f"{cat_name}_ratio"] = ratio
            if not np.isnan(ratio):
                ratios.append(ratio)
            
        avg_density = sum(densities) / len(densities) if densities else 0
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        
        layer_result['Average_Density'] = avg_density
        layer_result['Average_Ratio'] = avg_ratio
        
        print(f"  Avg Density: {avg_density:.4f}, Avg Ratio: {avg_ratio:.4f}")
        
        results.append(layer_result)
        
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("output/layer_analysis.csv", index=False)
    
    # Plot Density
    plt.figure(figsize=(12, 8))
    for cat_name in categories.keys():
        plt.plot(df['layer'], df[f"{cat_name}_density"], alpha=0.4, linewidth=1, label=cat_name)
    plt.plot(df['layer'], df['Average_Density'], color='black', linewidth=3, label='Average')
    plt.title("Semantic Clustering Quality (Neighborhood Density)")
    plt.xlabel("Layer Index")
    plt.ylabel("Density (Higher is Better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/layer_analysis_plot.png")
    
    # Plot Ratio
    plt.figure(figsize=(12, 8))
    for cat_name in categories.keys():
        plt.plot(df['layer'], df[f"{cat_name}_ratio"], alpha=0.4, linewidth=1, label=cat_name)
    plt.plot(df['layer'], df['Average_Ratio'], color='black', linewidth=3, label='Average')
    plt.title("Cluster Tightness (Intra/Inter Distance Ratio)")
    plt.xlabel("Layer Index")
    plt.ylabel("Ratio (Lower is Better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/layer_analysis_ratio_plot.png")
    
    print("\nSaved analysis to output/layer_analysis.csv, output/layer_analysis_plot.png, and output/layer_analysis_ratio_plot.png")
    
    # Find best layer
    best_density_layer = df.loc[df['Average_Density'].idxmax()]
    best_ratio_layer = df.loc[df['Average_Ratio'].idxmin()]
    
    print(f"\nBest Layer (Density): {int(best_density_layer['layer'])} with score {best_density_layer['Average_Density']:.4f}")
    print(f"Best Layer (Ratio):   {int(best_ratio_layer['layer'])} with score {best_ratio_layer['Average_Ratio']:.4f}")

if __name__ == "__main__":
    scan_layers()
