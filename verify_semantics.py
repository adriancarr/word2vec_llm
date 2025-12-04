import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import random

def verify_semantics(tsne_path, n_samples=10, n_neighbors=10):
    print(f"Loading t-SNE coordinates from {tsne_path}...")
    df = pd.read_csv(tsne_path)
    
    # Fit Nearest Neighbors on the t-SNE coordinates
    # Note: We are finding neighbors in the 2D t-SNE space as requested, 
    # though high-dimensional embedding space is usually better for semantics.
    X = df[['x', 'y']].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(X)
    
    # Pick random words
    indices = range(len(df))
    random_indices = random.sample(indices, n_samples)
    
    print(f"\nFinding {n_neighbors} nearest neighbors for {n_samples} random words (in t-SNE space):")
    print("-" * 80)
    
    for idx in random_indices:
        word = df.iloc[idx]['word']
        distances, neighbor_indices = nbrs.kneighbors([X[idx]])
        
        # Skip the first one because it's the word itself
        neighbor_words = df.iloc[neighbor_indices[0][1:]]['word'].tolist()
        
        print(f"Word: {word:<20} | Neighbors: {', '.join(neighbor_words)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify semantic clustering")
    parser.add_argument("--tsne-file", type=str, default="output/tsne_coordinates.csv", help="Path to t-SNE coordinates CSV")
    
    args = parser.parse_args()
    
    verify_semantics(args.tsne_file)
