import numpy as np
import json
import pandas as pd
import umap
import matplotlib.pyplot as plt
import os
import argparse

def generate_umap(embeddings_path, words_path, output_dir):
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    
    # Check for tags file
    tags_path = os.path.join(output_dir, "word_tags.json")
    if os.path.exists(tags_path):
        print(f"Loading tags from {tags_path}...")
        with open(tags_path, 'r') as f:
            tags_data = json.load(f)
        words = [item['word'] for item in tags_data]
        tags = [item['simple_tag'] for item in tags_data]
    else:
        print(f"Loading words from {words_path}...")
        with open(words_path, 'r') as f:
            words = json.load(f)
        tags = ['Unknown'] * len(words)
        
    if len(embeddings) != len(words):
        print(f"Error: Number of embeddings ({len(embeddings)}) does not match number of words ({len(words)}).")
        return

    print(f"Running UMAP on {len(embeddings)} vectors of dimension {embeddings.shape[1]}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, verbose=True)
    umap_results = reducer.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'word': words,
        'tag': tags,
        'x': umap_results[:, 0],
        'y': umap_results[:, 1]
    })
    
    # Save coordinates
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "umap_coordinates.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved coordinates to {csv_path}")
    
    # Plot
    print("Generating plot...")
    plt.figure(figsize=(16, 10))
    
    # Color by tag
    unique_tags = df['tag'].unique()
    for tag in unique_tags:
        subset = df[df['tag'] == tag]
        plt.scatter(subset['x'], subset['y'], alpha=0.5, s=10, label=tag)
        
    plt.legend()
    
    # Annotate a few random words to make it interesting
    num_annotations = 50
    indices = np.random.choice(len(words), num_annotations, replace=False)
    for i in indices:
        plt.annotate(words[i], (df['x'][i], df['y'][i]), fontsize=8, alpha=0.7)
        
    plt.title("UMAP Visualization of Word Embeddings (Colored by POS)")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    
    plot_path = os.path.join(output_dir, "umap_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UMAP plot from embeddings")
    parser.add_argument("--embeddings", type=str, default="output/embeddings.npy", help="Path to embeddings .npy file")
    parser.add_argument("--words", type=str, default="output/words.json", help="Path to words .json file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    generate_umap(args.embeddings, args.words, args.output_dir)
