import numpy as np
import json
import pandas as pd
import umap
import plotly.express as px
import os
import argparse

def generate_umap_3d(embeddings_path, words_path, output_dir):
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

    print(f"Running 3D UMAP on {len(embeddings)} vectors of dimension {embeddings.shape[1]}...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, verbose=True)
    umap_results = reducer.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'word': words,
        'tag': tags,
        'x': umap_results[:, 0],
        'y': umap_results[:, 1],
        'z': umap_results[:, 2]
    })
    
    # Save coordinates
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "umap_3d_coordinates.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved coordinates to {csv_path}")
    
    # Plot
    print("Generating interactive 3D plot...")
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='tag',
        hover_name='word',
        title='3D UMAP Visualization of Word Embeddings (Colored by POS)',
        opacity=0.7,
        size_max=5
    )
    
    # Make markers smaller
    fig.update_traces(marker=dict(size=3))
    
    html_path = os.path.join(output_dir, "umap_3d_plot.html")
    fig.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D UMAP plot from embeddings")
    parser.add_argument("--embeddings", type=str, default="output/embeddings.npy", help="Path to embeddings .npy file")
    parser.add_argument("--words", type=str, default="output/words.json", help="Path to words .json file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    generate_umap_3d(args.embeddings, args.words, args.output_dir)
