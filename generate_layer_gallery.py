import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import imageio.v2 as imageio
from tqdm import tqdm
import argparse

def load_categories():
    try:
        with open("output/categories.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: output/categories.json not found.")
        return {}

def generate_layer_gallery():
    # Configuration
    layers_dir = "output/layers"
    gallery_dir = "output/gallery"
    tsne_dir = "output/tsne_layers"
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)
    
    # Load words
    with open("google-10000-english.txt", "r") as f:
        words = [line.strip() for line in f if line.strip()]
    
    # Load categories
    categories = load_categories()
    
    # Define colors for categories
    # Use a distinct colormap
    cmap = plt.get_cmap('tab10')
    category_colors = {cat: cmap(i) for i, cat in enumerate(categories.keys())}
    
    # Map words to categories for faster lookup
    word_to_category = {}
    for cat, cat_words in categories.items():
        for w in cat_words:
            word_to_category[w] = cat
            
    images = []
    
    # Iterate through layers
    # Detect number of layers
    layer_files = [f for f in os.listdir(layers_dir) if f.startswith("layer_") and f.endswith(".npy")]
    num_layers = len(layer_files)
    print(f"Found {num_layers} layers in {layers_dir}")
    
    for layer_idx in range(num_layers):
        print(f"\nProcessing Layer {layer_idx}...")
        
        tsne_path = os.path.join(tsne_dir, f"tsne_layer_{layer_idx:02d}.csv")
        
        # 1. Get t-SNE coordinates
        if os.path.exists(tsne_path):
            print(f"  Loading existing t-SNE coordinates from {tsne_path}...")
            df = pd.read_csv(tsne_path)
        else:
            npy_path = os.path.join(layers_dir, f"layer_{layer_idx:02d}.npy")
            if not os.path.exists(npy_path):
                print(f"  Error: {npy_path} not found. Please run generate_embeddings.py with --save-layers-dir first.")
                continue
                
            print(f"  Running t-SNE on {npy_path}...")
            embeddings = np.load(npy_path)
            
            # Run t-SNE
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
            coords = tsne.fit_transform(embeddings)
            
            df = pd.DataFrame(coords, columns=['x', 'y'])
            df['word'] = words
            df.to_csv(tsne_path, index=False)
            
        # 2. Generate Plot
        print(f"  Generating plot...")
        plt.figure(figsize=(12, 12), dpi=100)
        
        # Plot all points as background
        plt.scatter(df['x'], df['y'], c='lightgray', alpha=0.2, s=5, label='Other')
        
        # Plot each category
        for cat, color in category_colors.items():
            # Filter df for this category
            # We need to match words. 
            # Optimization: Add category column to df?
            # Or just filter using the set we loaded
            cat_words = set(categories[cat])
            mask = df['word'].isin(cat_words)
            subset = df[mask]
            
            plt.scatter(subset['x'], subset['y'], c=[color], label=cat, s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
            
            # Optional: Add labels for some points? Maybe too cluttered.
            
        plt.title(f"Layer {layer_idx} Semantic Clusters", fontsize=16)
        plt.axis('off')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.tight_layout()
        
        img_path = os.path.join(gallery_dir, f"layer_{layer_idx:02d}_plot.png")
        plt.savefig(img_path)
        plt.close()
        
        images.append(imageio.imread(img_path))
        
    # 3. Create GIF
    print(f"\nCreating animation...")
    gif_path = os.path.join(gallery_dir, "layers_animation.gif")
    imageio.mimsave(gif_path, images, duration=1.0) # 1 second per frame
    print(f"Saved animation to {gif_path}")

if __name__ == "__main__":
    generate_layer_gallery()
