import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import argparse

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def analyze_clusters(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate the median x to split roughly in half, or use 0 if it's a clear split
    # Let's inspect the histogram of x first
    print("X coordinate stats:")
    print(df['x'].describe())
    
    # Assuming a split around 0 (common in t-SNE/UMAP if there are two distinct blobs)
    # But let's be more robust: use K-Means with k=2 to find the split
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['x']])
    
    group0 = df[df['cluster'] == 0]
    group1 = df[df['cluster'] == 1]
    
    print(f"\nGroup 0 size: {len(group0)}")
    print(f"Group 1 size: {len(group1)}")
    
    print(f"Group 0 X mean: {group0['x'].mean():.2f}")
    print(f"Group 1 X mean: {group1['x'].mean():.2f}")
    
    # Analyze properties
    def analyze_group(group, name):
        words = group['word'].astype(str).tolist()
        
        # 1. Capitalization
        capitalized = sum(1 for w in words if w[0].isupper())
        print(f"\n[{name}] Capitalized: {capitalized} ({capitalized/len(words)*100:.1f}%)")
        
        # 2. Length
        avg_len = np.mean([len(w) for w in words])
        print(f"[{name}] Avg Word Length: {avg_len:.2f}")
        
        # 3. First Letter
        first_letters = pd.Series([w[0].lower() for w in words]).value_counts().head(5)
        print(f"[{name}] Top 5 First Letters:\n{first_letters}")
        
        # 4. Tokenization
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        token_counts = [len(tokenizer(w, add_special_tokens=False)['input_ids']) for w in words]
        avg_tokens = np.mean(token_counts)
        print(f"[{name}] Avg Tokens: {avg_tokens:.2f}")
        
        # 5. Sample words
        print(f"[{name}] Sample words: {words[:10]}")

    print("\n--- Analyzing Group 0 ---")
    analyze_group(group0, "Group 0")
    
    print("\n--- Analyzing Group 1 ---")
    analyze_group(group1, "Group 1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze clusters in t-SNE/UMAP plot")
    parser.add_argument("--csv", type=str, default="output/tsne_coordinates.csv", help="Path to coordinates CSV")
    
    args = parser.parse_args()
    
    analyze_clusters(args.csv)
