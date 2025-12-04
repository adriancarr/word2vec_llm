import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import Counter
import os
import argparse
from tqdm import tqdm

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def analyze_token_frequency(words_path, output_dir):
    print(f"Loading words from {words_path}...")
    with open(words_path, 'r') as f:
        words = json.load(f)
        
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizing words and counting frequencies...")
    all_tokens = []
    for word in tqdm(words):
        # add_special_tokens=False to avoid counting BOS/EOS for every single word
        tokens = tokenizer(word, add_special_tokens=False)['input_ids']
        all_tokens.extend(tokens)
        
    token_counts = Counter(all_tokens)
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(token_counts)}")
    
    # Create DataFrame
    data = []
    for token_id, count in token_counts.items():
        token_str = tokenizer.decode([token_id])
        data.append({'token_id': token_id, 'token_str': token_str, 'count': count})
        
    df = pd.DataFrame(data)
    df = df.sort_values('count', ascending=False)
    
    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "token_frequencies.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved frequencies to {csv_path}")
    
    # Plot Histogram
    print("Generating histogram...")
    plt.figure(figsize=(12, 6))
    plt.hist(df['count'], bins=50, log=True, color='skyblue', edgecolor='black')
    plt.title("Token Frequency Distribution (Log Scale)")
    plt.xlabel("Frequency Count")
    plt.ylabel("Number of Unique Tokens (Log Scale)")
    plt.grid(axis='y', alpha=0.5)
    
    hist_path = os.path.join(output_dir, "token_frequency_histogram.png")
    plt.savefig(hist_path)
    print(f"Saved histogram to {hist_path}")
    
    # Print top 20 tokens
    print("\nTop 20 Most Frequent Tokens:")
    print(df.head(20)[['token_str', 'count']].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token frequencies")
    parser.add_argument("--words", type=str, default="output/words.json", help="Path to words .json file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    analyze_token_frequency(args.words, args.output_dir)
