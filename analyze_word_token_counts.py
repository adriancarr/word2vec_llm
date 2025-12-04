import json
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def analyze_word_token_counts(words_path):
    print(f"Loading words from {words_path}...")
    with open(words_path, 'r') as f:
        words = json.load(f)
        
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizing words...")
    word_token_counts = []
    for word in tqdm(words):
        # add_special_tokens=False to count actual tokens for the word
        tokens = tokenizer(word, add_special_tokens=False)['input_ids']
        word_token_counts.append((word, len(tokens), tokens))
        
    # Sort by token count descending
    word_token_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Words by Token Count:")
    print(f"{'Word':<30} | {'Count':<5} | {'Tokens'}")
    print("-" * 60)
    for word, count, tokens in word_token_counts[:10]:
        token_strs = [tokenizer.decode([t]) for t in tokens]
        print(f"{word:<30} | {count:<5} | {token_strs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find words with most tokens")
    parser.add_argument("--words", type=str, default="output/words.json", help="Path to words .json file")
    
    args = parser.parse_args()
    
    analyze_word_token_counts(args.words)
