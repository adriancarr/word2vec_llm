```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import sys
import json
from tqdm import tqdm
import os
import pandas as pd

# Check for MPS device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Metal Performance Shaders) is available.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("MPS not available. Using CPU.")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model(device):
    print(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Llama models often don't have a pad token, so we set it to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16, # Use float16 for efficiency
            device_map="auto" if device.type != "mps" else None # MPS sometimes has issues with device_map="auto"
        )
        if device.type == "mps":
            model.to(device)
            
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have access to the model and have logged in with 'huggingface-cli login'.")
        sys.exit(1)

def pool_embeddings(token_embeddings, pooling_method, input_ids=None, token_freqs=None, device=None):
    """
    Combines token embeddings into a single word embedding using the specified method.
    """
    if pooling_method == 'max':
        # Max pooling: take the max value across the token dimension
        word_embedding, _ = torch.max(token_embeddings, dim=0)
        return word_embedding
        
    elif pooling_method == 'mean':
        # Mean pooling: average across the token dimension
        word_embedding = torch.mean(token_embeddings, dim=0)
        return word_embedding
        
    elif pooling_method == 'weighted':
        if token_freqs is None or input_ids is None:
            raise ValueError("Weighted pooling requires token_freqs and input_ids")
            
        # Calculate weights: 1 / frequency
        weights = []
        for tid in input_ids:
            freq = token_freqs.get(tid, 1) # Default to 1 if not found (shouldn't happen if freq file is complete)
            weights.append(1.0 / freq)
        
        weights = torch.tensor(weights, device=device).unsqueeze(1) # Shape: (sequence_length, 1)
        
        # Normalize weights so they sum to 1
        weights = weights / weights.sum()
        
        # Weighted mean
        word_embedding = torch.sum(token_embeddings * weights, dim=0)
        return word_embedding
        
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

def get_word_embedding(word, tokenizer, model, pooling_method='mean', token_freqs=None):
    """
    Generates an embedding for a word using the specified pooling method.
    """
    # Tokenize the word
    # add_special_tokens=False because we want the embedding of the word itself, 
    # not surrounded by BOS/EOS tokens which might dominate the representation
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Get the input_ids to look up frequencies if needed
    input_ids = inputs['input_ids'][0].cpu().numpy()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
        # Get the last hidden state
        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Remove batch dimension
        # Shape: (sequence_length, hidden_size)
        token_embeddings = last_hidden_state[0]
        
        word_embedding = pool_embeddings(
            token_embeddings, 
            pooling_method, 
            input_ids=input_ids, 
            token_freqs=token_freqs, 
            device=device
        )
    
    return word_embedding.cpu().numpy()

def process_bulk(input_file, output_dir, tokenizer, model, pooling_method, token_freqs=None):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading words from {input_file}...")
    with open(input_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    print(f"Found {len(words)} words.")
    
    embeddings = []
    processed_words = []
    
    print(f"Generating embeddings using {pooling_method} pooling...")
    for word in tqdm(words):
        try:
            emb = get_word_embedding(word, tokenizer, model, pooling_method, token_freqs)
            embeddings.append(emb)
            processed_words.append(word)
        except Exception as e:
            print(f"Error processing word '{word}': {e}")

    embeddings_array = np.array(embeddings)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, "embeddings.npy")
    json_path = os.path.join(output_dir, "words.json")
    
    np.save(npy_path, embeddings_array)
    with open(json_path, 'w') as f:
        json.dump(processed_words, f)
        
    print(f"Saved {len(embeddings)} embeddings to {npy_path}")
    print(f"Saved word list to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word embeddings using TinyLlama 1.1B")
    parser.add_argument("--test-word", type=str, help="Word to test embedding generation for")
    parser.add_argument("--bulk", action="store_true", help="Process a list of words from a file")
    parser.add_argument("--input-file", type=str, default="google-10000-english.txt", help="Path to input text file with words")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--token-frequencies", type=str, help="Path to token frequencies CSV for weighted pooling")
    parser.add_argument("--pooling", type=str, default="mean", choices=["max", "mean", "weighted"], help="Pooling method to use")
    
    args = parser.parse_args()

    # Load token frequencies if provided or needed
    token_freqs = None
    if args.token_frequencies:
        print(f"Loading token frequencies from {args.token_frequencies}...")
        df = pd.read_csv(args.token_frequencies)
        # Create a dictionary mapping token_id to count
        token_freqs = dict(zip(df['token_id'], df['count']))
        
    if args.pooling == 'weighted' and token_freqs is None:
        print("Error: --token-frequencies is required for weighted pooling.")
        sys.exit(1)

    tokenizer, model = load_model(device)

    if args.test_word:
        print(f"Generating embedding for '{args.test_word}' using {args.pooling} pooling...")
        embedding = get_word_embedding(args.test_word, tokenizer, model, args.pooling, token_freqs)
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")
    elif args.bulk:
        process_bulk(args.input_file, args.output_dir, tokenizer, model, args.pooling, token_freqs)
    else:
        # Interactive mode
        while True:
            word = input("Enter a word (or 'q' to quit): ")
            if word.lower() == 'q':
                break
            
            embedding = get_word_embedding(word, tokenizer, model, args.pooling, token_freqs)
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 10 values: {embedding[:10]}")
```
