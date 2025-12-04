import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import sys
import json
from tqdm import tqdm
import os

# Check for MPS device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    print(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Llama models often don't have a pad token, so we set it to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16, # Use float16 for efficiency
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

def get_word_embedding(word, tokenizer, model):
    """
    Generates an embedding for a word by max-pooling the last hidden state of its tokens.
    """
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get the last hidden state
    # Shape: (batch_size, sequence_length, hidden_size)
    last_hidden_state = outputs.hidden_states[-1]
    
    # Remove batch dimension
    # Shape: (sequence_length, hidden_size)
    token_embeddings = last_hidden_state[0]
    
    # Max pool across the sequence length dimension (tokens)
    # Shape: (hidden_size,)
    word_embedding, _ = torch.max(token_embeddings, dim=0)
    
    return word_embedding.cpu().numpy()

def process_bulk(input_file, output_dir, tokenizer, model):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading words from {input_file}...")
    with open(input_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    print(f"Found {len(words)} words.")
    
    embeddings = []
    processed_words = []
    
    print("Generating embeddings...")
    for word in tqdm(words):
        try:
            emb = get_word_embedding(word, tokenizer, model)
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
    parser.add_argument("--bulk", action="store_true", help="Process bulk words from file")
    parser.add_argument("--input-file", type=str, default="google-10000-english.txt", help="Input text file with words")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for embeddings")
    
    args = parser.parse_args()

    tokenizer, model = load_model()

    if args.test_word:
        print(f"Generating embedding for '{args.test_word}'...")
        embedding = get_word_embedding(args.test_word, tokenizer, model)
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")
    elif args.bulk:
        process_bulk(args.input_file, args.output_dir, tokenizer, model)
    else:
        print("No action specified. Use --test-word or --bulk.")

