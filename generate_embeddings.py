import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import numpy as np
import argparse
import json
import logging
import os
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WordDataset(Dataset):
    """Dataset for iterating over a list of words."""
    def __init__(self, words: List[str]):
        self.words = words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx]

class WordEmbeddingGenerator:
    """
    A class to generate word embeddings using a pre-trained LLM.
    Supports batch processing and multiple pooling strategies.
    """
    
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self, device: Optional[str] = None, token_freqs_path: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            device: 'cpu', 'cuda', or 'mps'. If None, auto-detects.
            token_freqs_path: Path to CSV file containing token frequencies (required for weighted/rarest pooling).
        """
        self.device = self._get_device(device)
        self.tokenizer, self.model = self._load_model()
        self.token_freqs = self._load_token_freqs(token_freqs_path) if token_freqs_path else None

    def _get_device(self, device_name: Optional[str]) -> torch.device:
        if device_name:
            return torch.device(device_name)
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available.")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("CUDA is available.")
            return torch.device("cuda")
        else:
            logger.info("Using CPU.")
            return torch.device("cpu")

    def _load_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        logger.info(f"Loading model: {self.MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                dtype=torch.float16,
                device_map="auto" if self.device.type != "mps" else None
            )
            if self.device.type == "mps":
                model.to(self.device)
            
            model.eval()
            return tokenizer, model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_token_freqs(self, path: str) -> Dict[int, int]:
        logger.info(f"Loading token frequencies from {path}...")
        try:
            df = pd.read_csv(path)
            return dict(zip(df['token_id'], df['count']))
        except Exception as e:
            logger.error(f"Error loading token frequencies: {e}")
            raise

    def pool_embeddings(self, 
                       token_embeddings: torch.Tensor, 
                       input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor, 
                       pooling_method: str) -> torch.Tensor:
        """
        Pools token embeddings into word embeddings.
        
        Args:
            token_embeddings: (batch_size, seq_len, hidden_size)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            pooling_method: 'max', 'mean', 'weighted', 'rarest'
            
        Returns:
            (batch_size, hidden_size)
        """
        batch_size, seq_len, hidden_size = token_embeddings.shape
        
        if pooling_method == 'mean':
            # Mask padding tokens (set to 0)
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif pooling_method == 'max':
            # Set padded values to -inf so they don't affect max
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[~mask_expanded.bool()] = float('-inf')
            return torch.max(token_embeddings, dim=1)[0]

        elif pooling_method == 'weighted':
            if self.token_freqs is None:
                raise ValueError("Token frequencies required for weighted pooling")
            
            # Vectorized weight lookup is tricky with dict, so we loop or map
            # For simplicity and safety with the dict, we'll iterate
            # (Optimization: convert dict to tensor array if vocab is contiguous, but simple loop is fine for inference)
            weights = torch.zeros_like(input_ids, dtype=torch.float, device=self.device)
            
            # TODO: Optimize this loop
            input_ids_cpu = input_ids.cpu().numpy()
            for b in range(batch_size):
                for s in range(seq_len):
                    if attention_mask[b, s]:
                        tid = input_ids_cpu[b, s]
                        freq = self.token_freqs.get(tid, 1)
                        weights[b, s] = 1.0 / freq
            
            weights = weights.unsqueeze(-1) # (batch, seq, 1)
            
            # Normalize weights per sequence
            # Mask padding weights
            weights = weights * attention_mask.unsqueeze(-1)
            sum_weights = torch.sum(weights, dim=1, keepdim=True)
            norm_weights = weights / torch.clamp(sum_weights, min=1e-9)
            
            return torch.sum(token_embeddings * norm_weights, dim=1)

        elif pooling_method == 'rarest':
            if self.token_freqs is None:
                raise ValueError("Token frequencies required for rarest pooling")
                
            word_embeddings = []
            input_ids_cpu = input_ids.cpu().numpy()
            
            for b in range(batch_size):
                min_freq = float('inf')
                min_idx = 0
                valid_token_found = False
                
                for s in range(seq_len):
                    if attention_mask[b, s]:
                        tid = input_ids_cpu[b, s]
                        freq = self.token_freqs.get(tid, float('inf'))
                        if freq < min_freq:
                            min_freq = freq
                            min_idx = s
                            valid_token_found = True
                
                if valid_token_found:
                    word_embeddings.append(token_embeddings[b, min_idx])
                else:
                    # Should not happen for valid words, but fallback to mean or first
                    word_embeddings.append(token_embeddings[b, 0])
            
            return torch.stack(word_embeddings)

        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

    def generate_embeddings(self, 
                          words: List[str], 
                          pooling_method: str = 'mean', 
                          batch_size: int = 32,
                          layer: int = -1) -> Tuple[np.ndarray, List[str]]:
        """
        Generates embeddings for a list of words using batch processing.
        """
        dataset = WordDataset(words)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_embeddings = []
        processed_words = []
        
        logger.info(f"Generating embeddings for {len(words)} words using {pooling_method} pooling (batch_size={batch_size}, layer={layer})...")
        
        with torch.no_grad():
            for batch_words in tqdm(dataloader):
                # Tokenize batch
                # add_special_tokens=False to get pure word representation
                inputs = self.tokenizer(
                    batch_words, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128,
                    add_special_tokens=False
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                # Extract hidden states from the specified layer
                # outputs.hidden_states is a tuple of (num_layers + 1) tensors
                # Index -1 is the last layer, 0 is the embedding layer
                hidden_state = outputs.hidden_states[layer] 
                
                embeddings = self.pool_embeddings(
                    hidden_state, 
                    inputs['input_ids'], 
                    inputs['attention_mask'], 
                    pooling_method
                )
                
                all_embeddings.append(embeddings.cpu().numpy())
                processed_words.extend(batch_words)
                
        return np.concatenate(all_embeddings, axis=0), processed_words

    def generate_all_layers(self, 
                          words: List[str], 
                          pooling_method: str = 'mean', 
                          batch_size: int = 32,
                          save_layers_dir: Optional[str] = None) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """
        Generates embeddings for all layers at once.
        Returns a dict mapping layer_index -> embeddings array.
        If save_layers_dir is provided, saves each layer to a .npy file.
        """
        # Check if all files already exist
        num_layers = self.model.config.num_hidden_layers
        if save_layers_dir:
            os.makedirs(save_layers_dir, exist_ok=True)
            all_exist = True
            for i in range(num_layers + 1):
                if not os.path.exists(os.path.join(save_layers_dir, f"layer_{i:02d}.npy")):
                    all_exist = False
                    break
            
            if all_exist:
                logger.info(f"All layer embeddings already exist in {save_layers_dir}. Loading...")
                final_embeddings = {}
                processed_words = words # Assuming words match
                for i in range(num_layers + 1):
                    final_embeddings[i] = np.load(os.path.join(save_layers_dir, f"layer_{i:02d}.npy"))
                return final_embeddings, processed_words

        dataset = WordDataset(words)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize dict to store list of embeddings for each layer
        all_layers_embeddings = {i: [] for i in range(num_layers + 1)}
        processed_words = []
        
        logger.info(f"Generating embeddings for all {num_layers+1} layers for {len(words)} words...")
        
        with torch.no_grad():
            for batch_words in tqdm(dataloader):
                inputs = self.tokenizer(
                    batch_words, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128,
                    add_special_tokens=False
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Process each layer
                for layer_idx in range(num_layers + 1):
                    hidden_state = outputs.hidden_states[layer_idx]
                    
                    embeddings = self.pool_embeddings(
                        hidden_state, 
                        inputs['input_ids'], 
                        inputs['attention_mask'], 
                        pooling_method
                    )
                    
                    all_layers_embeddings[layer_idx].append(embeddings.cpu().numpy())
                
                processed_words.extend(batch_words)
        
        # Concatenate results and save if requested
        final_embeddings = {}
        for layer_idx, emb_list in all_layers_embeddings.items():
            emb_array = np.concatenate(emb_list, axis=0)
            final_embeddings[layer_idx] = emb_array
            
            if save_layers_dir:
                save_path = os.path.join(save_layers_dir, f"layer_{layer_idx:02d}.npy")
                np.save(save_path, emb_array)
            
        return final_embeddings, processed_words

def main():
    parser = argparse.ArgumentParser(description="Generate word embeddings using TinyLlama 1.1B")
    parser.add_argument("--test-word", type=str, help="Word to test embedding generation for")
    parser.add_argument("--bulk", action="store_true", help="Process a list of words from a file")
    parser.add_argument("--input-file", type=str, default="google-10000-english.txt", help="Path to input text file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--token-frequencies", type=str, help="Path to token frequencies CSV")
    parser.add_argument("--pooling", type=str, default="mean", choices=["max", "mean", "weighted", "rarest"], help="Pooling method")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--device", type=str, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to extract embeddings from (default: -1 for last layer)")
    parser.add_argument("--save-layers-dir", type=str, help="Directory to save embeddings for all layers")
    
    args = parser.parse_args()
    
    # Validate args
    if args.pooling in ['weighted', 'rarest'] and not args.token_frequencies:
        logger.error(f"--token-frequencies is required for {args.pooling} pooling.")
        return

    generator = WordEmbeddingGenerator(device=args.device, token_freqs_path=args.token_frequencies)

    if args.test_word:
        embeddings, _ = generator.generate_embeddings([args.test_word], args.pooling, batch_size=1)
        print(f"Embedding shape: {embeddings[0].shape}")
        print(f"First 10 values: {embeddings[0][:10]}")
        
    elif args.bulk:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file '{args.input_file}' not found.")
            return
            
        logger.info(f"Reading words from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
            
        if args.save_layers_dir:
            generator.generate_all_layers(words, args.pooling, args.batch_size, args.save_layers_dir)
            logger.info(f"Saved all layer embeddings to {args.save_layers_dir}")
            return

        embeddings, processed_words = generator.generate_embeddings(words, args.pooling, args.batch_size, args.layer)
        
        os.makedirs(args.output_dir, exist_ok=True)
        npy_path = os.path.join(args.output_dir, "embeddings.npy")
        json_path = os.path.join(args.output_dir, "words.json")
        
        np.save(npy_path, embeddings)
        with open(json_path, 'w') as f:
            json.dump(processed_words, f)
            
        logger.info(f"Saved {len(embeddings)} embeddings to {npy_path}")
        logger.info(f"Saved word list to {json_path}")
        
    else:
        # Interactive mode
        while True:
            word = input("Enter a word (or 'q' to quit): ")
            if word.lower() == 'q':
                break
            embeddings, _ = generator.generate_embeddings([word], args.pooling, batch_size=1)
            print(f"Embedding shape: {embeddings[0].shape}")
            print(f"First 10 values: {embeddings[0][:10]}")

if __name__ == "__main__":
    main()
