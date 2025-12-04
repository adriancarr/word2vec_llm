import json
import nltk
import os
import argparse
from tqdm import tqdm

def tag_words(words_path, output_dir):
    print(f"Loading words from {words_path}...")
    with open(words_path, 'r') as f:
        words = json.load(f)
        
    print("Downloading NLTK data...")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')

    print(f"Tagging {len(words)} words...")
    
    # NLTK tags a list of words better than individual words (context matters, but here we have none)
    # However, for single words, it usually defaults to the most common usage.
    # We will tag them individually to ensure 1-to-1 mapping easily, 
    # but tagging the whole list at once might be faster/different.
    # Let's tag individually to be safe about alignment.
    
    tagged_words = []
    for word in tqdm(words):
        # pos_tag expects a list of words
        tag = nltk.pos_tag([word])[0][1]
        tagged_words.append({'word': word, 'tag': tag})
        
    # Simplify tags to basic categories for better plotting
    # N: Noun, V: Verb, J: Adjective, R: Adverb
    simplified_tags = []
    for item in tagged_words:
        tag = item['tag']
        if tag.startswith('N'):
            simple_tag = 'Noun'
        elif tag.startswith('V'):
            simple_tag = 'Verb'
        elif tag.startswith('J'):
            simple_tag = 'Adjective'
        elif tag.startswith('R'):
            simple_tag = 'Adverb'
        else:
            simple_tag = 'Other'
        item['simple_tag'] = simple_tag
        simplified_tags.append(item)
        
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "word_tags.json")
    
    with open(output_path, 'w') as f:
        json.dump(simplified_tags, f, indent=2)
        
    print(f"Saved tagged words to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag words with Part-of-Speech")
    parser.add_argument("--words", type=str, default="output/words.json", help="Path to words .json file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    tag_words(args.words, args.output_dir)
