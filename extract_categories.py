import nltk
from nltk.corpus import wordnet as wn
import json
import os

def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')

def get_hyponyms(synset_name):
    """
    Get all single-word hyponyms for a given synset name.
    """
    try:
        synset = wn.synset(synset_name)
    except Exception as e:
        print(f"Error loading synset {synset_name}: {e}")
        return set()
        
    hyponyms = set()
    # Use a stack for DFS to find all hyponyms
    stack = [synset]
    
    while stack:
        current = stack.pop()
        for lemma in current.lemmas():
            name = lemma.name().lower()
            if '_' not in name and '-' not in name: # Single words only
                hyponyms.add(name)
        
        stack.extend(current.hyponyms())
        stack.extend(current.instance_hyponyms())
        
    return hyponyms

def main():
    download_nltk_data()
    
    # Load dataset words
    input_file = "google-10000-english.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    with open(input_file, 'r') as f:
        dataset_words = set(line.strip() for line in f if line.strip())
    
    print(f"Loaded {len(dataset_words)} words from dataset.")
    
    # Define categories and their root synsets
    # Note: Synset choices are important.
    categories = {
        "Animals": "animal.n.01",
        "Plants": "plant.n.02", # plant.n.02 is 'living organism', plant.n.01 is 'industrial plant'
        "Countries": "country.n.02", # country.n.02 is 'territory'
        "Cities": "city.n.01",
        "Colors": "color.n.01",
        "Vehicles": "vehicle.n.01",
        "Food": "food.n.01", # food.n.01 is 'substance'
        "Sports": "sport.n.01",
        "Professions": "professional.n.01", # professional.n.01 is 'person'
        "Emotions": "emotion.n.01"
    }
    
    extracted_categories = {}
    
    for category, synset_name in categories.items():
        print(f"Extracting {category} ({synset_name})...")
        candidates = get_hyponyms(synset_name)
        
        # Intersect with dataset
        intersected = list(candidates.intersection(dataset_words))
        intersected.sort()
        
        extracted_categories[category] = intersected
        print(f"  Found {len(intersected)} words.")
        
        # Print a few examples
        print(f"  Examples: {intersected[:5]}")
        
    # Save to JSON
    output_file = "output/categories.json"
    os.makedirs("output", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(extracted_categories, f, indent=2)
        
    print(f"\nSaved categories to {output_file}")

if __name__ == "__main__":
    main()
