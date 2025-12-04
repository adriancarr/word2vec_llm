import requests
import json
import os
import nltk
from nltk.corpus import wordnet as wn

def get_all_animals_from_wordnet():
    print("Downloading WordNet...")
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        
    print("Extracting all animals from WordNet...")
    animals = set()
    animal_synset = wn.synset('animal.n.01')
    
    # Get all hyponyms recursively
    # closure(lambda s: s.hyponyms()) returns a generator of all hyponyms
    for synset in animal_synset.closure(lambda s: s.hyponyms()):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if '_' not in name and '-' not in name:
                animals.add(name)
                
    print(f"Found {len(animals)} unique single-word animals in WordNet.")
    return animals

def intersect_animals():
    # Get comprehensive list from WordNet
    external_animals = get_all_animals_from_wordnet()

    # Load dataset words
    words_path = "output/words.json"
    if not os.path.exists(words_path):
        print(f"Error: {words_path} not found.")
        return

    print(f"Loading dataset words from {words_path}...")
    with open(words_path, 'r') as f:
        dataset_words = set(json.load(f))

    # Filter and intersect
    intersected_animals = set()
    for animal in external_animals:
        # Check intersection
        if animal in dataset_words:
            intersected_animals.add(animal)

    # Sort for consistent output
    sorted_animals = sorted(list(intersected_animals))
    
    print(f"\nFound {len(sorted_animals)} animals in the dataset:")
    print(sorted_animals)
    
    # Save to file for easy reading/usage
    with open("output/intersected_animals.json", "w") as f:
        json.dump(sorted_animals, f, indent=2)
    print("\nSaved list to output/intersected_animals.json")

    # Sort for consistent output
    sorted_animals = sorted(list(intersected_animals))
    
    print(f"\nFound {len(sorted_animals)} animals in the dataset:")
    print(sorted_animals)
    
    # Save to file for easy reading/usage
    with open("output/intersected_animals.json", "w") as f:
        json.dump(sorted_animals, f, indent=2)
    print("\nSaved list to output/intersected_animals.json")

if __name__ == "__main__":
    intersect_animals()
