import json
import os

def filter_animals():
    input_path = "output/intersected_animals.json"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading animals from {input_path}...")
    with open(input_path, 'r') as f:
        animals = json.load(f)

    # Blocklist of ambiguous words or non-animals that slipped through
    BLOCKLIST = {
        'adult', 'ai', 'amazon', 'arab', 'asp', 'ass', 'baby', 'bass', 'bay', 
        'billy', 'bitch', 'blue', 'buck', 'bug', 'bull', 'bunny', 'cayman', 
        'char', 'charger', 'chat', 'chick', 'cisco', 'cock', 'cod', 'copper', 
        'coral', 'dam', 'das', 'devon', 'doe', 'dragon', 'drill', 'drum', 
        'durham', 'emperor', 'entire', 'female', 'fisher', 'fly', 'game', 
        'giant', 'grade', 'gray', 'grey', 'guinea', 'hack', 'hampshire', 
        'hart', 'head', 'hobby', 'human', 'humanity', 'humans', 'jack', 
        'jade', 'jay', 'jenny', 'jersey', 'kid', 'killer', 'kit', 'kitty', 
        'layer', 'lincoln', 'livestock', 'lucy', 'male', 'man', 'martin', 
        'mate', 'miller', 'monitor', 'monster', 'morgan', 'mount', 'newfoundland', 
        'pen', 'permit', 'pest', 'pet', 'pike', 'plug', 'pointer', 'poll', 
        'poster', 'poultry', 'pussy', 'queen', 'rail', 'ram', 'ray', 'redhead', 
        'robin', 'roller', 'royal', 'runner', 'sierra', 'soldier', 'sole', 
        'springer', 'stock', 'stud', 'survivor', 'swift', 'tit', 'tom', 'toy', 
        'welsh', 'worker', 'world', 'young'
    }

    filtered_animals = [word for word in animals if word not in BLOCKLIST]
    
    print(f"Original count: {len(animals)}")
    print(f"Filtered count: {len(filtered_animals)}")
    print(f"Removed {len(animals) - len(filtered_animals)} words.")
    
    print("\nRemaining animals:")
    print(filtered_animals)

    with open(input_path, 'w') as f:
        json.dump(filtered_animals, f, indent=2)
    print(f"\nSaved filtered list to {input_path}")

if __name__ == "__main__":
    filter_animals()
