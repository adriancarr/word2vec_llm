import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import wordnet as wn
import os

def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        print("Downloading OMW-1.4...")
        nltk.download('omw-1.4')

import json

def load_animals_list():
    try:
        with open("output/intersected_animals.json", "r") as f:
            return set(json.load(f))
    except FileNotFoundError:
        print("Error: output/intersected_animals.json not found. Using empty list.")
        return set()

# Global variable to cache the list
ANIMALS = None

def is_animal(word):
    """
    Checks if a word is in the loaded animal list.
    """
    global ANIMALS
    if ANIMALS is None:
        ANIMALS = load_animals_list()
    return str(word).lower() in ANIMALS

def highlight_animals(tsne_path, output_path, is_3d=True):
    print(f"Loading t-SNE coordinates from {tsne_path}...")
    df = pd.read_csv(tsne_path)
    
    print("Identifying animals...")
    # Apply is_animal to each word
    df['is_animal'] = df['word'].apply(is_animal)
    
    num_animals = df['is_animal'].sum()
    print(f"Found {num_animals} animals out of {len(df)} words.")
    if num_animals > 0:
        print("Sample animals found:", df[df['is_animal']]['word'].head(20).tolist())
    
    # Create label column: show word only if it's an animal
    df['label'] = df.apply(lambda row: row['word'] if row['is_animal'] else '', axis=1)
    
    # Create category column for plotting
    df['Category'] = df['is_animal'].map({True: 'Animal', False: 'Other'})
    
    print(f"Generating {'3D' if is_3d else '2D'} plot...")
    
    if is_3d:
        fig = px.scatter_3d(
            df, 
            x='x', 
            y='y', 
            z='z',
            color='Category',
            text='label',
            hover_name='word',
            hover_data=['is_animal'],
            title='3D t-SNE Visualization (Animals Highlighted)',
            color_discrete_map={'Animal': 'red', 'Other': 'lightgray'},
            opacity=0.7
        )
    else:
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color='Category',
            text='label',
            hover_name='word',
            hover_data=['is_animal'],
            title='2D t-SNE Visualization (Animals Highlighted)',
            color_discrete_map={'Animal': 'red', 'Other': 'lightgray'},
            opacity=0.7
        )
    
    # Make non-animals smaller and more transparent to make animals pop
    fig.update_traces(marker=dict(size=3))
    
    # Configure text labels
    fig.update_traces(textposition='top center')
    if not is_3d:
        fig.update_traces(textfont=dict(size=10))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        legend_title_text='Category'
    )
    
    print(f"Saving plot to {output_path}...")
    fig.write_html(output_path)
    print("Done.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highlight animals in t-SNE plot")
    parser.add_argument("--2d", action="store_true", dest="is_2d", help="Generate 2D plot instead of 3D")
    parser.add_argument("--input", type=str, help="Input CSV file with t-SNE coordinates")
    parser.add_argument("--output", type=str, help="Output HTML file")
    
    args = parser.parse_args()
    
    download_nltk_data()
    
    is_3d = not args.is_2d
    
    if args.input:
        tsne_file = args.input
    else:
        tsne_file = "output/tsne_3d_coordinates.csv" if is_3d else "output/tsne_coordinates.csv"
        
    if args.output:
        output_file = args.output
    else:
        output_file = "output/tsne_animals_plot.html" if is_3d else "output/tsne_animals_plot_2d.html"
    
    if os.path.exists(tsne_file):
        highlight_animals(tsne_file, output_file, is_3d)
    else:
        print(f"Error: {tsne_file} not found. Please run generate_tsne_3d.py (for 3D) or generate_tsne.py (for 2D) first.")
