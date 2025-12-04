import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import os

def generate_fully_labeled_plot():
    input_file = "output/tsne_coordinates.csv"
    output_html = "output/tsne_all_labels_2d.html"
    output_png = "output/tsne_all_labels_2d.png"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading t-SNE coordinates from {input_file}...")
    df = pd.read_csv(input_file)
    
    # --- Interactive WebGL Plot (Hover Only) ---
    print(f"Generating interactive WebGL plot (hover only) for {len(df)} words...")
    # Use standard scatter for interactivity, but only show labels on hover to prevent browser crash
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        hover_name='word',
        title='2D t-SNE Visualization (Interactive - Hover for Labels)',
        opacity=0.6,
        render_mode='webgl'
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2"
    )
    
    print(f"Saving HTML plot to {output_html}...")
    fig.write_html(output_html)

    # --- Static Matplotlib Plot (All Labels) ---
    print(f"Generating static PNG plot (All Labels)...")
    plt.figure(figsize=(24, 24), dpi=150) # High resolution
    plt.scatter(df['x'], df['y'], alpha=0.3, s=5, c='blue')
    
    # Add labels
    for i, row in df.iterrows():
        plt.annotate(
            row['word'], 
            (row['x'], row['y']), 
            fontsize=6, 
            alpha=0.7,
            xytext=(0, 2), 
            textcoords='offset points',
            ha='center'
        )
        
    plt.title('2D t-SNE Visualization (All Words Labeled)')
    plt.axis('off')
    
    print(f"Saving PNG plot to {output_png}...")
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    generate_fully_labeled_plot()
