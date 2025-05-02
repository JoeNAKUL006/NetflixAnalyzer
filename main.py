import os
import pandas as pd

# Prepare a reduced dataset for Render deployment
def prepare_reduced_dataset():
    if not os.path.exists('netflix_titles_reduced.csv') and os.path.exists('netflix_titles.csv'):
        try:
            # Load the original dataset
            df = pd.read_csv('netflix_titles.csv')
            
            # Keep only essential columns
            essential_cols = ['title', 'type', 'director', 'cast', 'listed_in', 'description']
            reduced_df = df[essential_cols].copy()
            
            # Save the reduced dataset
            reduced_df.to_csv('netflix_titles_reduced.csv', index=False)
            print(f"Created reduced dataset: {len(reduced_df)} entries saved to netflix_titles_reduced.csv")
        except Exception as e:
            print(f"Error creating reduced dataset: {e}")

# Create reduced dataset when first run
prepare_reduced_dataset()

# Import Flask app
from app import app