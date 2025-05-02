import pandas as pd
import numpy as np
import re
import os
import logging
import gc
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)  # Reducing log level to save memory
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Memory optimization - load data only when necessary
netflix_overall = None
netflix_data = None
indices = None
count_vectorizer = None
cosine_sim = None

def load_and_process_data():
    global netflix_overall, netflix_data, indices, count_vectorizer, cosine_sim
    
    # Only load if not already loaded
    if netflix_data is None:
        try:
            # Load Netflix data (only once) - try to use reduced dataset if available
            logger.info("Loading dataset...")
            
            # Try to load the reduced dataset first to save memory
            if os.path.exists('netflix_titles_reduced.csv'):
                netflix_overall = pd.read_csv('netflix_titles_reduced.csv')
                netflix_data = netflix_overall.copy()
                logger.info("Using reduced dataset to save memory")
            else:
                netflix_overall = pd.read_csv('netflix_titles.csv')
                # Keep only essential columns to reduce memory
                essential_columns = ['title', 'type', 'director', 'cast', 'listed_in', 'description']
                netflix_data = netflix_overall[essential_columns].copy()
            
            # Fill NaN values with empty strings more efficiently
            netflix_data = netflix_data.fillna('')
            
            logger.info(f"Loaded dataset with {len(netflix_data)} entries")
            
            # Process data for recommendations
            process_data_for_recommendations()
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

def process_data_for_recommendations():
    global netflix_data, indices, count_vectorizer, cosine_sim
    
    try:
        # Function to clean data by removing spaces and converting to lowercase
        def clean_data(x):
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            return ""
        
        # Function to create a 'soup' of features for content-based filtering
        def create_soup(row):
            return (row['title'] + ' ' + row['director'] + ' ' + 
                    row['cast'] + ' ' + row['listed_in'] + ' ' + row['description'])
        
        logger.info("Processing data for recommendations...")
        
        # Clean each feature (more memory efficient)
        for feature in netflix_data.columns:
            netflix_data[feature] = netflix_data[feature].apply(clean_data)
        
        # Create a soup of all features for each content
        netflix_data['soup'] = netflix_data.apply(create_soup, axis=1)
        
        # Use TF-IDF instead of CountVectorizer for better memory efficiency
        count_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        count_matrix = count_vectorizer.fit_transform(netflix_data['soup'])
        
        # Release memory
        netflix_data.drop('soup', axis=1, inplace=True)
        gc.collect()
        
        logger.info("Creating similarity matrix...")
        # Generate similarity matrix in smaller batches to save memory
        cosine_sim = cosine_similarity(count_matrix, count_matrix, dense_output=False)
        
        # Clear more memory
        del count_matrix
        gc.collect()
        
        # Reset index and create indices series for faster lookups
        netflix_data.reset_index(inplace=True)
        indices = pd.Series(netflix_data.index, index=netflix_data['title'])
        
        logger.info("Successfully preprocessed data and created similarity matrix")
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

# Load data on startup
load_and_process_data()

# Helper function to clean input data
def clean_data(x):
    if isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ""

# Function to recommend similar content
def get_recommendations(title, cosine_sim_matrix):
    try:
        # Clean the input title
        clean_title = clean_data(title)
        
        # Check if the title exists in our dataset
        if clean_title not in indices:
            # Try to find the closest match
            all_titles = list(indices.index)
            matches = [t for t in all_titles if clean_title in t]
            
            if matches:
                # Use the first match
                clean_title = matches[0]
                logger.info(f"Used closest match: {clean_title}")
            else:
                logger.warning(f"No matches found for title: {title}")
                return pd.DataFrame()
        
        # Get the index of the title
        idx = indices[clean_title]
        
        # Get similarity scores for all content (using sparse matrix for memory efficiency)
        sim_scores = list(enumerate(cosine_sim_matrix[idx].toarray()[0]))
        
        # Sort content based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 10 similar content (excluding the input content itself)
        sim_scores = sim_scores[1:11]
        
        # Get content indices
        content_indices = [i[0] for i in sim_scores]
        
        # Return the top 10 most similar content with additional info
        result = netflix_overall.iloc[content_indices][['title', 'type', 'director', 'cast', 'listed_in']]
        return result
        
    except Exception as e:
        logger.error(f"Error in recommendation function: {str(e)}")
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        # Get movie name from form
        title = request.form.get('moviename', '')
        
        if not title:
            return render_template('error.html', error="Please enter a movie or TV show title")
        
        # Get recommendations
        recommendations = get_recommendations(title, cosine_sim)
        
        if recommendations.empty:
            return render_template('error.html', 
                                   error=f"No recommendations found for '{title}'. Please check the title and try again.")
        
        # Convert to HTML for display
        recommendations_html = recommendations.to_html(classes='table table-striped table-dark table-hover')
        
        return render_template('results.html', 
                               query=title, 
                               recommendations=recommendations_html)
    
    except Exception as e:
        logger.error(f"Error processing recommendation: {str(e)}")
        return render_template('error.html', 
                               error=f"An error occurred: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
