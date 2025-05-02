# Netflix Content Recommendation System

A Flask-based recommendation system that suggests similar Netflix shows and movies based on user input. The application uses content-based filtering with cosine similarity to find and recommend content that is similar to what the user has enjoyed.

## Features

- **Content-Based Recommendations**: Get personalized Netflix content suggestions based on title, director, cast, genre, and description
- **User-Friendly Interface**: Clean, responsive design with Netflix-inspired styling
- **Error Handling**: Helpful error messages and suggestions when titles aren't found
- **Search Suggestions**: Example searches to help users get started
- **Mobile-Responsive**: Works on various screen sizes and devices

## Screenshots

- Main Search Page
- Recommendation Results Page 
- Error Handling Page

## Dataset

The application uses the Netflix dataset containing over 6,000 movies and TV shows available on Netflix, including:
- Title information
- Director and cast details
- Genre categories
- Release information
- Content descriptions

## Technology Stack

- **Backend**: Flask (Python)
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Styling**: Bootstrap with Netflix-inspired dark theme
- **Deployment**: Gunicorn

## Installation and Setup

### Prerequisites
- Python 3.9+
- pip

### Dependencies
```
flask==2.3.3
flask-sqlalchemy==3.0.5
gunicorn==23.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
```

### Local Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/netflix-recommendation-system.git
cd netflix-recommendation-system
```

2. Install dependencies:
```
pip install flask pandas numpy scikit-learn gunicorn flask-sqlalchemy
```

3. Make sure the Netflix dataset (netflix_titles.csv) is in the root directory

4. Run the application:
```
python app.py
```

5. Open your browser and navigate to http://localhost:5000

## How It Works

1. The system uses content-based filtering techniques
2. Text features (title, director, cast, listed_in, description) are combined into a 'soup'
3. CountVectorizer transforms the text data into numerical vectors 
4. Cosine similarity measures how similar items are to each other
5. When a user searches for a title, the system finds the most similar content

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Netflix dataset used for educational purposes
- Flask documentation and community
- scikit-learn documentation for content-based filtering techniques