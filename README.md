# 🎬 Movie Recommendation System (Content-Based Machine Learning)

A content-based movie recommendation system built using Machine Learning, a large local dataset (~10,000 movies), and the TMDB API for visual metadata (movie posters).

This project focuses on clean system design, scalability, reproducibility, and explainability, rather than over-engineering.

## 📌 Project Overview

Movie recommendation systems are widely used in real-world platforms such as Netflix, Prime Video, and IMDb.
The goal of this project is to design and implement a realistic recommendation pipeline that:

- Recommends movies based on content similarity

- Works entirely on a fixed local dataset

- Is fast, stable, and reproducible

- Uses external APIs only where necessary

- Is easy to understand and explain

This project was developed as part of a remote internship and emphasizes real-world engineering decisions.

## 🧠 Recommendation Approach
Content-Based Filtering

The system uses content-based filtering, where movies are recommended based on the similarity of their textual descriptions (movie overviews).

Why Content-Based?

- No user history required (cold-start friendly)

- Fully explainable and deterministic

- Easy to debug and improve

- Suitable for academic and internship projects

##  Dataset Details

- Size: ~10,000 movies

- Coverage: Worldwide movies (multiple languages & regions)

### Key Columns Used

- id	
- title	
- overview	
- vote_average	
- vote_count	

Data Cleaning:

- Removed movies with missing or empty overviews

- Ensured consistent indexing

- Retained only relevant columns

- Saved a cleaned dataset for reproducibility

## 🔬 Feature Engineering & Vectorization
Text Features

For each movie:

- title and overview are combined

- Text is lower-cased and cleaned

Vectorization

- TF-IDF (Term Frequency–Inverse Document Frequency) is used

- Each movie is represented as a 5000-dimensional vector

Why TF-IDF?

- Highlights important keywords

- Reduces the impact of common words

- Lightweight and efficient

- Well-suited for medium-scale datasets

📐 Similarity Computation

- Cosine Similarity measures similarity between movie vectors

- Similarity is computed on demand

- No full similarity matrix is stored (memory-efficient)

### Recommendation Logic

1. User selects a movie (from the dataset)

2. TF-IDF vector for the movie is retrieved

3. Cosine similarity is computed against all movies

4. The selected movie itself is excluded

5. Top-N most similar movies are returned



## 🖥️ Streamlit Web Application

An interactive Streamlit dashboard is provided for easy usage.

### UI Features

- Dataset-only movie selection dropdown

- Slider to control number of recommendations

- Loading spinner for better user experience

- Grid-based card layout

- Movie posters and ratings display

### Poster Handling

- Posters are fetched using TMDB movie IDs

- Ensures accurate and stable poster retrieval

- Poster responses are cached to avoid API rate limits

- Posters are displayed for top 5 recommendations only

## Why Dataset-Only Search?

Although TMDB supports live movie search, this project deliberately avoids using TMDB for recommendations.

Reasons:

- Reproducibility

- Stable and deterministic results

- Faster execution

- No dependency on external APIs for core logic

- Clear evaluation criteria

TMDB-based recommendation was explored separately during development and is kept as an experimental notebook, not part of the final pipeline.


## ⚠️ Limitations

- No collaborative filtering (no user behavior data)

- Recommendations rely only on textual similarity

- No genre-weighted or popularity-weighted ranking

- Poster availability depends on TMDB

These limitations are intentional to keep the system clean and explainable.

## 🚀 Future Improvements

- Genre-aware similarity scoring

- Popularity-adjusted ranking

- Hybrid recommendation (content + collaborative)

- Advanced NLP models (Word2Vec, BERT)

- Pagination for recommendations

- Cloud deployment (Streamlit Cloud / Docker)

## 🧠 Key Learnings

- System design matters more than model complexity

- External APIs should be used carefully

- Caching is critical in real-world applications

- Clear separation of ML logic and UI improves maintainability

- Explainability is essential in ML projects

###  How to Run the Project
pip install -r requirements.txt
streamlit run app.py


Create a .env file with:

TMDB_API_KEY=your_api_key_here

## 👤 Author

Aayush Maurya

B.Tech CSE (AI & ML)
