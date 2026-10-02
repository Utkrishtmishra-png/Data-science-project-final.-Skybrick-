# 🎬 Content-Based Movie Recommendation System

**Data Science Internship – Month 3 Project**

A content-based recommender that suggests movies similar to one you already like, by comparing
their genres, keywords, cast, director and plot summaries. Comes with a Streamlit web UI.

## Features
- **Find similar movies** – pick a title, get the top-N most similar films with a match score
- **Genre filter** – narrow recommendations to a single genre
- **Describe what you want** – free-text search ("astronaut survival on a lonely planet")
- **Bring your own data** – upload any CSV in the same format from the sidebar
- **TMDB 5000 support** – converter script included for the popular Kaggle dataset
- Unit tests with `pytest`

## How it works
1. **Load & clean** – read movie metadata, drop duplicates, fill missing values.
2. **Feature "soup"** – merge genres (×2 weight), keywords, top-3 cast, director (×2 weight) and overview into one text field. Multi-word names like `Ryan Gosling` are collapsed to a single token so they aren't split.
3. **Vectorise** – `TfidfVectorizer` (English stop-words, unigrams + bigrams) turns each movie into a sparse vector; TF-IDF down-weights terms common to every film.
4. **Similarity** – cosine similarity (`linear_kernel` on L2-normalised TF-IDF vectors) between the chosen movie and all others.
5. **Rank** – sort by score, exclude the movie itself, optionally filter by genre, return top N.

## Project structure
```
movie-recommender/
├── app.py                  # Streamlit UI
├── src/
│   ├── recommender.py      # core model (TF-IDF + cosine similarity)
│   └── prepare_tmdb.py     # convert Kaggle TMDB 5000 -> project CSV
├── data/movies_sample.csv  # 51-movie demo dataset
├── tests/test_recommender.py
├── requirements.txt
└── README.md
```

## Setup & run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # opens http://localhost:8501
pytest                           # run tests
```

### Use as a library
```python
from src.recommender import MovieRecommender

rec = MovieRecommender.from_csv()
for r in rec.recommend("Inception", n=5):
    print(f"{r.title} ({r.year}) - {r.score:.0%}")
```

## Using a bigger dataset (TMDB 5000)
1. Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from Kaggle ("TMDB 5000 Movie Dataset").
2. Convert:
   ```bash
   python -m src.prepare_tmdb tmdb_5000_movies.csv tmdb_5000_credits.csv data/movies_tmdb.csv
   ```
3. Upload `data/movies_tmdb.csv` in the app sidebar (or change `DEFAULT_DATA` in `src/recommender.py`).

### CSV format
| column | required | example |
|---|---|---|
| title | ✅ | `Inception` |
| genres | ✅ | `Action\|Sci-Fi\|Thriller` |
| overview | ✅ | plot summary text |
| year, keywords, cast, director | optional | list values separated by `\|` |

## Sample output
```
Because you liked "The Godfather":
1. The Godfather Part II  39%
2. Goodfellas             15%
3. Pulp Fiction           14%
4. The Departed           13%
```

## Limitations & ideas
- Content-based only: it can't learn from user ratings or tastes of other users.
- Sample dataset is small, so match scores are low in absolute terms; use TMDB 5000 for richer results.
- Ideas: hybrid model with collaborative filtering, sentence-embedding (BERT) similarity, poster images via the TMDB API, deploy to Streamlit Community Cloud.

## Submission (GitHub)
```bash
git init && git add . && git commit -m "Month 3: content-based movie recommender"
git branch -M main
git remote add origin https://github.com/<your-username>/movie-recommender.git
git push -u origin main
```
Then paste the repository link in the Classroom submission.
