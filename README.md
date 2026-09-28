# Movie Recommendation System

A small, beginner-friendly recommender built with Python, pandas and scikit-learn. It implements three approaches:

| Approach | Idea | Function |
|---|---|---|
| **Content-based filtering** | Recommend movies with similar genres (TF-IDF + cosine similarity) | `ContentBased` |
| **Collaborative filtering** | Item-based CF on mean-centered user ratings | `ItemBasedCF` |
| **Hybrid** | Weighted blend of the two | `Hybrid` |

**Live demo:** _add your website link here_

## Quick start

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
python generate_sample_data.py      # only needed if data/ is empty

python main.py similar "The Matrix"                      # content-based
python main.py user 7                                    # collaborative, for user 7
python main.py hybrid "Toy Story=5" "Titanic=2" "Shrek=4"  # your own ratings
python main.py evaluate                                  # RMSE check
```

Example output for `python main.py similar "The Matrix"`:

```
             title  year                  genres  score
 Avengers: Endgame  2019 Action|Adventure|Sci-Fi  0.874
          Iron Man  2008 Action|Adventure|Sci-Fi  0.874
Mad Max: Fury Road  2015 Action|Adventure|Sci-Fi  0.874
     Jurassic Park  1993 Action|Adventure|Sci-Fi  0.874
         Inception  2010  Action|Sci-Fi|Thriller  0.798
```

## Data

`data/movies.csv` has 40 real movies with genres. `data/ratings.csv` holds **simulated** ratings from 60 fake users (each prefers two genres), created by `generate_sample_data.py` with a fixed random seed. Results on this data only show that the code works.

To use real data, download [MovieLens](https://grouplens.org/datasets/movielens/) (`ml-latest-small`) and convert it to the same columns:

- `movies.csv`: `movie_id, title, year, genres` (genres separated by `|`)
- `ratings.csv`: `user_id, movie_id, rating`

## Evaluation

`python main.py evaluate` holds out 20% of ratings and compares item-based CF with a simple baseline that predicts each user's average rating (RMSE, lower is better):

| Model | RMSE |
|---|---|
| Baseline (user average) | 0.754 |
| Item-based CF | 0.774 |

On this tiny, sparse sample the baseline is about as good as CF. That is expected with 40 movies and roughly 18 ratings per user, since there is little overlap to learn from. On a larger dataset such as MovieLens, CF should pull ahead.

## Project structure

```
movie-recommender/
├── data/movies.csv, ratings.csv
├── generate_sample_data.py
├── recommender.py      # the three recommenders + evaluation
├── main.py             # command-line interface
├── requirements.txt
└── README.md
```

## Ideas to extend

- Use MovieLens data and add matrix factorization (e.g. SVD).
- Add the cold-start case: recommend popular movies to brand-new users.
- Precision@k and recall@k for ranking quality.
- Build a web app with Streamlit or Flask.
