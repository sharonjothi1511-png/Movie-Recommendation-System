"""Create a small sample dataset: 40 real movies + simulated user ratings.

The ratings are SIMULATED (each fake user prefers two genres), so results on
this data only demonstrate that the code works. To use real data, drop
MovieLens files into data/ (see README).
"""
import numpy as np
import pandas as pd

MOVIES = [
    ("The Matrix", 1999, "Action|Sci-Fi"),
    ("Inception", 2010, "Action|Sci-Fi|Thriller"),
    ("Interstellar", 2014, "Adventure|Drama|Sci-Fi"),
    ("The Dark Knight", 2008, "Action|Crime|Drama"),
    ("Toy Story", 1995, "Animation|Adventure|Comedy"),
    ("Finding Nemo", 2003, "Animation|Adventure|Comedy"),
    ("The Lion King", 1994, "Animation|Adventure|Drama"),
    ("Shrek", 2001, "Animation|Comedy|Fantasy"),
    ("Titanic", 1997, "Drama|Romance"),
    ("The Notebook", 2004, "Drama|Romance"),
    ("La La Land", 2016, "Comedy|Drama|Musical|Romance"),
    ("Pride & Prejudice", 2005, "Drama|Romance"),
    ("The Godfather", 1972, "Crime|Drama"),
    ("Goodfellas", 1990, "Crime|Drama"),
    ("Pulp Fiction", 1994, "Comedy|Crime|Thriller"),
    ("The Shawshank Redemption", 1994, "Crime|Drama"),
    ("Forrest Gump", 1994, "Comedy|Drama|Romance"),
    ("The Hangover", 2009, "Comedy"),
    ("Superbad", 2007, "Comedy"),
    ("Bridesmaids", 2011, "Comedy|Romance"),
    ("Jurassic Park", 1993, "Action|Adventure|Sci-Fi"),
    ("Avengers: Endgame", 2019, "Action|Adventure|Sci-Fi"),
    ("Iron Man", 2008, "Action|Adventure|Sci-Fi"),
    ("Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi"),
    ("The Conjuring", 2013, "Horror|Mystery|Thriller"),
    ("Get Out", 2017, "Horror|Mystery|Thriller"),
    ("A Quiet Place", 2018, "Drama|Horror|Sci-Fi"),
    ("Hereditary", 2018, "Drama|Horror|Mystery"),
    ("The Fellowship of the Ring", 2001, "Adventure|Drama|Fantasy"),
    ("Harry Potter and the Sorcerer's Stone", 2001, "Adventure|Family|Fantasy"),
    ("The Hobbit: An Unexpected Journey", 2012, "Adventure|Fantasy"),
    ("Spirited Away", 2001, "Animation|Adventure|Fantasy"),
    ("Up", 2009, "Animation|Adventure|Comedy"),
    ("Coco", 2017, "Animation|Adventure|Family|Musical"),
    ("Parasite", 2019, "Comedy|Drama|Thriller"),
    ("Whiplash", 2014, "Drama|Musical"),
    ("Gladiator", 2000, "Action|Adventure|Drama"),
    ("Braveheart", 1995, "Action|Drama|History"),
    ("Se7en", 1995, "Crime|Mystery|Thriller"),
    ("The Silence of the Lambs", 1991, "Crime|Horror|Thriller"),
]

N_USERS = 60
rng = np.random.default_rng(42)

movies = pd.DataFrame(MOVIES, columns=["title", "year", "genres"])
movies.insert(0, "movie_id", range(1, len(movies) + 1))
all_genres = sorted({g for gs in movies.genres for g in gs.split("|")})

rows = []
for user_id in range(1, N_USERS + 1):
    favs = set(rng.choice(all_genres, size=2, replace=False))
    n_rated = rng.integers(15, 23)
    for idx in rng.choice(len(movies), size=n_rated, replace=False):
        genres = set(movies.genres[idx].split("|"))
        affinity = len(genres & favs) / len(genres)
        rating = np.clip(round(2.5 + 2.0 * affinity + rng.normal(0, 0.6)), 1, 5)
        rows.append((user_id, int(movies.movie_id[idx]), int(rating)))

ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])
movies.to_csv("data/movies.csv", index=False)
ratings.to_csv("data/ratings.csv", index=False)
print(f"Wrote {len(movies)} movies and {len(ratings)} ratings from {N_USERS} users.")
