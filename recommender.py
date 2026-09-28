"""Content-based, collaborative (item-based) and hybrid movie recommenders."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(movies_path="data/movies.csv", ratings_path="data/ratings.csv"):
    """Load movies (movie_id, title, year, genres) and ratings (user_id, movie_id, rating)."""
    return pd.read_csv(movies_path), pd.read_csv(ratings_path)


class ContentBased:
    """Recommend movies with similar genres (TF-IDF + cosine similarity)."""

    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)
        tfidf = TfidfVectorizer(tokenizer=lambda s: s.split("|"), token_pattern=None, lowercase=False)
        self.matrix = tfidf.fit_transform(self.movies["genres"])
        self.sim = cosine_similarity(self.matrix)
        self.index = {t.lower(): i for i, t in enumerate(self.movies["title"])}

    def similar_to(self, title, n=5):
        i = self.index[title.lower()]
        order = np.argsort(-self.sim[i])
        order = [j for j in order if j != i][:n]
        return self.movies.iloc[order][["title", "year", "genres"]].assign(score=self.sim[i][order].round(3))

    def scores_from_liked(self, liked_ids):
        """Average genre similarity of every movie to the movies the user liked."""
        idx = [self.movies.index[self.movies.movie_id == m][0] for m in liked_ids]
        return self.sim[:, idx].mean(axis=1) if idx else np.zeros(len(self.movies))


class ItemBasedCF:
    """Item-based collaborative filtering on mean-centered ratings."""

    def __init__(self, movies, ratings):
        self.movies = movies.reset_index(drop=True)
        self.ids = self.movies["movie_id"].tolist()
        self.pos = {m: i for i, m in enumerate(self.ids)}
        table = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
        table = table.reindex(columns=self.ids)
        self.user_means = table.mean(axis=1)
        self.global_mean = ratings["rating"].mean()
        centered = table.sub(self.user_means, axis=0).fillna(0.0)
        self.sim = cosine_similarity(centered.T.values)
        np.fill_diagonal(self.sim, 0.0)
        self.table = table

    def predict_for_ratings(self, user_ratings):
        """user_ratings: {movie_id: rating}. Returns a predicted rating for every movie."""
        mean = np.mean(list(user_ratings.values())) if user_ratings else self.global_mean
        rated = [(self.pos[m], r - mean) for m, r in user_ratings.items() if m in self.pos]
        preds = np.full(len(self.ids), mean)
        if not rated:
            return preds
        idx, dev = zip(*rated)
        sims = self.sim[:, list(idx)]
        denom = np.abs(sims).sum(axis=1)
        num = sims @ np.array(dev)
        preds = mean + np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-9)
        return np.clip(preds, 1, 5)

    def recommend_for_user(self, user_id, n=5):
        row = self.table.loc[user_id].dropna()
        preds = self.predict_for_ratings(row.to_dict())
        seen = set(row.index)
        out = self.movies.assign(predicted_rating=preds.round(2))
        out = out[~out.movie_id.isin(seen)]
        return out.sort_values("predicted_rating", ascending=False).head(n)[["title", "year", "genres", "predicted_rating"]]


class Hybrid:
    """Blend collaborative and content-based scores (alpha = weight of collaborative)."""

    def __init__(self, movies, ratings, alpha=0.6):
        self.cb = ContentBased(movies)
        self.cf = ItemBasedCF(movies, ratings)
        self.alpha = alpha

    def recommend(self, user_ratings, n=5, like_threshold=4):
        cf = (self.cf.predict_for_ratings(user_ratings) - 1) / 4
        liked = [m for m, r in user_ratings.items() if r >= like_threshold]
        cb = self.cb.scores_from_liked(liked)
        score = self.alpha * cf + (1 - self.alpha) * cb
        out = self.cf.movies.assign(score=score.round(3))
        out = out[~out.movie_id.isin(user_ratings)]
        return out.sort_values("score", ascending=False).head(n)[["title", "year", "genres", "score"]]


def evaluate(movies, ratings, test_size=0.2, seed=42):
    """Hold out a share of ratings, then compare CF against a simple baseline (RMSE, lower is better)."""
    test = ratings.sample(frac=test_size, random_state=seed)
    train = ratings.drop(test.index)
    cf = ItemBasedCF(movies, train)
    by_user = {u: g.set_index("movie_id")["rating"].to_dict() for u, g in train.groupby("user_id")}
    cf_err, base_err = [], []
    for row in test.itertuples():
        history = by_user.get(row.user_id, {})
        pred = cf.predict_for_ratings(history)[cf.pos[row.movie_id]]
        base = np.mean(list(history.values())) if history else cf.global_mean
        cf_err.append((pred - row.rating) ** 2)
        base_err.append((base - row.rating) ** 2)
    return {"cf_rmse": float(np.sqrt(np.mean(cf_err))), "baseline_rmse": float(np.sqrt(np.mean(base_err))), "n_test": len(test)}
