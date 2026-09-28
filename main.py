"""Command-line interface.

Examples:
    python main.py similar "The Matrix"
    python main.py user 7
    python main.py hybrid "Toy Story=5" "Titanic=2" "Shrek=4"
    python main.py evaluate
"""
import argparse
import pandas as pd
from recommender import load_data, ContentBased, ItemBasedCF, Hybrid, evaluate

pd.set_option("display.width", 120)


def main():
    p = argparse.ArgumentParser(description="Movie recommendation system")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("similar", help="content-based: movies similar to a title")
    s.add_argument("title"); s.add_argument("-n", type=int, default=5)
    u = sub.add_parser("user", help="collaborative: recommendations for an existing user id")
    u.add_argument("user_id", type=int); u.add_argument("-n", type=int, default=5)
    h = sub.add_parser("hybrid", help='hybrid: give your own ratings as "Title=Rating"')
    h.add_argument("ratings", nargs="+"); h.add_argument("-n", type=int, default=5)
    sub.add_parser("evaluate", help="RMSE of collaborative filtering vs baseline")
    args = p.parse_args()

    movies, ratings = load_data()
    if args.cmd == "similar":
        print(ContentBased(movies).similar_to(args.title, args.n).to_string(index=False))
    elif args.cmd == "user":
        print(ItemBasedCF(movies, ratings).recommend_for_user(args.user_id, args.n).to_string(index=False))
    elif args.cmd == "hybrid":
        lookup = {t.lower(): m for t, m in zip(movies.title, movies.movie_id)}
        mine = {}
        for item in args.ratings:
            title, value = item.rsplit("=", 1)
            mine[lookup[title.strip().lower()]] = float(value)
        print(Hybrid(movies, ratings).recommend(mine, args.n).to_string(index=False))
    else:
        r = evaluate(movies, ratings)
        print(f"Test ratings: {r['n_test']}")
        print(f"Baseline (user average) RMSE: {r['baseline_rmse']:.3f}")
        print(f"Item-based CF RMSE:           {r['cf_rmse']:.3f}")


if __name__ == "__main__":
    main()
