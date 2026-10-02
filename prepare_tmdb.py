"""Convert the Kaggle 'TMDB 5000 Movie Dataset' into the CSV format used by this project.

Usage:
    python -m src.prepare_tmdb tmdb_5000_movies.csv tmdb_5000_credits.csv data/movies_tmdb.csv
"""
import ast
import sys

import pandas as pd


def _names(cell, limit=None):
    try:
        items = [d["name"] for d in ast.literal_eval(cell)]
    except (ValueError, SyntaxError):
        return ""
    return "|".join(items[:limit] if limit else items)


def _director(cell):
    try:
        return next((d["name"] for d in ast.literal_eval(cell) if d.get("job") == "Director"), "")
    except (ValueError, SyntaxError):
        return ""


def main(movies_csv, credits_csv, out_csv):
    m = pd.read_csv(movies_csv)
    c = pd.read_csv(credits_csv).rename(columns={"movie_id": "id"})
    df = m.merge(c[["id", "cast", "crew"]], on="id")
    out = pd.DataFrame({
        "title": df["title"],
        "year": pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int).replace(0, ""),
        "genres": df["genres"].map(_names),
        "keywords": df["keywords"].map(_names),
        "cast": df["cast"].map(lambda x: _names(x, 3)),
        "director": df["crew"].map(_director),
        "overview": df["overview"].fillna(""),
    })
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} movies to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
