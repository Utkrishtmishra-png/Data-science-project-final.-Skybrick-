"""Content-based movie recommender (TF-IDF + cosine similarity)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "movies_sample.csv"
LIST_COLS = ["genres", "keywords", "cast"]


def _clean_tokens(value: str) -> str:
    """'Sci-Fi|Action' -> 'scifi action' ; multi-word tokens are glued so 'Ryan Gosling' stays one token."""
    if pd.isna(value) or not str(value).strip():
        return ""
    return " ".join(
        part.strip().lower().replace(" ", "").replace("-", "").replace(".", "")
        for part in str(value).split("|")
        if part.strip()
    )


def load_movies(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    required = {"title", "genres", "overview"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    for col in ("keywords", "cast", "director", "year"):
        if col not in df.columns:
            df[col] = ""
    return df.drop_duplicates(subset="title").reset_index(drop=True)


def build_soup(df: pd.DataFrame) -> pd.Series:
    """Combine metadata into one text 'soup'. Genres/director are repeated to weight them higher."""
    genres = df["genres"].map(_clean_tokens)
    keywords = df["keywords"].map(_clean_tokens)
    cast = df["cast"].map(lambda v: _clean_tokens("|".join(str(v).split("|")[:3])))
    director = df["director"].map(lambda v: _clean_tokens(v))
    return (genres + " ") * 2 + keywords + " " + cast + " " + (director + " ") * 2 + df["overview"].astype(str)


@dataclass
class Recommendation:
    title: str
    year: str
    genres: str
    director: str
    overview: str
    score: float


class MovieRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(build_soup(df))
        self._index = pd.Series(df.index, index=df["title"].str.lower())

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_DATA) -> "MovieRecommender":
        return cls(load_movies(path))

    @property
    def titles(self) -> list[str]:
        return sorted(self.df["title"].tolist())

    def _to_rec(self, idx: int, score: float) -> Recommendation:
        r = self.df.iloc[idx]
        return Recommendation(
            title=r["title"], year=str(r["year"]), genres=str(r["genres"]).replace("|", ", "),
            director=str(r["director"]), overview=str(r["overview"]), score=float(score),
        )

    def recommend(self, title: str, n: int = 5, genre: str | None = None) -> list[Recommendation]:
        key = title.strip().lower()
        if key not in self._index:
            raise KeyError(f"'{title}' not found in the dataset")
        idx = int(self._index[key])
        scores = linear_kernel(self.matrix[idx], self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        out: list[Recommendation] = []
        for i in ranked:
            if i == idx:
                continue
            if genre and genre.lower() not in str(self.df.at[i, "genres"]).lower().split("|"):
                continue
            out.append(self._to_rec(int(i), scores[i]))
            if len(out) == n:
                break
        return out

    def search(self, query: str, n: int = 5) -> list[Recommendation]:
        """Free-text search: describe what you want to watch."""
        scores = linear_kernel(self.vectorizer.transform([_clean_tokens(query.replace(" ", "|")) + " " + query]), self.matrix).ravel()
        return [self._to_rec(int(i), scores[i]) for i in scores.argsort()[::-1][:n] if scores[i] > 0]

    def all_genres(self) -> list[str]:
        return sorted({g for row in self.df["genres"] for g in str(row).split("|") if g})
