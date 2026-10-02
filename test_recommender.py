import pytest
from src.recommender import MovieRecommender


@pytest.fixture(scope="module")
def rec():
    return MovieRecommender.from_csv()


def test_returns_n_results_without_self(rec):
    out = rec.recommend("Inception", n=5)
    assert len(out) == 5
    assert all(r.title != "Inception" for r in out)


def test_scores_sorted_descending(rec):
    scores = [r.score for r in rec.recommend("The Matrix", n=6)]
    assert scores == sorted(scores, reverse=True)


def test_sensible_neighbours(rec):
    titles = [r.title for r in rec.recommend("The Godfather", n=3)]
    assert "The Godfather Part II" in titles


def test_genre_filter(rec):
    assert all("Animation" in r.genres for r in rec.recommend("Toy Story", n=4, genre="Animation"))


def test_unknown_title(rec):
    with pytest.raises(KeyError):
        rec.recommend("Not A Movie")
