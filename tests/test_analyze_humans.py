import pytest

from evaluation.analyze_humans import (
    cohens_kappa,
    fit_threshold,
    fleiss_kappa,
    krippendorff_alpha_interval,
    majority,
    parse_tile_pairs,
)


def test_cohens_kappa_known_values():
    assert cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    # Classic 2x2 example: 20/50 agree-yes, 15/50 agree-no -> kappa 0.4
    r1 = ["y"] * 25 + ["n"] * 25
    r2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(r1, r2) == pytest.approx(0.4)


def test_fleiss_kappa_matches_wikipedia_example():
    # Fleiss (1971) worked example as reproduced on Wikipedia: kappa = 0.210.
    table = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    ratings = [[str(k) for k, c in enumerate(row) for _ in range(c)] for row in table]
    assert fleiss_kappa(ratings) == pytest.approx(0.210, abs=1e-3)


def test_krippendorff_alpha_interval():
    assert krippendorff_alpha_interval([[1, 1], [3, 3], [5, 5]]) == pytest.approx(1.0)
    disagree = krippendorff_alpha_interval([[1, 5], [5, 1], [1, 5]])
    assert disagree < 0
    assert krippendorff_alpha_interval([[2, None], [3, 3], [4, 4]]) == pytest.approx(1.0)


def test_parse_tile_pairs_and_majority():
    assert parse_tile_pairs("12 34") == {(0, 1), (2, 3)}
    assert parse_tile_pairs("21,") == {(0, 1)}
    assert parse_tile_pairs("") == set()
    with pytest.raises(ValueError):
        parse_tile_pairs("11")
    assert majority(["A", "A", "B"]) == "A"
    assert majority(["A", "B"]) == "tie"


def test_fit_threshold_separates_clean_data():
    distances = [0.05, 0.08, 0.1, 0.4, 0.5, 0.6]
    labels = [True, True, True, False, False, False]
    assert fit_threshold(distances, labels) == 0.1
