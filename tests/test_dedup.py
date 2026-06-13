import numpy as np
import pytest

from src.rag_lit.dedup import find_near_duplicates, DEFAULT_NEAR_DUPLICATE_THRESHOLD


def test_identical_vectors_flagged_as_duplicates():
    ids = ["A", "B", "C"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # identical to A
        [0.0, 1.0, 0.0],  # orthogonal to A and B
    ]

    result = find_near_duplicates(ids, embeddings)

    assert result["A"] == ["B"]
    assert result["B"] == ["A"]
    assert "C" not in result


def test_near_identical_vectors_above_threshold_flagged():
    ids = ["A", "B"]
    # Very small perturbation -> cosine similarity > 0.99
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.999, 0.001, 0.0],
    ]

    result = find_near_duplicates(ids, embeddings, threshold=0.95)

    assert result["A"] == ["B"]
    assert result["B"] == ["A"]


def test_dissimilar_vectors_not_flagged():
    ids = ["A", "B", "C"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    result = find_near_duplicates(ids, embeddings)

    assert result == {}


def test_unnormalized_vectors_still_compared_via_cosine():
    ids = ["A", "B"]
    # Same direction, different magnitude -> cosine similarity == 1.0
    embeddings = [
        [1.0, 0.0],
        [5.0, 0.0],
    ]

    result = find_near_duplicates(ids, embeddings)

    assert result["A"] == ["B"]
    assert result["B"] == ["A"]


def test_below_threshold_not_flagged():
    ids = ["A", "B"]
    # cosine similarity == 0.6, below default threshold
    embeddings = [
        [1.0, 0.0],
        [0.6, 0.8],
    ]

    result = find_near_duplicates(ids, embeddings, threshold=DEFAULT_NEAR_DUPLICATE_THRESHOLD)

    assert result == {}


def test_single_result_returns_empty():
    result = find_near_duplicates(["A"], [[1.0, 0.0, 0.0]])
    assert result == {}


def test_empty_result_set_returns_empty():
    result = find_near_duplicates([], [])
    assert result == {}


def test_multiple_duplicates_in_group():
    ids = ["A", "B", "C", "D"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    result = find_near_duplicates(ids, embeddings)

    assert result["A"] == ["B", "C"]
    assert result["B"] == ["A", "C"]
    assert result["C"] == ["A", "B"]
    assert "D" not in result


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        find_near_duplicates(["A", "B"], [[1.0, 0.0]])


def test_numpy_array_input_accepted():
    ids = ["A", "B"]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    result = find_near_duplicates(ids, embeddings)

    assert result["A"] == ["B"]
