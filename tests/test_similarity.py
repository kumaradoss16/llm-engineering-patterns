import pytest

from app.patterns.similarity import (
    SimilarityError,
    cosine_similarity,
)


def test_identical_vectors_have_similarity_one() -> None:
    vector = [1.0, 2.0, 3.0]

    result = cosine_similarity(vector, vector)

    assert result == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity() -> None:
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(0.0)


def test_opposite_vectors_have_negative_similarity() -> None:
    vector_a = [1.0, 0.0]
    vector_b = [-1.0, 0.0]

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(-1.0)


def test_different_lengths_raise_error() -> None:
    with pytest.raises(SimilarityError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_zero_vector_returns_zero() -> None:
    result = cosine_similarity([0.0, 0.0], [1.0, 2.0])

    assert result == 0.0