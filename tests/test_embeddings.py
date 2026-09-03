import numpy as np
import pytest

from tests.conftest import FakeEmbeddings


def test_embed_documents_shape():
    emb = FakeEmbeddings(dimension=32)
    result = emb.embed_documents(["hello", "world"])
    assert result.shape == (2, 32)


def test_embed_documents_normalized():
    emb = FakeEmbeddings(dimension=32)
    result = emb.embed_documents(["test text"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_query_shape():
    emb = FakeEmbeddings(dimension=32)
    result = emb.embed_query("hello")
    assert result.shape == (1, 32)


def test_embed_query_normalized():
    emb = FakeEmbeddings(dimension=32)
    result = emb.embed_query("query text")
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5


def test_empty_list_returns_empty_array():
    emb = FakeEmbeddings(dimension=32)
    result = emb.embed_documents([])
    assert result.shape == (0, 32)


def test_dimension_property():
    emb = FakeEmbeddings(dimension=64)
    assert emb.dimension == 64


def test_deterministic_for_same_input():
    emb = FakeEmbeddings(dimension=32)
    a = emb.embed_documents(["wheat rust"])
    b = emb.embed_documents(["wheat rust"])
    np.testing.assert_array_equal(a, b)


def test_different_texts_different_embeddings():
    emb = FakeEmbeddings(dimension=32)
    a = emb.embed_documents(["wheat rust"])
    b = emb.embed_documents(["cotton bollworm"])
    assert not np.allclose(a, b)


def test_dtype_is_float32():
    emb = FakeEmbeddings(dimension=32)
    result = emb.embed_documents(["test"])
    assert result.dtype == np.float32
