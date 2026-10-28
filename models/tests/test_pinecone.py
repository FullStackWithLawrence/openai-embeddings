# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test this model's Pinecone helper class.
"""

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from pinecone import Pinecone
from pinecone.db_data import Index
from pinecone.exceptions import PineconeApiException
from pydantic import SecretStr

from models.conf import settings
from models.pinecone import PineconeIndex


class TestPinecone:
    """Test HybridSearchRetriever class."""

    def test_01_can_instantiate(self):
        """Ensure that we instantiate the object."""
        # pylint: disable=broad-except
        try:
            PineconeIndex()
        except Exception as e:
            assert False, f"Pinecone() failed with exception: {e}"

    def test_02_init(self):
        """Ensure that we can initialize Pinecone."""
        pinecone = PineconeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.init()
        except Exception as e:
            assert False, f"Pinecone.init() failed with exception: {e}"

    def test_03_index(self):
        """Test that the index name is correct."""
        pinecone = PineconeIndex()
        assert pinecone.index_name == settings.pinecone_index_name

    def test_04_initialize(self):
        """Test that the index initializes."""
        pinecone = PineconeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.initialize()
        except Exception as e:
            assert False, f"Pinecone.initialize() failed with exception: {e}"
        assert isinstance(pinecone.index, Index)

    def test_05_delete(self):
        """Test that the index can be deleted."""
        pinecone_index = PineconeIndex()

        if not isinstance(settings.pinecone_api_key, SecretStr):
            raise ValueError("Pinecone API key is not a SecretStr")
        # pylint: disable=no-member
        api_key = settings.pinecone_api_key.get_secret_value()
        pinecone = Pinecone(api_key=api_key)
        indexes = pinecone.list_indexes().names()
        assert pinecone_index.index_name in indexes
        # pylint: disable=broad-except
        try:
            pinecone_index.delete()
        except Exception as e:
            assert False, f"Pinecone.delete() failed with exception: {e}"

    def test_06_create(self):
        """Test that the index can be created."""
        pinecone_index = PineconeIndex()

        if not isinstance(settings.pinecone_api_key, SecretStr):
            raise ValueError("Pinecone API key is not a SecretStr")

        # pylint: disable=no-member
        api_key = settings.pinecone_api_key.get_secret_value()
        pinecone = Pinecone(api_key=api_key)

        indexes = pinecone.list_indexes().names()
        if pinecone_index.index_name in indexes:
            pinecone_index.delete()

        # pylint: disable=broad-except
        try:
            pinecone_index.create()
        except Exception as e:
            assert False, f"Pinecone.create() failed with exception: {e}"
        assert isinstance(pinecone_index.index, Index)
        pinecone_index.delete()

    def test_07_load_pdf(self):
        """Test that we can load a PDF document to the index."""
        HERE = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(HERE, "mock_data", "test_load.pdf")

        if not os.path.exists(test_file):
            pytest.skip(f"File {test_file} does not exist")

        pinecone = PineconeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.pdf_loader(filepath=test_file)
        except Exception as e:
            assert False, f"Pinecone.load_pdf() failed with exception: {e}"
        pinecone.delete()


def offline_index(existing_indexes=("unit-test",)):
    """
    Return a PineconeIndex wired to a mock Pinecone client, without calling the constructor.

    The constructor talks to Pinecone, so these tests bypass it and set the client and
    index name directly.

    :param existing_indexes: The index names that the mock client lists.
    :returns: The PineconeIndex and its mock client.
    """
    index = PineconeIndex.__new__(PineconeIndex)
    client = MagicMock(spec=Pinecone)
    client.list_indexes.return_value.names.return_value = list(existing_indexes)
    index._pinecone = client  # pylint: disable=protected-access
    index._index_name = "unit-test"  # pylint: disable=protected-access
    return index, client


class TestPineconeOffline:
    """Test PineconeIndex against a mock Pinecone client."""

    def test_index_stats_without_index(self):
        """Index_stats says so when there is no index."""
        index, _ = offline_index()
        with patch.object(PineconeIndex, "index", new_callable=PropertyMock, return_value=None):
            assert index.index_stats == "Index not initialized."

    def test_initialized_without_client(self):
        """An index without a Pinecone client is not initialized."""
        index, _ = offline_index()
        with patch.object(PineconeIndex, "pinecone", new_callable=PropertyMock, return_value=None):
            assert index.initialized is False

    def test_text_splitter_is_lazy(self):
        """The text splitter is built once."""
        index, _ = offline_index()
        first = index.text_splitter
        assert index.text_splitter is first

    def test_vector_store_creates_missing_index(self):
        """Vector_store creates the index when it doesn't exist yet."""
        index, client = offline_index(existing_indexes=())
        with (
            patch("models.pinecone.PineconeVectorStore") as vector_store_class,
            patch.object(PineconeIndex, "openai_embeddings", new_callable=PropertyMock),
        ):
            assert index.vector_store is vector_store_class.return_value
        client.create_index.assert_called()

    def test_delete_missing_index(self):
        """Delete() does nothing when the index doesn't exist."""
        index, client = offline_index(existing_indexes=())
        index.delete()
        client.delete_index.assert_not_called()

    def test_delete_existing_index(self):
        """Delete() deletes the index when it exists."""
        index, client = offline_index()
        index.delete()
        client.delete_index.assert_called_once_with("unit-test")

    def test_create_ignores_api_exception(self):
        """Create() swallows a Pinecone API error, such as an index that already exists."""
        index, client = offline_index()
        client.create_index.side_effect = PineconeApiException(status=409, reason="conflict")
        index.create()
        client.create_index.assert_called_once()

    def test_pdf_loader(self, tmp_path):
        """Pdf_loader() embeds every page of every PDF in a folder, and adds it to the vector store."""
        (tmp_path / "a.pdf").write_bytes(b"")
        (tmp_path / "b.pdf").write_bytes(b"")
        index, _ = offline_index()
        page = MagicMock(page_content="Some text about accounting.")
        with (
            patch.object(PineconeIndex, "initialize") as initialize,
            patch("models.pinecone.PyPDFLoader") as loader_class,
            patch.object(PineconeIndex, "openai_embeddings", new_callable=PropertyMock) as embeddings,
            patch.object(PineconeIndex, "vector_store", new_callable=PropertyMock) as vector_store,
            patch.object(PineconeIndex, "index_stats", new_callable=PropertyMock, return_value="{}"),
        ):
            loader_class.return_value.load.return_value = [page, page]
            index.pdf_loader(filepath=str(tmp_path))
        initialize.assert_called_once()
        assert loader_class.call_count == 2
        assert embeddings.return_value.embed_documents.call_count == 4
        assert vector_store.return_value.add_documents.call_count == 4
