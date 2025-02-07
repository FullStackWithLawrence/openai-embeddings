# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test this model's Pinecone helper class.
"""

import os

import pinecone as oem_pinecone
import pytest  # pylint: disable=unused-import
from pinecone import Pinecone

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
        assert isinstance(pinecone.index, oem_pinecone.Index)

    def test_05_delete(self):
        """Test that the index can be deleted."""
        pinecone_index = PineconeIndex()

        # pylint: disable=E1101
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

        # pylint: disable=E1101
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
        assert isinstance(pinecone_index.index, oem_pinecone.Index)
        pinecone_index.delete()

    def test_07_load_pdf(self):
        """Test that we can load a PDF document to the index."""
        if not os.path.exists("./data/test_07_load.pdf"):
            pytest.skip("File './data/test_07_load.pdf' does not exist")

        pinecone = PineconeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.pdf_loader(filepath="./data/test_07_load.pdf")
        except Exception as e:
            assert False, f"Pinecone.load_pdf() failed with exception: {e}"
        pinecone.delete()
