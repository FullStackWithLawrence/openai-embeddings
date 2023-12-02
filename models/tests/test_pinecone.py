# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test this model's Pinecone helper class.
"""

import os

import pinecone as oem_pinecone
import pytest  # pylint: disable=unused-import

from models.const import Config
from models.pinecone import PineConeIndex


class TestPinecone:
    """Test HybridSearchRetriever class."""

    def test_01_can_instantiate(self):
        """Ensure that we instantiate the object."""
        # pylint: disable=broad-except
        try:
            PineConeIndex()
        except Exception as e:
            assert False, f"Pinecone() failed with exception: {e}"

    def test_02_init(self):
        """Ensure that we can initialize Pinecone."""
        pinecone = PineConeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.init()
        except Exception as e:
            assert False, f"Pinecone.init() failed with exception: {e}"

    def test_03_index(self):
        """Test that the index name is correct."""
        pinecone = PineConeIndex()
        assert pinecone.index_name == Config.PINECONE_INDEX_NAME

    def test_04_initialize(self):
        """Test that the index initializes."""
        pinecone = PineConeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.initialize()
        except Exception as e:
            assert False, f"Pinecone.initialize() failed with exception: {e}"
        assert isinstance(pinecone.index, oem_pinecone.Index)

    def test_05_delete(self):
        """Test that the index can be deleted."""
        pinecone = PineConeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.delete()
        except Exception as e:
            assert False, f"Pinecone.delete() failed with exception: {e}"

    def test_06_create(self):
        """Test that the index can be created."""
        pinecone = PineConeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.create()
        except Exception as e:
            assert False, f"Pinecone.create() failed with exception: {e}"
        pinecone.delete()

    def test_07_load_pdf(self):
        """Test that we can load a PDF document to the index."""
        if not os.path.exists("./data/test_07_load.pdf"):
            pytest.skip("File './data/test_07_load.pdf' does not exist")

        pinecone = PineConeIndex()
        # pylint: disable=broad-except
        try:
            pinecone.pdf_loader(filepath="./data/test_07_load.pdf")
        except Exception as e:
            assert False, f"Pinecone.load_pdf() failed with exception: {e}"
        pinecone.delete()
