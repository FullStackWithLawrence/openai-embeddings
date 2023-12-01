# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Index

from models.ssm import SalesSupportModel, TextSplitter


class TestSalesSupportModel:
    """Test SalesSupportModel class."""

    def test_01_basic(self):
        """Ensure that we can instantiate the class."""

        # pylint: disable=broad-except
        try:
            SalesSupportModel()
        except Exception as e:
            assert False, f"initialization of SalesSupportModel() failed with exception: {e}"

    def test_02_class_aatribute_types(self):
        """ensure that class attributes are of the correct type"""

        ssm = SalesSupportModel()
        assert isinstance(ssm.chat, ChatOpenAI)
        assert isinstance(ssm.pinecone_index, Index)
        assert isinstance(ssm.text_splitter, TextSplitter)
        assert isinstance(ssm.openai_embeddings, OpenAIEmbeddings)
