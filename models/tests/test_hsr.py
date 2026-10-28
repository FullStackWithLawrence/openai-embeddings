# -*- coding: utf-8 -*-
# pylint: disable=E0611,E1101
# flake8: noqa: F401
"""
Test integrity of base class.
"""
from unittest.mock import MagicMock, patch

import pytest  # pylint: disable=unused-import
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from models.hybrid_search_retriever import HybridSearchRetriever
from models.pinecone import PineconeIndex


class TestSalesSupportModel:
    """Test HybridSearchRetriever class."""

    def test_01_basic(self):
        """Ensure that we can instantiate the class."""

        # pylint: disable=broad-except
        try:
            HybridSearchRetriever()
        except Exception as e:
            assert False, f"initialization of HybridSearchRetriever() failed with exception: {e}"

    def test_02_class_aatribute_types(self):
        """ensure that class attributes are of the correct type"""

        hsr = HybridSearchRetriever()
        assert isinstance(hsr.chat, ChatOpenAI)
        assert isinstance(hsr.pinecone, PineconeIndex)


class TestHybridSearchRetrieverOffline:
    """Test HybridSearchRetriever with mock Pinecone, BM25 and chat clients."""

    def offline_hsr(self):
        """Return a HybridSearchRetriever whose Pinecone index and chat model are mocks."""
        hsr = HybridSearchRetriever()
        hsr._pinecone = MagicMock()  # pylint: disable=protected-access
        hsr._chat = MagicMock()  # pylint: disable=protected-access
        hsr._chat.invoke.return_value = AIMessage(content="ANSWER")  # pylint: disable=protected-access
        hsr._pinecone.vector_store.similarity_search.return_value = [  # pylint: disable=protected-access
            Document(page_content="fact one"),
            Document(page_content="fact two"),
        ]
        return hsr

    def test_retriever(self):
        """The retriever combines the index, the OpenAI embeddings and a BM25 encoder."""
        hsr = self.offline_hsr()
        with (
            patch("models.hybrid_search_retriever.BM25Encoder") as encoder_class,
            patch("models.hybrid_search_retriever.PineconeHybridSearchRetriever") as retriever_class,
        ):
            assert hsr.retriever is retriever_class.return_value
            assert hsr.retriever is retriever_class.return_value
        retriever_class.assert_called_once_with(
            embeddings=hsr.pinecone.openai_embeddings,
            sparse_encoder=encoder_class.return_value.default.return_value,
            index=hsr.pinecone.index,
        )

    def test_load(self):
        """Load() hands the path to the Pinecone index's PDF loader."""
        hsr = self.offline_hsr()
        hsr.load(filepath="/tmp/pdfs")
        hsr.pinecone.pdf_loader.assert_called_once_with(filepath="/tmp/pdfs")

    def test_rag_with_str(self):
        """Rag() accepts a string, and puts the retrieved documents in the system prompt."""
        hsr = self.offline_hsr()
        assert hsr.rag("what is a fact?") == "ANSWER"
        messages = hsr.chat.invoke.call_args.args[0]
        assert "fact one" in messages[0].content
        assert "fact two" in messages[0].content
        assert messages[1].content == "what is a fact?"

    def test_rag_with_list_content(self):
        """Rag() searches on the JSON of a message whose content is a list."""
        hsr = self.offline_hsr()
        message = HumanMessage(content=["what", "is", "a", "fact?"])
        assert hsr.rag(message) == "ANSWER"
        hsr.pinecone.vector_store.similarity_search.assert_called_once_with(query='["what", "is", "a", "fact?"]')

    def test_prompt_with_template(self):
        """Prompt_with_template() formats the template and returns the reply's text."""
        hsr = self.offline_hsr()
        prompt = PromptTemplate(input_variables=["concept"], template="Tell me about {concept}")
        assert hsr.prompt_with_template(prompt=prompt, concept="ratios") == "ANSWER"
        hsr.chat.invoke.assert_called_once_with("Tell me about ratios")
