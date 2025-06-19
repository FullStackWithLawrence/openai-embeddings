# -*- coding: utf-8 -*-
# pylint: disable=E0611,E1101
"""
Hybrid Search Retriever. A class that combines the following:
    - OpenAI prompting and ChatModel
    - PromptingWrapper
    - Vector embedding with Pinecone
    - Hybrid Retriever to combine vector embeddings with text search

Provides a pdf loader program that extracts text, vectorizes, and
loads into a Pinecone dot product vector database that is dimensioned
to match OpenAI embeddings.

See: https://python.langchain.com/docs/modules/model_io/llms/llm_caching
     https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
     https://python.langchain.com/docs/integrations/retrievers/pinecone_hybrid_search
"""

# general purpose imports
import logging
import textwrap
from typing import Union

# embedding
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

# pinecone integration
from langchain_community.cache import InMemoryCache

# hybrid search capability
from langchain_community.retrievers.pinecone_hybrid_search import (
    PineconeHybridSearchRetriever,
)

# from langchain_community.chat_models import ChatOpenAI
# prompting and chat
from langchain_openai import ChatOpenAI
from pinecone_text.sparse import BM25Encoder  # pylint: disable=import-error

# this project
from models.conf import settings
from models.pinecone import PineconeIndex


logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.INFO)
logger = logging.getLogger(__name__)


class HybridSearchRetriever:
    """Hybrid Search Retriever"""

    _chat: ChatOpenAI = None
    _b25_encoder: BM25Encoder = None
    _pinecone: PineconeIndex = None
    _retriever: PineconeHybridSearchRetriever = None

    def __init__(self):
        """Constructor"""
        set_llm_cache(InMemoryCache())

    @property
    def pinecone(self) -> PineconeIndex:
        """PineconeIndex lazy read-only property."""
        if self._pinecone is None:
            self._pinecone = PineconeIndex()
        return self._pinecone

    # prompting wrapper
    @property
    def chat(self) -> ChatOpenAI:
        """ChatOpenAI lazy read-only property."""
        if self._chat is None:
            self._chat = ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),  # pylint: disable=no-member
                organization=settings.openai_api_organization,
                cache=settings.openai_chat_cache,
                max_retries=settings.openai_chat_max_retries,
                model=settings.openai_chat_model_name,
                temperature=settings.openai_chat_temperature,
            )
        return self._chat

    @property
    def bm25_encoder(self) -> BM25Encoder:
        """BM25Encoder lazy read-only property."""
        if self._b25_encoder is None:
            self._b25_encoder = BM25Encoder().default()
        return self._b25_encoder

    @property
    def retriever(self) -> PineconeHybridSearchRetriever:
        """PineconeHybridSearchRetriever lazy read-only property."""
        if self._retriever is None:
            self._retriever = PineconeHybridSearchRetriever(
                embeddings=self.pinecone.openai_embeddings, sparse_encoder=self.bm25_encoder, index=self.pinecone.index
            )
        return self._retriever

    def cached_chat_request(
        self, system_message: Union[str, SystemMessage], human_message: Union[str, HumanMessage]
    ) -> BaseMessage:
        """Cached chat request."""
        if not isinstance(system_message, SystemMessage):
            logger.info("Converting system message to SystemMessage")
            system_message = SystemMessage(content=str(system_message))

        if not isinstance(human_message, HumanMessage):
            logger.info("Converting human message to HumanMessage")
            human_message = HumanMessage(content=str(human_message))
        messages = [system_message, human_message]
        # pylint: disable=not-callable
        # retval = self.chat(messages)
        retval = self.chat.invoke(messages)
        return retval

    # pylint: disable=unused-argument
    def prompt_with_template(
        self, prompt: PromptTemplate, concept: str, model: str = settings.openai_prompt_model_name
    ) -> str:
        """Prompt with template."""
        retval = self.chat.invoke(prompt.format(concept=concept))
        return str(retval.content) if retval else "no response"

    def load(self, filepath: str):
        """Pdf loader."""
        self.pinecone.pdf_loader(filepath=filepath)

    def rag(self, human_message: Union[str, HumanMessage]):
        """
        Retrieval Augmented Generation prompt.
        1. Retrieve human message prompt: Given a user input, relevant splits are retrieved
           from storage using a Retriever.
        2. Generate: A ChatModel / LLM produces an answer using a prompt that includes
           the question and the retrieved data

        To prompt OpenAI's GPT-3 model to consider the embeddings from the Pinecone
        vector database, you would typically need to convert the embeddings back
        into a format that GPT-3 can understand, such as text. However, GPT-3 does
        not natively support direct input of embeddings.

        The typical workflow is to use the embeddings to retrieve relevant documents,
        and then use the text of these documents as part of the prompt for GPT-3.
        """
        if not isinstance(human_message, HumanMessage):
            logger.info("Converting human_message to HumanMessage")
            human_message = HumanMessage(content=human_message)

        # ---------------------------------------------------------------------
        # 1.) Retrieve relevant documents from Pinecone vector database
        # ---------------------------------------------------------------------
        documents = self.pinecone.vector_store.similarity_search(query=human_message.content)

        # Extract the text from the documents
        document_texts = [doc.page_content for doc in documents]
        leader = textwrap.dedent(
            """\n
            You are a helpful assistant. You should assume that all of the
            following bullet points that follow are completely factual.
            You should prioritize these enumerated facts when formulating your response:"""
        )
        separator = "\n\n" + "-" * 40 + "\n"
        system_message_content = f"{leader} " + "".join(
            [separator + f"{i + 1}.) {text}\n" for i, text in enumerate(document_texts)]
        )
        system_message = SystemMessage(content=system_message_content)
        # ---------------------------------------------------------------------
        # finished with hybrid search setup
        # ---------------------------------------------------------------------
        star_line = 80 * "*"
        logger.info(
            "\n%s\n"
            "rag() Retrieval Augmented Generation prompt"
            "Diagnostic information:\n"
            "  Retrieved %i related documents from Pinecone\n"
            "  System messages contains %i words\n"
            "  System Prompt:"
            "\n  <============================ BEGIN ===============================>"
            "%s"
            "\n  <============================= END ================================>\n\n",
            star_line,
            len(documents),
            len(system_message.content.split()),
            system_message.content,
        )

        # 2.) get a response from the chat model
        response = self.cached_chat_request(system_message=system_message, human_message=human_message)

        return str(response.content)
