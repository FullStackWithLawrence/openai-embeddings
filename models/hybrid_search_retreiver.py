# -*- coding: utf-8 -*-
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

# pinecone integration
import pinecone
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI

# embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_llm_cache

# prompting and chat
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

# hybrid search capability
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.vectorstores.pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# this project
from models.const import Config, Credentials
from models.pinecone import PineConeIndex, TextSplitter


###############################################################################
# initializations
###############################################################################
logging.basicConfig(level=logging.DEBUG if Config.DEBUG_MODE else logging.INFO)


class HybridSearchRetriever:
    """Hybrid Search Retriever"""

    _chat: ChatOpenAI = None
    _openai_embeddings: OpenAIEmbeddings = None
    _vector_store: Pinecone = None
    _text_splitter: TextSplitter = None
    _b25_encoder: BM25Encoder = None
    _pinecone: PineConeIndex = None
    _retriever: PineconeHybridSearchRetriever = None

    def __init__(self):
        """Constructor"""
        pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        set_llm_cache(InMemoryCache())

    @property
    def pinecone(self) -> PineConeIndex:
        """PineConeIndex lazy read-only property."""
        if self._pinecone is None:
            self._pinecone = PineConeIndex()
        return self._pinecone

    # prompting wrapper
    @property
    def chat(self) -> ChatOpenAI:
        """ChatOpenAI lazy read-only property."""
        if self._chat is None:
            self._chat = ChatOpenAI(
                api_key=Credentials.OPENAI_API_KEY,
                organization=Credentials.OPENAI_API_ORGANIZATION,
                cache=Config.OPENAI_CHAT_CACHE,
                max_retries=Config.OPENAI_CHAT_MAX_RETRIES,
                model=Config.OPENAI_CHAT_MODEL_NAME,
                temperature=Config.OPENAI_CHAT_TEMPERATURE,
            )
        return self._chat

    # embeddings
    @property
    def openai_embeddings(self) -> OpenAIEmbeddings:
        """OpenAIEmbeddings lazy read-only property."""
        if self._openai_embeddings is None:
            self._openai_embeddings = OpenAIEmbeddings(
                api_key=Credentials.OPENAI_API_KEY, organization=Credentials.OPENAI_API_ORGANIZATION
            )
        return self._openai_embeddings

    @property
    def vector_store(self) -> Pinecone:
        """Pinecone lazy read-only property."""
        if self._vector_store is None:
            self._vector_store = Pinecone(
                index=self.pinecone.index,
                embedding=self.openai_embeddings,
                text_key=Config.PINECONE_VECTORSTORE_TEXT_KEY,
            )
        return self._vector_store

    @property
    def text_splitter(self) -> TextSplitter:
        """TextSplitter lazy read-only property."""
        if self._text_splitter is None:
            self._text_splitter = TextSplitter()
        return self._text_splitter

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
                embeddings=self.openai_embeddings, sparse_encoder=self.bm25_encoder, index=self.pinecone.index
            )
        return self._retriever

    def cached_chat_request(
        self, system_message: Union[str, SystemMessage], human_message: Union[str, HumanMessage]
    ) -> BaseMessage:
        """Cached chat request."""
        if not isinstance(system_message, SystemMessage):
            logging.debug("Converting system message to SystemMessage")
            system_message = SystemMessage(content=str(system_message))

        if not isinstance(human_message, HumanMessage):
            logging.debug("Converting human message to HumanMessage")
            human_message = HumanMessage(content=str(human_message))
        messages = [system_message, human_message]
        # pylint: disable=not-callable
        retval = self.chat(messages)
        return retval

    def prompt_with_template(
        self, prompt: PromptTemplate, concept: str, model: str = Config.OPENAI_PROMPT_MODEL_NAME
    ) -> str:
        """Prompt with template."""
        llm = OpenAI(model=model)
        retval = llm(prompt.format(concept=concept))
        return retval

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
            logging.debug("Converting human_message to HumanMessage")
            human_message = HumanMessage(content=human_message)

        # ---------------------------------------------------------------------
        # 1.) Retrieve relevant documents from Pinecone vector database
        # ---------------------------------------------------------------------
        documents = self.retriever.get_relevant_documents(query=human_message.content)

        # Extract the text from the documents
        document_texts = [doc.page_content for doc in documents]
        leader = textwrap.dedent(
            """You are a helpful assistant.
            You can assume that all of the following is true.
            You should attempt to incorporate these facts
            into your responses:\n\n
        """
        )
        system_message_content = f"{leader} {'. '.join(document_texts)}"
        system_message = SystemMessage(content=system_message_content)
        # ---------------------------------------------------------------------
        # finished with hybrid search setup
        # ---------------------------------------------------------------------

        # 2.) get a response from the chat model
        response = self.cached_chat_request(system_message=system_message, human_message=human_message)

        logging.debug("------------------------------------------------------")
        logging.debug("rag() Retrieval Augmented Generation prompt")
        logging.debug("Diagnostic information:")
        logging.debug("  Retrieved %i related documents from Pinecone", len(documents))
        logging.debug("  System messages contains %i words", len(system_message.content.split()))
        logging.debug("  Prompt: %s", system_message.content)
        logging.debug("------------------------------------------------------")
        return response.content
