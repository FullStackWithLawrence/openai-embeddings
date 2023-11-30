# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""
Sales Support Model (SSM) for the LangChain project.
See: https://python.langchain.com/docs/modules/model_io/llms/llm_caching
     https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
"""

import glob
import os
from typing import ClassVar, List

import pinecone
from langchain import hub
from langchain.cache import InMemoryCache

# prompting and chat
from langchain.chat_models import ChatOpenAI

# document loading
from langchain.document_loaders import PyPDFLoader

# embedding
from langchain.embeddings import OpenAIEmbeddings

# vector database
from langchain.globals import set_llm_cache
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, StrOutputParser, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import Document, RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from pydantic import BaseModel, ConfigDict, Field  # ValidationError

# this project
from models.const import Credentials


###############################################################################
# initializations
###############################################################################
DEFAULT_MODEL_NAME = "text-davinci-003"
pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Credentials.PINECONE_ENVIRONMENT)
set_llm_cache(InMemoryCache())


class SalesSupportModel(BaseModel):
    """Sales Support Model (SSM)."""

    Config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # prompting wrapper
    chat: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(
            api_key=Credentials.OPENAI_API_KEY,
            organization=Credentials.OPENAI_API_ORGANIZATION,
            cache=True,
            max_retries=3,
            model="gpt-3.5-turbo",
            temperature=0.0,
        )
    )

    # embeddings
    texts_splitter_results: List[Document] = Field(None, description="Text splitter results")
    pinecone_search: Pinecone = Field(None, description="Pinecone search")
    openai_embedding: OpenAIEmbeddings = Field(OpenAIEmbeddings())
    query_result: List[float] = Field(None, description="Vector database query result")

    def cached_chat_request(self, system_message: str, human_message: str) -> SystemMessage:
        """Cached chat request."""
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        # pylint: disable=not-callable
        return self.chat(messages)

    def prompt_with_template(self, prompt: PromptTemplate, concept: str, model: str = DEFAULT_MODEL_NAME) -> str:
        """Prompt with template."""
        llm = OpenAI(model=model)
        retval = llm(prompt.format(concept=concept))
        return retval

    def split_text(self, text: str) -> List[Document]:
        """Split text."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
        )
        retval = text_splitter.create_documents([text])
        return retval

    def embed(self, text: str) -> List[float]:
        """Embed."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
        )
        texts_splitter_results = text_splitter.create_documents([text])
        embedding = texts_splitter_results[0].page_content
        # pylint: disable=no-member
        self.openai_embedding.embed_query(embedding)

        self.pinecone_search = Pinecone.from_documents(
            texts_splitter_results,
            embedding=self.openai_embedding,
            index_name=Credentials.PINECONE_INDEX_NAME,
        )

    def rag(self, filepath: str, prompt: str):
        """
        Embed PDF.
        1. Load PDF document text data
        2. Split into pages
        3. Embed each page
        4. Store in Pinecone
        """

        # pylint: disable=unused-variable
        def format_docs(docs):
            """Format docs."""
            return "\n\n".join(doc.page_content for doc in docs)

        for pdf_file in glob.glob(os.path.join(filepath, "*.pdf")):
            loader = PyPDFLoader(file_path=pdf_file)
            docs = loader.load()
            for doc in docs:
                self.embed(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Pinecone.from_documents(documents=splits, embedding=self.openai_embedding)
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.chat
            | StrOutputParser()
        )

        return rag_chain.invoke(prompt)

    def embedded_prompt(self, prompt: str) -> List[Document]:
        """
        Embedded prompt.
        1. Retrieve prompt: Given a user input, relevant splits are retrieved
           from storage using a Retriever.
        2. Generate: A ChatModel / LLM produces an answer using a prompt that includes
           the question and the retrieved data
        """
        result = self.pinecone_search.similarity_search(prompt)
        return result
