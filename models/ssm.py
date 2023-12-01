# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""
Sales Support Model (SSM) for the LangChain project.
See: https://python.langchain.com/docs/modules/model_io/llms/llm_caching
     https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
"""

import glob
import os
from typing import List  # ClassVar

import pinecone
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
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import Document, RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

# this project
from models.const import Credentials


# from pydantic import BaseModel, ConfigDict, Field


###############################################################################
# initializations
###############################################################################
DEFAULT_MODEL_NAME = "text-davinci-003"
pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Credentials.PINECONE_ENVIRONMENT)
set_llm_cache(InMemoryCache())


class SalesSupportModel:
    """Sales Support Model (SSM)."""

    # prompting wrapper
    chat = ChatOpenAI(
        api_key=Credentials.OPENAI_API_KEY,
        organization=Credentials.OPENAI_API_ORGANIZATION,
        cache=True,
        max_retries=3,
        model="gpt-3.5-turbo",
        temperature=0.0,
    )

    # embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
    )
    openai_embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index(
        Credentials.PINECONE_INDEX_NAME,
        embedding=openai_embedding,
    )

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

    # FIX NOTE: DEPRECATED
    def split_text(self, text: str) -> List[Document]:
        """Split text."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
        )
        retval = text_splitter.create_documents([text])
        return retval

    def load(self, filepath: str):
        """
        Embed PDF.
        1. Load PDF document text data
        2. Split into pages
        3. Embed each page
        4. Store in Pinecone
        """

        pdf_files = glob.glob(os.path.join(filepath, "*.pdf"))
        i = 0
        for pdf_file in pdf_files:
            i += 1
            j = len(pdf_files)
            print(f"Loading PDF {i} of {j}: ", pdf_file)
            loader = PyPDFLoader(file_path=pdf_file)
            docs = loader.load()
            k = 0
            for doc in docs:
                k += 1
                print(k * "-", end="\r")
                texts_splitter_results = self.text_splitter.create_documents([doc.page_content])
                self.pinecone_index.from_existing_index(
                    index_name=Credentials.PINECONE_INDEX_NAME,
                    embedding=self.openai_embedding,
                    text_key=texts_splitter_results,
                )

        print("Finished loading PDFs")

    def rag(self, prompt: str):
        """
        Embedded prompt.
        1. Retrieve prompt: Given a user input, relevant splits are retrieved
           from storage using a Retriever.
        2. Generate: A ChatModel / LLM produces an answer using a prompt that includes
           the question and the retrieved data
        """

        # pylint: disable=unused-variable
        def format_docs(docs):
            """Format docs."""
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = self.pinecone_index.as_retriever()

        # Use the retriever to get relevant documents
        documents = retriever.get_relevant_documents(query=prompt)
        print(f"Retrieved {len(documents)} related documents from Pinecone")

        # Generate a prompt from the retrieved documents
        prompt += " ".join(doc.page_content for doc in documents)
        print(f"Prompt contains {len(prompt.split())} words")
        print("Prompt:", prompt)
        print(doc for doc in documents)

        # Get a response from the GPT-3.5-turbo model
        response = self.cached_chat_request(system_message="You are a helpful assistant.", human_message=prompt)

        return response
