# -*- coding: utf-8 -*-
# pylint: disable=E0611,E1101
"""A class to manage the lifecycle of Pinecone vector database indexes."""

# document loading
import glob

# general purpose imports
import json
import logging
import os
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# pinecone integration
from pinecone import AwsRegion, CloudProvider, Pinecone, ServerlessSpec, VectorType
from pinecone.core.openapi.db_data.models import (
    IndexDescription as PineconeIndexDescription,
)
from pinecone.db_control.models import IndexList
from pinecone.db_data import Index
from pinecone.exceptions import PineconeApiException
from pydantic import SecretStr

# this project
from models.conf import settings


logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.INFO)


class PineconeIndex:
    """Pinecone helper class."""

    _pinecone = None
    _index: Optional[Index] = None
    _index_name: Optional[str] = None
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    _openai_embeddings: Optional[OpenAIEmbeddings] = None
    _vector_store: Optional[PineconeVectorStore] = None

    def __init__(self, index_name: Optional[str] = None):
        self.init()
        self.index_name = index_name or settings.pinecone_index_name
        logging.debug("PineconeIndex initialized with index_name: %s", self.index_name)
        logging.debug(self.index_stats)

    @property
    def index_name(self) -> Optional[str]:
        """index name."""
        return self._index_name

    @index_name.setter
    def index_name(self, value: str) -> None:
        """Set index name."""
        if self._index_name != value:
            self.init()
            self._index_name = value
            self.init_index()

    @property
    def index(self) -> Optional[Index]:
        """pinecone.Index lazy read-only property."""
        if self._index is None:
            self.init_index()
            if isinstance(self.pinecone, Pinecone) and isinstance(self.index_name, str):
                self._index = self.pinecone.Index(name=self.index_name)

        return self._index

    @property
    def index_stats(self) -> str:
        """index stats."""
        if self.index is not None:
            retval: PineconeIndexDescription = self.index.describe_index_stats()
            return json.dumps(retval.to_dict(), indent=4)
        return "Index not initialized."

    @property
    def initialized(self) -> bool:
        """initialized read-only property."""
        if isinstance(self.pinecone, Pinecone) and isinstance(self.index_name, str):
            indexes = self.pinecone.list_indexes()
            return self.index_name in indexes.names()
        return False

    @property
    def vector_store(self) -> PineconeVectorStore:
        """Pinecone lazy read-only property."""
        if self._vector_store is None:
            if not self.initialized:
                self.init_index()
            self._vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.openai_embeddings,
                text_key=settings.pinecone_vectorstore_text_key,
            )
        return self._vector_store

    @property
    def openai_embeddings(self) -> OpenAIEmbeddings:
        """OpenAIEmbeddings lazy read-only property."""
        if self._openai_embeddings is None:
            # pylint: disable=no-member
            self._openai_embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                organization=settings.openai_api_organization,
            )
        return self._openai_embeddings

    @property
    def pinecone(self) -> Optional[Pinecone]:
        """Pinecone lazy read-only property."""
        if self._pinecone is None:
            print("Initializing Pinecone...")
            if isinstance(settings.pinecone_api_key, SecretStr):
                api_key = settings.pinecone_api_key.get_secret_value()
                print(f"API Key: {api_key[:12]}****------")
                self._pinecone = Pinecone(api_key=api_key)
        return self._pinecone

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """lazy read-only property."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter()
        return self._text_splitter

    def init_index(self):
        """Verify that an index named self.index_name exists in Pinecone. If not, create it."""
        if isinstance(self.pinecone, Pinecone):
            indexes: IndexList = self.pinecone.list_indexes()
            if self.index_name not in indexes.names():
                logging.debug("Index does not exist.")
                self.create()

    # pylint: disable=no-member
    def init(self):
        """Initialize Pinecone."""

        self._index = None
        self._index_name = None
        self._text_splitter = None
        self._openai_embeddings = None
        self._vector_store = None

    def delete(self):
        """Delete index."""
        if not self.initialized:
            logging.debug("Index does not exist. Nothing to delete.")
            return
        if isinstance(self.pinecone, Pinecone) and isinstance(self.index_name, str):
            print("Deleting index...")
            self.pinecone.delete_index(self.index_name)

    def create(self):
        """Create index."""
        print("Creating index. This may take a few minutes...")
        serverless_spec = ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1,
        )
        try:
            if isinstance(self.pinecone, Pinecone) and isinstance(self.index_name, str):
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=settings.pinecone_dimensions,
                    metric=settings.pinecone_metric,
                    spec=serverless_spec,
                    vector_type=VectorType.DENSE,
                )
                print("Index created.")
        except PineconeApiException:
            pass

    def initialize(self):
        """Initialize index."""
        self.delete()
        self.create()

    def pdf_loader(self, filepath: str):
        """
        Embed PDF.
        1. Load PDF document text data
        2. Split into pages
        3. Embed each page
        4. Store in Pinecone

        Note: it's important to make sure that the "context" field that holds the document text
        in the metadata is not indexed. Currently you need to specify explicitly the fields you
        do want to index. For more information checkout
        https://docs.pinecone.io/docs/manage-indexes#selective-metadata-indexing
        """
        self.initialize()

        pdf_files = glob.glob(os.path.join(filepath, "*.pdf"))
        i = 0
        for pdf_file in pdf_files:
            i += 1
            j = len(pdf_files)
            print(f"Loading PDF {i} of {j}: {pdf_file}")
            loader = PyPDFLoader(file_path=pdf_file)
            docs = loader.load()
            k = 0
            for doc in docs:
                k += 1
                print(k * "-", end="\r")
                documents = self.text_splitter.create_documents([doc.page_content])
                document_texts = [doc.page_content for doc in documents]
                embeddings = self.openai_embeddings.embed_documents(document_texts)
                self.vector_store.add_documents(documents=documents, embeddings=embeddings)

        print("Finished loading PDFs. \n" + self.index_stats)
