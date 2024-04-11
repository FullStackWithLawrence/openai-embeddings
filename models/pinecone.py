# -*- coding: utf-8 -*-
# pylint: disable=E0611,E1101
"""A class to manage the lifecycle of Pinecone vector database indexes."""

# document loading
import glob

# general purpose imports
import json
import logging
import os

# from langchain.text_splitter import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# pinecone integration
# import pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import PineconeApiException
from pinecone.models import IndexList

# this project
from models.conf import settings


logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.ERROR)


class PineconeIndex:
    """Pinecone helper class."""

    _pinecone = None
    _index: Pinecone.Index = None
    _index_name: str = None
    _text_splitter: RecursiveCharacterTextSplitter = None
    _openai_embeddings: OpenAIEmbeddings = None
    _vector_store: PineconeVectorStore = None

    def __init__(self, index_name: str = None):
        self.init()
        self.index_name = index_name or settings.pinecone_index_name
        logging.debug("PineconeIndex initialized with index_name: %s", self.index_name)
        logging.debug(self.index_stats)

    @property
    def index_name(self) -> str:
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
    def index(self) -> Pinecone.Index:
        """pinecone.Index lazy read-only property."""
        if self._index is None:
            self.init_index()
            self._index = self.pinecone.Index(name=self.index_name)
        return self._index

    @property
    def index_stats(self) -> dict:
        """index stats."""
        retval = self.index.describe_index_stats()
        return json.dumps(retval.to_dict(), indent=4)

    @property
    def initialized(self) -> bool:
        """initialized read-only property."""
        indexes = self.pinecone.list_indexes()
        return self.index_name in indexes.names()

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
                api_key=settings.openai_api_key.get_secret_value(),
                organization=settings.openai_api_organization,
            )
        return self._openai_embeddings

    @property
    def pinecone(self):
        """Pinecone lazy read-only property."""
        if self._pinecone is None:
            self._pinecone = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        return self._pinecone

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """lazy read-only property."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter()
        return self._text_splitter

    def init_index(self):
        """Verify that an index named self.index_name exists in Pinecone. If not, create it."""
        indexes: IndexList = None
        indexes = self.pinecone.list_indexes()
        if self.index_name not in indexes.names():
            logging.debug("Index does not exist.")
            self.create()

    def init(self):
        """Initialize Pinecone."""
        # pylint: disable=no-member

        # pinecone.init(api_key=settings.pinecone_api_key.get_secret_value(), environment=settings.pinecone_environment)
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
        print("Deleting index...")
        self.pinecone.delete_index(self.index_name)

    def create(self):
        """Create index."""
        # deprecated?
        # metadata_config = {
        #     "indexed": [settings.pinecone_vectorstore_text_key, "lc_type"],
        #     "context": ["lc_text"],
        # }
        print("Creating index. This may take a few minutes...")
        serverless_spec = ServerlessSpec(
            cloud="aws",
            region="us-west-2",
        )
        try:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=settings.pinecone_dimensions,
                metric=settings.pinecone_metric,
                spec=serverless_spec,
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
