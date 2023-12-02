# -*- coding: utf-8 -*-
"""Pinecone helper functions."""

# document loading
import glob

# general purpose imports
import logging
import os

# pinecone integration
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import Document
from langchain.vectorstores.pinecone import Pinecone as LCPinecone

# this project
from models.const import Config, Credentials


# pylint: disable=too-few-public-methods
class TextSplitter:
    """
    Custom text splitter that adds metadata to the Document object
    which is required by PineconeHybridSearchRetriever.
    """

    def create_documents(self, texts):
        """Create documents"""
        documents = []
        for text in texts:
            # Create a Document object with the text and metadata
            document = Document(page_content=text, metadata={"context": text})
            documents.append(document)
        return documents


class PineConeIndex:
    """Pinecone helper class."""

    _index: pinecone.Index = None
    _index_name: str = None
    _text_splitter: TextSplitter = None
    _openai_embeddings: OpenAIEmbeddings = None
    _vector_store: LCPinecone = None

    def __init__(self, index_name: str = None):
        self._index_name = index_name or Config.PINECONE_INDEX_NAME
        self.init()

    @property
    def vector_store(self) -> LCPinecone:
        """Pinecone lazy read-only property."""
        if self._vector_store is None:
            self._vector_store = LCPinecone(
                index=self.index,
                embedding=self.openai_embeddings,
                text_key=Config.PINECONE_VECTORSTORE_TEXT_KEY,
            )
        return self._vector_store

    @property
    def openai_embeddings(self) -> OpenAIEmbeddings:
        """OpenAIEmbeddings lazy read-only property."""
        if self._openai_embeddings is None:
            self._openai_embeddings = OpenAIEmbeddings(
                api_key=Credentials.OPENAI_API_KEY, organization=Credentials.OPENAI_API_ORGANIZATION
            )
        return self._openai_embeddings

    @property
    def text_splitter(self) -> TextSplitter:
        """TextSplitter lazy read-only property."""
        if self._text_splitter is None:
            self._text_splitter = TextSplitter()
        return self._text_splitter

    @property
    def index_name(self) -> str:
        """index name."""
        return self._index_name

    @index_name.setter
    def index_name(self, value: str) -> None:
        """Set index name."""
        if self._index_name != value:
            self._index_name = value
            self.initialize()

    @property
    def index(self) -> pinecone.Index:
        """pinecone.Index lazy read-only property."""
        if self._index is None:
            try:
                self._index = pinecone.Index(index_name=self.index_name)
            except pinecone.exceptions.PineconeException:
                # index does not exist, so create it.
                self.create()
                self._index = pinecone.Index(index_name=self.index_name)
        return self._index

    def init(self):
        """Initialize Pinecone."""
        pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)

    def delete(self):
        """Delete index."""
        try:
            logging.info("Deleting index...")
            pinecone.delete_index(self.index_name)
        except pinecone.exceptions.PineconeException:
            logging.info("Index does not exist. Continuing...")

    def create(self):
        """Create index."""
        metadata_config = {
            "indexed": [Config.PINECONE_VECTORSTORE_TEXT_KEY, "lc_type"],
            "context": ["lc_text"],
        }
        logging.info("Creating index. This may take a few minutes...")

        pinecone.create_index(
            self.index_name,
            dimension=Config.PINECONE_DIMENSIONS,
            metric=Config.PINECONE_METRIC,
            metadata_config=metadata_config,
        )

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
            logging.info("Loading PDF %s of %s: %s", i, j, pdf_file)
            loader = PyPDFLoader(file_path=pdf_file)
            docs = loader.load()
            k = 0
            for doc in docs:
                k += 1
                logging.info(k * "-", end="\r")
                documents = self.text_splitter.create_documents([doc.page_content])
                document_texts = [doc.page_content for doc in documents]
                embeddings = self.openai_embeddings.embed_documents(document_texts)
                self.vector_store.add_documents(documents=documents, embeddings=embeddings)

        logging.info("Finished loading PDFs")
