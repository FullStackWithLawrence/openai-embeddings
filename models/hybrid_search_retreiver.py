# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""
Sales Support Model (SSM) for the LangChain project.
See: https://python.langchain.com/docs/modules/model_io/llms/llm_caching
     https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
     https://python.langchain.com/docs/integrations/retrievers/pinecone_hybrid_search
"""

# document loading
import glob
import os
import textwrap
from typing import List

# pinecone integration
import pinecone
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_llm_cache

# prompting and chat
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

# hybrid search capability
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import Document
from langchain.vectorstores.pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# this project
from models.const import Credentials


###############################################################################
# initializations
###############################################################################
DEFAULT_MODEL_NAME = "text-davinci-003"
pinecone.init(api_key=Credentials.PINECONE_API_KEY, environment=Credentials.PINECONE_ENVIRONMENT)
set_llm_cache(InMemoryCache())


class TextSplitter:
    """
    Custom text splitter that add metadata to the Document object
    which is required by PineconeHybridSearchRetriever.
    """

    # ...

    def create_documents(self, texts):
        """Create documents"""
        documents = []
        for text in texts:
            # Create a Document object with the text and metadata
            document = Document(page_content=text, metadata={"context": text})
            documents.append(document)
        return documents


class HybridSearchRetriever:
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
    openai_embeddings = OpenAIEmbeddings(
        api_key=Credentials.OPENAI_API_KEY, organization=Credentials.OPENAI_API_ORGANIZATION
    )
    pinecone_index = pinecone.Index(index_name=Credentials.PINECONE_INDEX_NAME)
    vector_store = Pinecone(index=pinecone_index, embedding=openai_embeddings, text_key="lc_id")

    text_splitter = TextSplitter()
    bm25_encoder = BM25Encoder().default()

    def cached_chat_request(self, system_message: str, human_message: str) -> SystemMessage:
        """Cached chat request."""
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        # pylint: disable=not-callable
        retval = self.chat(messages).content
        return retval

    def prompt_with_template(self, prompt: PromptTemplate, concept: str, model: str = DEFAULT_MODEL_NAME) -> str:
        """Prompt with template."""
        llm = OpenAI(model=model)
        retval = llm(prompt.format(concept=concept))
        return retval

    def fit_tf_idf_values(self, corpus: List[str]):
        """Fit TF-IDF values.
        1. Fit the BM25 encoder on the corpus
        2. Encode the corpus
        3. Store the encoded corpus in Pinecone
        """
        corpus = ["foo", "bar", "world", "hello"]

        # fit tf-idf values on your corpus
        self.bm25_encoder.fit(corpus)

        # persist the values to a json file
        self.bm25_encoder.dump("bm25_values.json")
        self.bm25_encoder = BM25Encoder().load("bm25_values.json")
        self.bm25_encoder.fit(corpus)

    def load(self, filepath: str):
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
        try:
            print("Deleting index...")
            pinecone.delete_index(Credentials.PINECONE_INDEX_NAME)
        except pinecone.exceptions.PineconeException:
            print("Index does not exist. Continuing...")

        metadata_config = {
            "indexed": ["lc_id", "lc_type"],
            "context": ["lc_text"],
        }
        print("Creating index. This may take a few minutes...")
        pinecone.create_index(
            Credentials.PINECONE_INDEX_NAME, dimension=1536, metric="dotproduct", metadata_config=metadata_config
        )

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
                documents = self.text_splitter.create_documents([doc.page_content])
                document_texts = [doc.page_content for doc in documents]
                embeddings = self.openai_embeddings.embed_documents(document_texts)
                self.vector_store.add_documents(documents=documents, embeddings=embeddings)

        print("Finished loading PDFs")

    def rag(self, prompt: str):
        """
        Embedded prompt.
        1. Retrieve prompt: Given a user input, relevant splits are retrieved
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
        retriever = PineconeHybridSearchRetriever(
            embeddings=self.openai_embeddings, sparse_encoder=self.bm25_encoder, index=self.pinecone_index
        )
        documents = retriever.get_relevant_documents(query=prompt)
        print(f"Retrieved {len(documents)} related documents from Pinecone")

        # Extract the text from the documents
        document_texts = [doc.page_content for doc in documents]
        leader = textwrap.dedent(
            """\
            You can assume that the following is true,
            and you should attempt to incorporate these facts
            in your response:
        """
        )

        # Create a prompt that includes the document texts
        prompt_with_relevant_documents = f"{prompt + leader} {'. '.join(document_texts)}"

        print(f"Prompt contains {len(prompt_with_relevant_documents.split())} words")
        print("Prompt:", prompt_with_relevant_documents)

        # Get a response from the GPT-3.5-turbo model
        response = self.cached_chat_request(
            system_message="You are a helpful assistant.", human_message=prompt_with_relevant_documents
        )

        print("Response:")
        print("------------------------------------------------------")
        return response
