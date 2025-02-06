# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""

import logging

# this project
from models.conf import settings
from models.pinecone import PineconeIndex


logging.basicConfig(level=logging.DEBUG if settings.debug_mode else logging.INFO)
logger = logging.getLogger(__name__)

pinecone = PineconeIndex()

if __name__ == "__main__":
    pinecone.initialize()
    print("Pinecone index initialized. name: ", pinecone.index_name)
    print(pinecone.index.describe_index_stats())
