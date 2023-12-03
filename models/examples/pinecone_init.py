# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""

from models.pinecone import PineconeIndex


pinecone = PineconeIndex()

if __name__ == "__main__":
    pinecone.initialize()
    print("Pinecone index initialized. name: ", pinecone.index_name)
    print(pinecone.index.describe_index_stats())
