# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import argparse

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG example")
    parser.add_argument("filepath", type=str, help="Location of PDF documents")
    args = parser.parse_args()

    hsr.load(filepath=args.filepath)
