# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import argparse

from langchain.schema import HumanMessage

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG example")
    parser.add_argument("prompt", type=str, help="A question about the PDF contents")
    args = parser.parse_args()

    human_message = HumanMessage(content=args.prompt)
    result = hsr.rag(human_message=human_message)
    print(result)
