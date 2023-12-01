# -*- coding: utf-8 -*-
"""Sales Support Model (SSM) Retrieval Augmented Generation (RAG)"""
import argparse

from ..ssm import HybridSearchRetriever


ssm = HybridSearchRetriever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG example")
    parser.add_argument("prompt", type=str, help="A question about the PDF contents")
    args = parser.parse_args()

    result = ssm.rag(prompt=args.prompt)
    print(result)
