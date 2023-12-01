# -*- coding: utf-8 -*-
"""Sales Support Model (SSM) Retrieval Augmented Generation (RAG)"""
import argparse

from ..ssm import HybridSearchRetriever


ssm = HybridSearchRetriever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG example")
    parser.add_argument("filepath", type=str, help="Location of PDF documents")
    args = parser.parse_args()

    ssm.load(filepath=args.filepath)
