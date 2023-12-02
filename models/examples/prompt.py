# -*- coding: utf-8 -*-
"""Sales Support Model (hsr)"""
import argparse

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hsr examples")
    parser.add_argument("system_prompt", type=str, help="A system prompt to send to the model.")
    parser.add_argument("human_prompt", type=str, help="A human prompt to send to the model.")
    args = parser.parse_args()

    result = hsr.cached_chat_request(args.system_prompt, args.human_prompt)
    print(result)
