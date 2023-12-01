# -*- coding: utf-8 -*-
"""Sales Support Model (SSM)"""
import argparse

from models.hybrid_search_retreiver import HybridSearchRetriever


ssm = HybridSearchRetriever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSM examples")
    parser.add_argument("system_prompt", type=str, help="A system prompt to send to the model.")
    parser.add_argument("human_prompt", type=str, help="A human prompt to send to the model.")
    args = parser.parse_args()

    result = ssm.cached_chat_request(args.system_prompt, args.human_prompt)
    print(result.content, end="\n")
