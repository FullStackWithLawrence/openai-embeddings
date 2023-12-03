# -*- coding: utf-8 -*-
"""Sales Support Model (hsr)"""
import argparse

from langchain.schema import HumanMessage, SystemMessage

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hsr examples")
    parser.add_argument("system_message", type=str, help="A system prompt to send to the model.")
    parser.add_argument("human_message", type=str, help="A human prompt to send to the model.")
    args = parser.parse_args()

    system_message = SystemMessage(content=args.system_message)
    human_message = HumanMessage(content=args.human_message)
    result = hsr.cached_chat_request(system_message=system_message, human_message=human_message)
    print(result.content)
