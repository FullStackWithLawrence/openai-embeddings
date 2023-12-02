# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) for the LangChain project."""
import argparse

from models.hybrid_search_retreiver import HybridSearchRetriever
from models.prompt_templates import NetecPromptTemplates


hsr = HybridSearchRetriever()
templates = NetecPromptTemplates()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hsr Oracle examples")
    parser.add_argument("concept", type=str, help="An Oracle certification exam prep")
    args = parser.parse_args()

    prompt = templates.oracle_training_services
    result = hsr.prompt_with_template(prompt=prompt, concept=args.concept)
    print(result)
