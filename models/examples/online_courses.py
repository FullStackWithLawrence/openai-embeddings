# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) for the LangChain project."""
import argparse

from models.hybrid_search_retreiver import HybridSearchRetriever
from models.prompt_templates import UofPennPromptTemplates


hsr = HybridSearchRetriever()
templates = UofPennPromptTemplates()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hsr University of Pennsylvania examples")
    parser.add_argument("concept", type=str, help="A subject to study: accounting, finance, etc.")
    args = parser.parse_args()

    prompt = templates.online_courses
    result = hsr.prompt_with_template(prompt=prompt, concept=args.concept)
    print(result)
