# -*- coding: utf-8 -*-
"""Sales Support Model (SSM) for the LangChain project."""
import argparse

from models.hybrid_search_retreiver import HybridSearchRetriever
from models.prompt_templates import NetecPromptTemplates


ssm = HybridSearchRetriever()
templates = NetecPromptTemplates()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSM examples")
    parser.add_argument("concept", type=str, help="A kind of training that Netec provides.")
    args = parser.parse_args()

    prompt = templates.training_services
    result = ssm.prompt_with_template(prompt=prompt, concept=args.concept)
    print(result)