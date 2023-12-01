# -*- coding: utf-8 -*-
"""Sales Support Model (SSM) for the LangChain project."""
import argparse

from ..prompt_templates import NetecPromptTemplates
from ..ssm import HybridSearchRetriever


ssm = HybridSearchRetriever()
templates = NetecPromptTemplates()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSM Oracle examples")
    parser.add_argument("concept", type=str, help="An Oracle certification exam prep")
    args = parser.parse_args()

    prompt = templates.oracle_training_services
    result = ssm.prompt_with_template(prompt=prompt, concept=args.concept)
    print(result)
