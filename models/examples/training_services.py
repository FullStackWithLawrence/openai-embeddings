# -*- coding: utf-8 -*-
"""Sales Support Model (SSM) for the LangChain project."""
import argparse

from ..ssm import SalesSupportModel


ssm = SalesSupportModel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSM examples")
    parser.add_argument("concept", type=str, help="A kind of training that Netec provides.")
    args = parser.parse_args()

    result = ssm.prompt_with_template(concept=args.concept)
    print(result)
