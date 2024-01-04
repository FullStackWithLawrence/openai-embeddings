# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import argparse
import os

import pyodbc
from dotenv import find_dotenv, load_dotenv

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()

dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
    MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
    MYSQL_HOST = os.environ["MYSQL_HOST"]
    MYSQL_USERNAME = os.environ["MYSQL_USERNAME"]
    MYSQL_PORT = os.environ["MYSQL_PORT"]
else:
    raise FileNotFoundError("No .env file found in root directory of repository")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG example")
    parser.add_argument("sql", type=str, help="A valid SQL statement")
    args = parser.parse_args()

    connstring = (
        "DRIVER={driver_name};"
        "SERVER={MYSQL_HOST};"
        "DATABASE=database_name;"
        "UID={MYSQL_USERNAME};"
        "PWD={MYSQL_PASSWORD};"
    )

    try:
        connection = pyodbc.connect(connstring=connstring)
    except pyodbc.Error as e:
        print(f"An error occurred: {e}")

    hsr.load_sql(sql=args.sql)
