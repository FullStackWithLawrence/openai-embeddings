# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import argparse
import os
from dotenv import find_dotenv, load_dotenv

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()

dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
else:    
    raise FileNotFoundError("No .env file found in root directory of repository")

if __name__ == "__main__":
    sql_statement="SELECT clave,nombre, certificacion, disponible, tipo_curso_id, sesiones, pecio_lista, tecnologia_id, subcontratado, pre_requisitos, complejidad_id FROM cursos_habilitados WHERE disponible = 1 OR subcontratado = 1"
      #agregar la clave del curso   
    # parser = argparse.ArgumentParser(description="RAG example")
    # parser.add_argument("sql", type=str, help="A valid SQL statement")
    # args = parser.parse_args()

    hsr.load_sql(sql=sql_statement)