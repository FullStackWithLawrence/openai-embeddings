# -*- coding: utf-8 -*-
"""Sales Support Model (hsr) Retrieval Augmented Generation (RAG)"""
import argparse
import os
import requests
import dropbox
from dotenv import find_dotenv, load_dotenv
from bs4 import BeautifulSoup
import pdfplumber
import io

from models.hybrid_search_retreiver import HybridSearchRetriever


hsr = HybridSearchRetriever()


dotenv_path = find_dotenv()
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
    DROPBOX_ACCESS_TOKEN=os.environ["DROPBOX_ACCESS_TOKEN"]
    DROPBOX_WEB_URL=os.environ["DROPBOX_WEB_URL"]
else:
    raise FileNotFoundError("No .env file found in root directory of repository")

    
def load_data_from_dropbox(root_folder_url):
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    #Get PDFs links 
    response=requests.get(root_folder_url)
    soup=BeautifulSoup(response.content,'html.parser')
    pdf_file_paths=[file.get('href') for file in soup.find_all(name='a',attrs={'class': 'text-link'}) if file.get('href').endswith('.pdf')]

    #Process each route
    for pdf_path in pdf_file_paths:
        try:
            #descargar el PDF usando la API de dropbox
            metadata, response=dbx.files_download(path=pdf_path)
            pdf_bytes=response.content

            #procesar el pdf
            pdf=pdfplumber.open(io.BytesIO(pdf_bytes))
            text_data=[]
            for page in pdf.pages:
                text=page.extract_text()
                text_data.append(text)
            #Crear embeddings
            embeddings=[]
            for text in text_data:
                embedding=self.pinecone.openai_embeddings.embed_documents(text)
                embeddings.append(embedding)
            #Añadir embeddings al indice
                self.pinecone.vector_store.add_documents(documents=embeddings)
        
        except dropbox.exceptions.API.Error as e:
            print("Error al descargar el PDF de Dropbox", e)
        except Exception as a:
            print("Error al procesar el PDF", a)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG example")
    parser.add_argument("root_folder_url", type=str, help="The URL of the root Dropbox folder")
    args = parser.parse_args()

    
    hsr.load_data_from_dropbox(root_folder_url=args.root_folder_url)
