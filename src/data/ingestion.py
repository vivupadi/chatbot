import requests
from bs4 import BeautifulSoup
import PyPDF2
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict
import json
import io
#from docx import Document as DocxDocument
import logging

import os
from dotenv import load_dotenv
load_dotenv()

from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONNECT_STR = os.getenv("CONNECTION_STRING") 
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

class DocumentIngestion:
    def __init__(self):
        self.documents = []
        self.blob_service = BlobServiceClient.from_connection_string(CONNECT_STR)
        self.container = self.blob_service.get_container_client(CONTAINER_NAME)

    
    def load_pdf(self):  #Extract text from pdf
        try:
            blob_path = "raw/Vivek_Padayattil_CV__MLops.pdf"
            logger.info(f"PDF Loading: {blob_path}")
            pdf_bytes = self.container.get_blob_client(blob_path).download_blob().readall()   #gets pdf file bytes format
            
            pdf_file = io.BytesIO(pdf_bytes)
    
            reader = PyPDF2.PdfReader(pdf_file)

            documents = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text.strip():
                    documents.append({
                        "content": text,
                        "metadata": {
                            "source": blob_path,
                            "page": page_num + 1,
                            "type": "pdf"
                        }
                    })
            
            logger.info(f"Loaded pdf file and extracted {len(documents)} pages from PDF")
            self.documents.extend(documents)
            print(documents)
            return documents
        except Exception as e:
            logger.error(f" Error Loading pdf: {e}")
            return []


    def scrape_url(self, url: str, max_pages: int = 5):
        try:
            logger.info(f"Scrapping website: {url}")
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            #REmove script and style
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text()

            #Cleaning the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            document = {
                "content": text,
                "metadata": {
                    "source": url,
                    "type": "website"
                }
            }

            logger.info(f"Scrapped {len(text)} characters from website")
            self.documents.append(document)
            return [document]
        
        except Exception as e:
            logger.error(f" Error scrapping website: {e}")
            return []
        
    def save_documents(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.documents)} documents to {output_path}")

    def upload_json_to_blob(self):
        """Upload entire json folder to blob"""
        print("Uploading json to blob...")
        
        # Upload all json files
        json_str = json.dumps(self.documents, indent=2, ensure_ascii=False)

        blob_name = "json/documents.json"
        
        self.container.get_blob_client(blob_name).upload_blob(json_str, overwrite=True)
        
        print("✓ Uploaded json to blob")


#Usage
if __name__ == "__main__":
    ingestion = DocumentIngestion()

    #Load CV
    ingestion.load_pdf()


    #scrape website
    ingestion.scrape_url('https://www.vivekpadayattil.com/')


    ingestion.upload_json_to_blob()    # save json to azure blob storage


    #save documents
    ingestion.save_documents("D:\\Chat\\data\\processed\\documents.json")   #save json loaclly 
