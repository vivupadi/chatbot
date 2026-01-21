import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict
import json
#from docx import Document as DocxDocument
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngestion:
    def __init__(self):
        self.documents = []

    def download_cv_from_blob(self, CONNECT_STR, CONTAINER_NAME):
        """Download CV PDF from blob"""
        print("Downloading CV from blob...")
        blob_service = BlobServiceClient.from_connection_string(CONNECT_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)
        
        cv_path = Path(TEMP_DIR) / "cv.pdf"
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        
        #with open(cv_path, 'wb') as f:
        #    f.write(container.get_blob_client("cv/Vivek_Padayattil_CV.pdf").download_blob().readall())
        
        #print("✓ Downloaded CV")
        return cv_path

    
    def load_pdf(self, file_path):  #Extract text from pdf
        try:
            logger.info(f"PDF Loading: {file_path}")
            reader = PdfReader(file_path)

            documents = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text.strip():
                    documents.append({
                        "content": text,
                        "metadata": {
                            "source": file_path,
                            "page": page_num + 1,
                            "type": "pdf"
                        }
                    })
            
            logger.info(f"Loaded pdf file and extracted {len(documents)} pages from PDF")
            self.documents.extend(documents)
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

    def upload_json_to_blob(self, json_path):
        """Upload entire json folder to blob"""
        print("Uploading json to blob...")
        blob_service = BlobServiceClient.from_connection_string(CONNECT_STR)
        container = blob_service.get_container_client(CONTAINER_NAME)
        
        # Upload all json files
        for file in Path(json_path).rglob("*"):
            if file.is_file():
                blob_name = f"json/{file.relative_to(json_path)}"
                with open(file, 'rb') as data:
                    container.get_blob_client(blob_name).upload_blob(data, overwrite=True)
        
        print("✓ Uploaded json to blob")


#Usage
if __name__ == "__main__":
    ingestion = DocumentIngestion()

    #Load CV
    filepath = ingestion.download_cv_path()
    ingestion.load_pdf(filepath)

    #scrape website
    ingestion.scrape_url('https://www.vivekpadayattil.com/')


    #save documents
    ingestion.save_documents("D:\\Chat\\data\\processed\\documents.json")
