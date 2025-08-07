from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from app.services.invoice_processor import InvoiceProcessor
from app.services.gemini_invoice_extractor import GeminiInvoiceExtractor
from dotenv import load_dotenv
load_dotenv()
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gemma2-9b-it", groq_api_key = os.getenv('GROQ_API_KEY'))

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/extract-ocr")
async def extract_ocr(files: List[UploadFile] = File(...)):
    """
    For each uploaded PDF:
    - Save the file
    - Run OCR extraction (InvoiceProcessor)
    - Run Invoice mapping + save output (GeminiInvoiceExtractor)

    Returns:
        JSON: Summary with output files.
    """
    try:
        result_summary = []

        for file in files:
            file_name =  file.filename

            # Save the uploaded PDF file
            pdf_bytes = await file.read()

            # Step 1: OCR
            processor = InvoiceProcessor()
            page_texts = processor.extract_text_from_pdf_bytes(pdf_bytes,filename=file_name)

            # Step 2: Invoice Mapping
            extractor = GeminiInvoiceExtractor(llm=llm)
            output_files = extractor.ocr_to_yaml_text(page_texts)

            result_summary.append({
                "pdf_file": file.filename,
                "mapped_output_files": output_files  # This should be list of saved txt/yaml files
            })

        return JSONResponse(content={
            "message": "OCR and invoice mapping completed for all files.",
            "results": result_summary
        })

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        traceback.print_exc()  # <== Add this line
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
