# app/services/invoice_processor.py

import logging
from pdf2image import convert_from_bytes
import pytesseract
from app.models.invoice_schema import PageTextData
logger = logging.getLogger(__name__)

class InvoiceProcessor:
    """
    Handles OCR-based text extraction from PDF files, either via file path or raw bytes.
    """

    def __init__(self, dpi: int = 300):
        """
        Initialize InvoiceProcessor with DPI for image conversion.

        Args:
            dpi (int): Dots per inch used when converting PDF pages to images.
        """
        self.dpi = dpi


    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> PageTextData:
        """
        Extract text from PDF content received as bytes.

        Args:
            pdf_bytes (bytes): Raw PDF content.

        Returns:
            PageTextData: List of page-wise extracted text.
        """
        try:
            logger.info("Converting PDF bytes to images...")
            images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
            pages = []
            for page_number, image in enumerate(images, start=1):
                logger.info(f"Running OCR on byte-converted page {page_number}...")
                text = pytesseract.image_to_string(image)
                pages.append(PageTextData(page_number=page_number, text=text,filename= filename ))
            logger.info("OCR extraction from bytes completed.")
            return pages
        except Exception as e:
            logger.error(f"Failed to extract text from PDF bytes: {e}")
            raise
