# app/models/invoice_schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PageTextData(BaseModel):
    """
    Represents the text content extracted from a single page of a PDF invoice.
    
    """
    page_number: int
    filename: str
    text: str

class ElaboratedPageTextData(BaseModel):
    """
    Represents the detailed data for a single OCR-processed page, including both raw and elaborated text.

    """
    page_number: int
    filename: str
    text: str
    elaborated_text: str