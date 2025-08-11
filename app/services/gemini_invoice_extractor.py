import os
import json
import yaml
import logging
from typing import List, Dict
from langchain.schema import HumanMessage
from app.models.invoice_schema import  PageTextData, ElaboratedPageTextData
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class GeminiInvoiceExtractor:
    """
    A class to handle invoice page elaboration, mapping, and saving to YAML format.
    """

    def __init__(self, llm, output_path: str = './output/'):
        """
        Initialize the processor.

        :param llm: Language model used for text elaboration and invoice mapping.
        :param output_path: Directory path to save output files.
        """
        self.llm = llm
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def generate_elaborated_texts(self, pages: List[PageTextData]) -> List[ElaboratedPageTextData]:
        """
        Generate elaborated descriptions for each OCR page using LLM.

        :param pages: List of PageTextData.
        :return: List of ElaboratedPageTextData.
        """
        elaborated_pages = []

        for page in pages:
            try:
                prompt = (
                    "You are an expert in reading semi-structured invoices.\n"
                    "Below is the raw OCR text from one page of a PDF invoice.\n"
                    "Write a clear and complete elaboration of what this invoice page contains in natural language.\n\n"
                    f"Page {page.page_number}:\n{page.text}"
                )
                response = self.llm.invoke([HumanMessage(content=prompt)])

                elaborated = ElaboratedPageTextData(
                    page_number=page.page_number,
                    filename=page.filename,
                    text=page.text,
                    elaborated_text=response.content.strip()
                )
                elaborated_pages.append(elaborated)
                logging.info(f"Elaborated page {page.page_number}")
            except Exception as e:
                logging.error(f"Error elaborating page {page.page_number}: {e}")

        return elaborated_pages

    def map_invoice_pages(self, elaborated_pages: List[ElaboratedPageTextData]) -> Dict[str, List[int]]:
        """
        Map each invoice number to the list of pages that belong to it.

        :param elaborated_pages: List of elaborated page data.
        :return: Dictionary mapping invoice number to list of page numbers.
        """
        # Prepare LLM input by formatting all pages
        formatted = "\n\n".join([f"Page {i+1}:\n{txt}" for i, txt in enumerate(elaborated_pages)])

        system_prompt = ("""
            You are a document analysis assistant.
            You will be given a list of pages (with their page numbers) that may contain one or more invoice documents.\n"
            Your task is to identify which pages belong to the same invoice by recognizing invoice numbers and group them.\n\n"
            Return ONLY a valid JSON in the following format.
            like "{"invoice number\": [page numbers in int under invoice number]}
            for E.g.,
            "{"INV-003": [2]} or
            "{"INV-005": [4,5]} or 
            "{"344256": [1,2,3,4]}"
            json should contain have invoice number as key and list of page number belonging to that invoice  
        """)

        user_prompt = (
        "Here are the elaborated texts for all pages:\n\n"
        f"{formatted}\n\n"
        "Now identify and group the pages by invoice number."
        )

        response = self.llm.invoke([
        HumanMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
        ])
        # Parse and return the JSON result
        try:
            raw_content = response.content.strip()
            clean_response = self.clean_llm_json(raw_content)
            invoice_map = json.loads(clean_response)

            return invoice_map
        except json.JSONDecodeError:
            logging.error("LLM response was not valid JSON:\n", response.content)
            return {}

        except Exception as e:
            logging.error(f"Error mapping invoice pages: {e}")
            return {}

    def clean_llm_json(self, text: str) -> str:
        """
        Cleans the LLM response to extract only the JSON content.

        :param text: Raw LLM output, possibly wrapped in ```json.
        :return: Cleaned JSON string.
        """
        try:
            if text.startswith("```json"):
                text = text.replace("```json", "").strip()
            if text.startswith("```"):
                text = text.replace("```", "").strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            return text
        except Exception as e:
            logging.warning(f"Could not clean LLM JSON properly: {e}")
            return text


    def extract_invoice_table_from_dict(self,invoices_dict: dict) -> list[dict]:
        """
        Extracts structured invoice data from a dictionary of {invoice_id: invoice_text}.

        This method:
        - Iterates over each invoice's full text (already merged from multiple pages)
        - Sends it to the LLM to extract all possible fields as JSON
        - Adds the `Invoice_ID` to each extracted JSON

        Parameters
        ----------
        invoices_dict : dict
            Dictionary where:
                - Key   = Invoice ID (string)
                - Value = Full invoice text (string)

        Returns
        -------
        list[dict]
            A list of extracted invoice dictionaries, each containing:
                - 'Invoice_ID': The invoice ID from the key
                - All extracted fields from the LLM
            Returns an empty list if no valid data is extracted.

        Exceptions
        ----------
        Logs errors but does not raise exceptions.
        """
        results = []

        try:
            if not invoices_dict or not isinstance(invoices_dict, dict):
                logging.warning("⚠ No valid invoice dictionary provided.")
                return []

            for invoice_id, invoice_text in invoices_dict.items():
                try:
                    prompt = f"""
                    You are an expert in reading invoices.  
                    Below is the full text of one invoice.  
                    Extract **all available fields** such as invoice number, seller name, buyer name, address,
                    item details, amount, tax, total, etc.  
                    Output as **valid JSON** with keys as field names and values as strings.  
                    Do NOT skip any field present in the text.  
                    If a field is missing, do not include it.

                    Invoice Text:
                    {invoice_text}
                    """

                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    clean_json = self.clean_llm_json(response.content)
                    invoice_data = json.loads(clean_json)

                    if isinstance(invoice_data, dict):
                        invoice_data["Invoice_ID"] = invoice_id
                        results.append(invoice_data)
                    else:
                        logging.warning(f"⚠ LLM did not return a valid dict for invoice {invoice_id}")

                except json.JSONDecodeError as e:
                    logging.error(f"❌ JSON parsing failed for invoice {invoice_id}: {e}")
                except Exception as e:
                    logging.error(f"❌ Error processing invoice {invoice_id}: {e}")

            return results

        except Exception as e:
            logging.exception(f"Unexpected error in extract_invoice_table_from_dict: {e}")
            return []


    def combine_invoice_text(page_mapping: Dict[str, List[int]],
                             elaborated_text_pages: List[ElaboratedPageTextData]
        ) -> Dict[str, str]:
        """
        Combines elaborated text for each invoice based on a given page mapping.

        Args:
        page_mapping (Dict[str, List[int]]): Dictionary mapping invoice IDs to a list of page numbers.
        Example: {"INV001": [1, 2], "INV002": [3]}

        elaborated_text_pages (List[ElaboratedTextPageData]): List of ElaboratedTextPageData objects
        containing page numbers and text.

        Returns:
            Dict[str, str]: A dictionary mapping invoice IDs to concatenated elaborated text.
            Example: {"INV001": "Full text from page 1 and 2...", "INV002": "Text from page 3"}
        """
        try:
            # Convert list to dict for quick lookup
            page_text_lookup = {page.page_number: page.text for page in elaborated_text_pages}
            logging.info("Page text lookup table created with %d entries.", len(page_text_lookup))

            combined_texts = {}

            for invoice_id, pages in page_mapping.items():
                combined_text = []
                for page_num in pages:
                    text = page_text_lookup.get(page_num)
                    if text:
                        combined_text.append(text)
                    else:
                        logging.warning(
                            "Page number %d for invoice %s not found in provided elaborated text data.",
                            page_num, invoice_id
                        )
                combined_texts[invoice_id] = "\n".join(combined_text).strip()
                logging.info("Combined %d pages for invoice %s.", len(combined_text), invoice_id)

                return combined_texts

        except Exception as e:
            logging.exception("Error occurred while combining invoice texts: %s", e)
            raise
        

    def invoices_to_table(self, invoice_jsons):
        """
        Converts a list of invoice JSON objects into a two-column tabular format.

        This method iterates over each invoice's extracted key-value pairs and
        creates a DataFrame with:
        - Column 1: "Field Name"
        - Column 2: "Value"
        Four empty rows are inserted between invoices for visual separation.

        Parameters
        ----------
        invoice_jsons : list[dict]
            A list of dictionaries, where each dictionary contains key-value pairs
            representing an invoice's extracted fields.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame containing all invoices in the desired format.
            Returns an empty DataFrame if no valid data is provided.

        Exceptions
        ----------
        Logs errors but does not raise exceptions.
        """
        try:
            if not invoice_jsons or not isinstance(invoice_jsons, list):
                logging.warning("⚠ No invoice data provided to convert to table.")
                return pd.DataFrame(columns=["Field Name", "Value"])

            rows = []
            for invoice in invoice_jsons:
                if not isinstance(invoice, dict):
                    logging.warning(f"⚠ Skipping invalid invoice data: {invoice}")
                    continue

                for field, value in invoice.items():
                    rows.append({"Field Name": field, "Value": value})

                # Add 4 blank rows between invoices
                rows.extend([{"Field Name": "", "Value": ""} for _ in range(4)])

            return pd.DataFrame(rows)

        except Exception as e:
            logging.error(f"❌ Error creating invoice table: {e}")
            return pd.DataFrame(columns=["Field Name", "Value"])


    def ocr_to_yaml_text(self, pages: List[PageTextData]) -> None:
        """
        Processes a list of PageTextData objects to:
    1. Generate elaborated text from OCR pages.
    2. Map each invoice to its corresponding page indices.
    3. Save each invoice's content as a YAML-formatted .txt file.

        Args:
        pages (List[PageTextData]): List of OCR-processed page text objects.

        Returns:
        None

        Raises:
        Exception: Logs and raises any unexpected exceptions during the processing.
        """
        try:
            # Step 1: Generate elaborated text from OCR
            elaborative_text = self.generate_elaborated_texts(pages)

            # Step 2: Map invoice numbers to page indices
            mapped_pages = self.map_invoice_pages(elaborative_text)

            # Step 3: Save output in YAML format in .txt file
            self.save_invoices_to_yaml_txt(
                invoice_map = mapped_pages,
                elaborated_pages = elaborative_text
            )
        except Exception as e:
            print(f"❌ Error in ocr_to_yaml_text: {e}")
            raise


    # def save_invoices_to_yaml_txt(
    #     self,
    #     invoice_map: Dict[str, List[int]],
    #     elaborated_pages: List[ElaboratedPageTextData]
    # ):
    #     """
    #     Save mapped invoice pages into separate YAML .txt files by invoice number.

    #     :param invoice_map: Dictionary of invoice number to page numbers.
    #     :param elaborated_pages: List of elaborated page data.
    #     """
    #     try:
    #         for invoice_number, page_indices in invoice_map.items():
    #             combined_text = "\n\n".join([
    #                 elaborated_pages[i - 1].text
    #                 for i in page_indices
    #                 if 0 <= i - 1 < len(elaborated_pages)
    #             ])

    #             yaml_data = {
    #                 "invoice_number": invoice_number,
    #                 "content": combined_text
    #             }

    #             yaml_string = yaml.dump(yaml_data, sort_keys=False, allow_unicode=True)
    #             file_path = os.path.join(self.output_path, f"{invoice_number}.txt")

    #             with open(file_path, "w", encoding="utf-8") as f:
    #                 f.write(yaml_string)

    #             logging.info(f"✅ YAML saved for {invoice_number} at {file_path}")

    #     except Exception as e:
    #         logging.error(f"Failed to save invoice files: {e}")