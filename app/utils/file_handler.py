import os
import logging
from pydantic import BaseModel
from typing import Dict, Union, List


logger = logging.getLogger(__name__)

class YamlStyleTextWriter:
    """
    Writes a dictionary in YAML-style format into a .txt file,
    using the invoice number as the filename.
    """

    def write(self, invoice_data: Union[Dict, BaseModel, List[Union[Dict, BaseModel]]]):
        """
        Write one or more invoice data entries to YAML-style formatted .txt files.

        Args:
            invoice_data (Union[Dict, BaseModel, List[Union[Dict, BaseModel]]]): Parsed invoice data.

        Raises:
            ValueError: If 'invoice_number' is missing in any invoice entry.
        """
        if not isinstance(invoice_data, list):
            invoice_data = [invoice_data]  # Convert single entry to list

        for idx, invoice in enumerate(invoice_data):
            try:
                # Convert BaseModel to dict if needed
                if isinstance(invoice, BaseModel):
                    invoice_dict = invoice.model_dump()
                    print(invoice_dict)
                else:
                    invoice_dict = invoice
                    print(invoice_dict)

                invoice_number = invoice_dict.get("invoice_number")
                if not invoice_number:
                    raise ValueError(f"Missing 'invoice_number' in invoice data at index {idx}")

                file_path = os.path.join(self.output_dir, f"{invoice_number}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    self._write_dict(invoice_dict, f)

                logger.info(f"Saved invoice data to {file_path}")

            except Exception as e:
                logger.error(f"Failed to write invoice at index {idx}: {e}")
                continue 


    def _write_dict(self, data: Dict, f, indent: int = 0):
        """
        Recursively writes dictionary in YAML-style format.

        Args:
            data (Dict): Dictionary to write.
            f: Open file object.
            indent (int): Current indentation level.
        """
        for key, value in data.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                f.write(f"{prefix}{key}:\n")
                self._write_dict(value, f, indent + 1)
            elif isinstance(value, list):
                f.write(f"{prefix}{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        f.write(f"{prefix}-\n")
                        self._write_dict(item, f, indent + 2)
                    else:
                        f.write(f"{prefix}- {item}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")

