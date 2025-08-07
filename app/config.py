# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """
    Application configuration class for environment variables and constants.
    """
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "your-default-api-key")
    TEMP_FOLDER: str = "temp_files"
    OUTPUT_FOLDER: str = "outputs"

    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.TEMP_FOLDER, exist_ok=True)
        os.makedirs(cls.OUTPUT_FOLDER, exist_ok=True)

# Ensure required folders exist
AppConfig.ensure_directories()
