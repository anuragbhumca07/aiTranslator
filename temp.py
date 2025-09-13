import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Debug: check if key is loaded
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
