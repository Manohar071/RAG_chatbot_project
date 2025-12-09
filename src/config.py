"""
Configuration settings for the RAG chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ChromaDB Settings
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "document_collection"

# Document Processing Settings
CHUNK_SIZE = 600  # characters
CHUNK_OVERLAP = 60  # 10% overlap
SUPPORTED_FORMATS = [".pdf", ".txt", ".docx"]

# Embedding Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
# Alternative: "all-mpnet-base-v2" for higher quality

# Retrieval Settings
TOP_K_RESULTS = 3  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.5

# LLM Settings
LLM_MODEL = "models/gemini-2.5-flash"
LLM_TEMPERATURE = 0.3  # Lower for more focused answers
MAX_OUTPUT_TOKENS = 1024

# UI Settings
PAGE_TITLE = "RAG Chatbot - Q&A System"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Testing Settings
TEST_QUESTIONS_COUNT = 20
MIN_ACCURACY_THRESHOLD = 0.6  # 60%
