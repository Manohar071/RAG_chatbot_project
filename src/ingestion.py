"""
DELIVERABLE 1: Data Pipeline & Vector Store
Document ingestion, chunking, and ChromaDB storage
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx

from .config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    SUPPORTED_FORMATS
)


class DocumentIngestion:
    """Handles document loading, processing, and storage in ChromaDB"""
    
    def __init__(self):
        """Initialize the document ingestion pipeline"""
        # Initialize ChromaDB with persistent storage
        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_DIR
            )
        except Exception:
            # If client exists, get the existing one
            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_DIR
            )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Document chunks for RAG chatbot"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print(f"✅ ChromaDB initialized at: {CHROMA_DB_DIR}")
        print(f"✅ Collection: {COLLECTION_NAME}")
        print(f"✅ Embedding model: {EMBEDDING_MODEL}")
    
    def load_document(self, file_path: str) -> str:
        """
        Load text content from a document file
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif extension == ".pdf":
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif extension == ".docx":
                doc = docx.Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def chunk_document(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split document into chunks with metadata
        
        Args:
            text: Document text
            source: Document source/filename
            
        Returns:
            List of chunk dictionaries
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Skip very small chunks
                chunk_dicts.append({
                    "text": chunk,
                    "metadata": {
                        "source": source,
                        "chunk_index": idx,
                        "chunk_size": len(chunk)
                    }
                })
        
        return chunk_dicts
    
    def process_document(self, file_path: str) -> int:
        """
        Process a single document: load, chunk, embed, and store
        
        Args:
            file_path: Path to the document
            
        Returns:
            Number of chunks created
        """
        print(f"📄 Processing: {file_path}")
        
        # Load document
        text = self.load_document(file_path)
        if not text:
            print(f"⚠️  Skipping empty document: {file_path}")
            return 0
        
        # Get filename for metadata
        filename = Path(file_path).name
        
        # Chunk document
        chunks = self.chunk_document(text, filename)
        
        if not chunks:
            print(f"⚠️  No chunks created for: {file_path}")
            return 0
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Create unique IDs
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadata
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks from {filename}")
        return len(chunks)
    
    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Statistics about processed documents
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory_path}")
            return {"total_files": 0, "total_chunks": 0, "failed": 0}
        
        # Find all supported documents
        files = []
        for ext in SUPPORTED_FORMATS:
            files.extend(directory.glob(f"*{ext}"))
        
        if not files:
            print(f"⚠️  No supported documents found in {directory_path}")
            return {"total_files": 0, "total_chunks": 0, "failed": 0}
        
        print(f"\n📚 Found {len(files)} documents to process\n")
        
        # Process each file
        total_chunks = 0
        failed = 0
        
        for file_path in files:
            try:
                chunks = self.process_document(str(file_path))
                total_chunks += chunks
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {str(e)}")
                failed += 1
        
        stats = {
            "total_files": len(files),
            "successful": len(files) - failed,
            "failed": failed,
            "total_chunks": total_chunks
        }
        
        print(f"\n{'='*50}")
        print(f"📊 Processing Complete:")
        print(f"   • Total files: {stats['total_files']}")
        print(f"   • Successful: {stats['successful']}")
        print(f"   • Failed: {stats['failed']}")
        print(f"   • Total chunks: {stats['total_chunks']}")
        print(f"{'='*50}\n")
        
        return stats
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        count = self.collection.count()
        
        return {
            "total_documents": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE
        }
    
    def reset_collection(self):
        """Delete all documents from the collection"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Document chunks for RAG chatbot"}
        )
        print("🗑️  Collection reset successfully")


if __name__ == "__main__":
    # Test the ingestion pipeline
    ingestion = DocumentIngestion()
    
    # Process documents from the data directory
    stats = ingestion.process_directory("./data/documents")
    
    # Show collection stats
    print("\n" + "="*50)
    collection_stats = ingestion.get_collection_stats()
    print("📊 Vector Store Statistics:")
    for key, value in collection_stats.items():
        print(f"   • {key}: {value}")
    print("="*50)
