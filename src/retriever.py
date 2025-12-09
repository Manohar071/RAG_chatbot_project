"""
DELIVERABLE 2: RAG Pipeline - Retrieval System
Semantic search and context retrieval from ChromaDB
"""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD
)


class RAGRetriever:
    """Handles semantic search and document retrieval"""
    
    def __init__(self):
        """Initialize the retriever with ChromaDB and embedding model"""
        # Connect to existing ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        except Exception:
            self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Get collection
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception:
            print("⚠️  Collection not found. Please process documents first.")
            self.collection = None
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        print("✅ Retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filter_metadata: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        if not self.collection:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved_chunks = []
        
        if results["documents"] and results["documents"][0]:
            for idx in range(len(results["documents"][0])):
                # Convert distance to similarity score (lower distance = higher similarity)
                distance = results["distances"][0][idx]
                similarity = 1 / (1 + distance)  # Normalize to 0-1 range
                
                # Filter by similarity threshold
                if similarity >= SIMILARITY_THRESHOLD:
                    chunk = {
                        "text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "similarity": round(similarity, 3),
                        "distance": round(distance, 3)
                    }
                    retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def retrieve_with_scores(self, query: str, top_k: int = TOP_K_RESULTS) -> tuple:
        """
        Retrieve chunks and return both chunks and average similarity score
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (chunks, average_similarity)
        """
        chunks = self.retrieve(query, top_k)
        
        if not chunks:
            return [], 0.0
        
        avg_similarity = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
        
        return chunks, round(avg_similarity, 3)
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            text = chunk["text"]
            similarity = chunk["similarity"]
            
            context_part = f"[Source {idx}: {source} (Relevance: {similarity})]\n{text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def get_unique_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source documents from chunks"""
        sources = set()
        for chunk in chunks:
            source = chunk["metadata"].get("source", "Unknown")
            sources.add(source)
        return sorted(list(sources))
    
    def search_by_source(self, source_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific source document
        
        Args:
            source_name: Name of the source document
            limit: Maximum number of chunks to retrieve
            
        Returns:
            List of chunks from the source
        """
        if not self.collection:
            return []
        
        results = self.collection.get(
            where={"source": source_name},
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results["documents"]:
            for idx in range(len(results["documents"])):
                chunk = {
                    "text": results["documents"][idx],
                    "metadata": results["metadatas"][idx]
                }
                chunks.append(chunk)
        
        return chunks
    
    def get_all_sources(self) -> List[str]:
        """Get list of all source documents in the collection"""
        if not self.collection:
            return []
        
        # Get all documents
        results = self.collection.get(include=["metadatas"])
        
        # Extract unique sources
        sources = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                source = metadata.get("source", "Unknown")
                sources.add(source)
        
        return sorted(list(sources))
    
    def check_collection_status(self) -> Dict[str, Any]:
        """Check if collection exists and has documents"""
        if not self.collection:
            return {
                "status": "not_initialized",
                "document_count": 0,
                "message": "Collection not found. Please process documents first."
            }
        
        try:
            count = self.collection.count()
        except (StopIteration, Exception) as e:
            # Collection may have been deleted/reset
            return {
                "status": "error",
                "document_count": 0,
                "message": "Collection needs reinitialization. Please refresh the page."
            }
        
        return {
            "status": "ready" if count > 0 else "empty",
            "document_count": count,
            "message": f"Collection has {count} chunks" if count > 0 else "No documents in collection"
        }


if __name__ == "__main__":
    # Test the retriever
    retriever = RAGRetriever()
    
    # Check status
    status = retriever.check_collection_status()
    print(f"Collection Status: {status}")
    
    # Test query
    if status["document_count"] > 0:
        query = "What is machine learning?"
        print(f"\nTest Query: {query}")
        
        chunks, avg_sim = retriever.retrieve_with_scores(query)
        
        print(f"\nRetrieved {len(chunks)} chunks (Avg Similarity: {avg_sim})")
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {idx} ---")
            print(f"Source: {chunk['metadata']['source']}")
            print(f"Similarity: {chunk['similarity']}")
            print(f"Text: {chunk['text'][:200]}...")
