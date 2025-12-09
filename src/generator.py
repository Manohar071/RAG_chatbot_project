"""
DELIVERABLE 2: RAG Pipeline - Response Generation
LLM integration for generating answers with Google Gemini
"""
import os
from typing import List, Dict, Any
import google.generativeai as genai

from .config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_OUTPUT_TOKENS
)


class ResponseGenerator:
    """Generates responses using Google Gemini LLM"""
    
    def __init__(self):
        """Initialize the Gemini model"""
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Google API key not found. "
                "Please set GOOGLE_API_KEY in your .env file"
            )
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            generation_config={
                "temperature": LLM_TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        )
        
        print(f"✅ Gemini model initialized: {LLM_MODEL}")
    
    def create_prompt(
        self,
        query: str,
        context: str,
        include_sources: bool = True
    ) -> str:
        """
        Create a prompt for the LLM with context and instructions
        
        Args:
            query: User question
            context: Retrieved context from documents
            include_sources: Whether to request source citations
            
        Returns:
            Formatted prompt
        """
        if not context:
            prompt = f"""You are a helpful AI assistant. The user asked a question, but no relevant information was found in the knowledge base.

User Question: {query}

Please provide a polite response indicating that you don't have information about this topic in your current knowledge base. Suggest that the user:
1. Rephrase the question
2. Check if relevant documents have been uploaded
3. Ask about a different topic

Response:"""
        else:
            sources_instruction = (
                "\n\nIMPORTANT: At the end of your answer, list the source documents you used (e.g., 'Sources: document1.pdf, document2.txt')."
                if include_sources else ""
            )
            
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. Follow these guidelines:

1. Answer ONLY based on the information in the context below
2. If the context doesn't contain enough information, say so clearly
3. Be concise but comprehensive
4. Use a professional and friendly tone
5. If you're not certain about something, express appropriate uncertainty{sources_instruction}

Context:
{context}

User Question: {query}

Answer:"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response to the user's query
        
        Args:
            query: User question
            context_chunks: Retrieved chunks from the retriever
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Format context
        if context_chunks:
            context_parts = []
            for idx, chunk in enumerate(context_chunks, 1):
                source = chunk["metadata"].get("source", "Unknown")
                text = chunk["text"]
                context_parts.append(f"[Document {idx}: {source}]\n{text}")
            context = "\n\n".join(context_parts)
        else:
            context = ""
        
        # Create prompt
        prompt = self.create_prompt(query, context, include_sources)
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Extract sources
            sources = []
            if context_chunks:
                sources = list(set([
                    chunk["metadata"].get("source", "Unknown")
                    for chunk in context_chunks
                ]))
            
            # Calculate confidence based on similarity scores
            if context_chunks:
                avg_similarity = sum(
                    chunk.get("similarity", 0) for chunk in context_chunks
                ) / len(context_chunks)
                confidence = round(avg_similarity, 2)
            else:
                confidence = 0.0
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "num_chunks_used": len(context_chunks),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error generating response: {error_msg}")
            
            return {
                "answer": f"I apologize, but I encountered an error while generating a response: {error_msg}",
                "sources": [],
                "confidence": 0.0,
                "num_chunks_used": 0,
                "status": "error"
            }
    
    def generate_with_conversation(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate response with conversation history
        
        Args:
            query: Current user question
            context_chunks: Retrieved chunks
            conversation_history: Previous conversation turns
            
        Returns:
            Response dictionary
        """
        # Format conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_parts.append(f"User: {turn['user']}")
                history_parts.append(f"Assistant: {turn['assistant']}")
            history_text = "\n".join(history_parts)
        
        # Add history to context if available
        if history_text:
            history_context = f"\n\nPrevious Conversation:\n{history_text}\n"
        else:
            history_context = ""
        
        # Generate response (similar to generate method)
        return self.generate(query, context_chunks, include_sources=True)
    
    def test_connection(self) -> bool:
        """Test if the Gemini API is working"""
        try:
            response = self.model.generate_content("Say 'hello'")
            return bool(response.text)
        except Exception as e:
            print(f"❌ API connection test failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the generator
    try:
        generator = ResponseGenerator()
        
        # Test connection
        print("\nTesting API connection...")
        if generator.test_connection():
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
        
        # Test generation with sample context
        print("\nTesting response generation...")
        sample_chunks = [
            {
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                "metadata": {"source": "test_document.txt"},
                "similarity": 0.85
            }
        ]
        
        query = "What is machine learning?"
        result = generator.generate(query, sample_chunks)
        
        print(f"\nQuery: {query}")
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']}")
        
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease create a .env file with your GOOGLE_API_KEY")
