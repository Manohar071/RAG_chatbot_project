"""
DELIVERABLE 3: Streamlit User Interface
Interactive web interface for the RAG chatbot
"""
import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict

# Import our modules
from src.ingestion import DocumentIngestion
from src.retriever import RAGRetriever
from src.generator import ResponseGenerator
from src.config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    SUPPORTED_FORMATS,
    TOP_K_RESULTS
)


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "ingestion" not in st.session_state:
        st.session_state.ingestion = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "generator" not in st.session_state:
        st.session_state.generator = None
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False
    
    if "collection_stats" not in st.session_state:
        st.session_state.collection_stats = {}


def initialize_system():
    """Initialize the RAG system components"""
    try:
        with st.spinner("🔧 Initializing RAG system..."):
            # Check for API key
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("❌ Google API key not found! Please set GOOGLE_API_KEY in your .env file")
                return False
            
            # Initialize components
            st.session_state.ingestion = DocumentIngestion()
            st.session_state.retriever = RAGRetriever()
            st.session_state.generator = ResponseGenerator()
            
            # Get collection stats
            st.session_state.collection_stats = st.session_state.ingestion.get_collection_stats()
            
            st.session_state.system_initialized = True
            return True
            
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return False


def save_uploaded_file(uploaded_file, save_dir: str = "./data/documents") -> str:
    """Save uploaded file to the documents directory"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_path = save_path / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def display_chat_message(role: str, content: str, sources: List[str] = None, confidence: float = None):
    """Display a chat message with sources and confidence"""
    with st.chat_message(role):
        st.markdown(content)
        
        # Display sources and confidence for assistant messages
        if role == "assistant" and sources:
            st.markdown("---")
            
            # Confidence score
            if confidence is not None:
                confidence_pct = int(confidence * 100)
                st.progress(confidence, text=f"🎯 Confidence: {confidence_pct}%")
            
            # Sources
            with st.expander("📚 View Sources", expanded=False):
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"{idx}. `{source}`")


def sidebar_content():
    """Render sidebar content"""
    with st.sidebar:
        st.markdown("### 🤖 RAG Chatbot")
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        if st.session_state.system_initialized:
            stats = st.session_state.collection_stats
            
            st.metric("Documents Indexed", stats.get("total_documents", 0))
            st.metric("Embedding Model", stats.get("embedding_model", "N/A"))
            st.metric("Chunk Size", stats.get("chunk_size", "N/A"))
            
            # Collection info
            if st.session_state.retriever:
                retriever_status = st.session_state.retriever.check_collection_status()
                status_color = "🟢" if retriever_status["status"] == "ready" else "🟡"
                st.info(f"{status_color} {retriever_status['message']}")
        else:
            st.warning("⚠️ System not initialized")
        
        st.markdown("---")
        
        # Document Management
        st.markdown("### 📄 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=[ext.replace(".", "") for ext in SUPPORTED_FORMATS],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_files:
            if st.button("📥 Process Uploaded Files", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Process existing documents
        if st.button("🔄 Process All Documents"):
            process_all_documents()
        
        # Reset collection
        if st.button("🗑️ Reset Collection", type="secondary"):
            if st.session_state.ingestion:
                st.session_state.ingestion.reset_collection()
                st.session_state.collection_stats = st.session_state.ingestion.get_collection_stats()
                st.success("✅ Collection reset successfully")
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.markdown("**Retrieval Settings**")
            top_k = st.slider("Number of chunks to retrieve", 1, 10, TOP_K_RESULTS)
            st.session_state.top_k = top_k
            
            st.markdown("**Display Settings**")
            show_confidence = st.checkbox("Show confidence scores", value=True)
            st.session_state.show_confidence = show_confidence
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This RAG chatbot uses:
        - **ChromaDB** for vector storage
        - **Google Gemini** for LLM
        - **Sentence Transformers** for embeddings
        
        Upload documents and ask questions!
        """)


def process_uploaded_files(uploaded_files):
    """Process newly uploaded files"""
    if not st.session_state.system_initialized:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner(f"📥 Processing {len(uploaded_files)} file(s)..."):
        total_chunks = 0
        
        for uploaded_file in uploaded_files:
            try:
                # Save file
                file_path = save_uploaded_file(uploaded_file)
                
                # Process file
                chunks = st.session_state.ingestion.process_document(file_path)
                total_chunks += chunks
                
                st.success(f"✅ Processed {uploaded_file.name}: {chunks} chunks")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Update stats
        st.session_state.collection_stats = st.session_state.ingestion.get_collection_stats()
        st.success(f"🎉 Total chunks added: {total_chunks}")
        st.rerun()


def process_all_documents():
    """Process all documents in the data/documents directory"""
    if not st.session_state.system_initialized:
        st.error("❌ Please initialize the system first")
        return
    
    with st.spinner("📚 Processing all documents..."):
        stats = st.session_state.ingestion.process_directory("./data/documents")
        st.session_state.collection_stats = st.session_state.ingestion.get_collection_stats()
        
        st.success(f"""
        ✅ Processing complete!
        - Files processed: {stats['successful']}/{stats['total_files']}
        - Total chunks: {stats['total_chunks']}
        """)
        st.rerun()


def handle_user_query(query: str):
    """Handle user query and generate response"""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Retrieve relevant chunks
    with st.spinner("🔍 Searching knowledge base..."):
        top_k = st.session_state.get("top_k", TOP_K_RESULTS)
        chunks, avg_similarity = st.session_state.retriever.retrieve_with_scores(query, top_k=top_k)
    
    # Generate response
    with st.spinner("💭 Generating answer..."):
        result = st.session_state.generator.generate(query, chunks)
    
    # Add assistant message to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"]
    })


def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG Chatbot - Q&A System</div>', unsafe_allow_html=True)
    st.markdown("Ask questions and get answers from your document knowledge base!")
    
    # Sidebar
    sidebar_content()
    
    # Initialize system if not already done
    if not st.session_state.system_initialized:
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_system():
                st.success("✅ System initialized successfully!")
                st.rerun()
        
        st.info("""
        👋 Welcome! Please click "Initialize System" to start.
        
        Then:
        1. Upload documents using the sidebar
        2. Click "Process Uploaded Files" or "Process All Documents"
        3. Start asking questions!
        """)
        return
    
    # Check if documents are indexed
    if st.session_state.collection_stats.get("total_documents", 0) == 0:
        st.warning("""
        ⚠️ No documents in the knowledge base yet!
        
        Please upload and process documents using the sidebar before asking questions.
        """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources"),
            confidence=message.get("confidence") if st.session_state.get("show_confidence", True) else None
        )
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query
        handle_user_query(query)
        
        # Display assistant response
        last_message = st.session_state.messages[-1]
        display_chat_message(
            role=last_message["role"],
            content=last_message["content"],
            sources=last_message.get("sources"),
            confidence=last_message.get("confidence") if st.session_state.get("show_confidence", True) else None
        )


if __name__ == "__main__":
    main()
