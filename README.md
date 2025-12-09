# RAG Chatbot - Q&A System with ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot that answers domain-specific questions using ChromaDB vector database and Google's Gemini LLM.

## 🎬 Demo Video

**Watch the 3-minute project demonstration:** [https://www.loom.com/share/3053c3df52fd407a81f2fc17902365b2](https://www.loom.com/share/3053c3df52fd407a81f2fc17902365b2)

## 🎯 Project Overview

This chatbot retrieves information from a custom knowledge base and generates accurate answers with source attribution. It demonstrates practical implementation of:
- Vector databases (ChromaDB)
- Embeddings (sentence-transformers)
- LLM integration (Google Gemini)
- Web interface (Streamlit)

## 📋 Features

- **Document Ingestion**: Support for PDF, TXT, and DOCX files
- **Semantic Search**: ChromaDB vector store with persistent storage
- **RAG Pipeline**: Retrieve relevant context and generate answers
- **Source Attribution**: Each answer includes source documents
- **Chat Interface**: User-friendly Streamlit web UI
- **Conversation History**: Session-based chat memory
- **Performance Metrics**: Testing and evaluation framework

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API Key ([Get it here](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Manohar071/RAG_chatbot_project.git
cd RAG_chatbot_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Google API key
GOOGLE_API_KEY=your_actual_api_key_here
```

4. Add your documents to the `data/documents/` folder (minimum 20 documents)

5. Run the application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
rag_chatbot_working/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .env                    # Your API keys (create this)
├── app.py                  # Main Streamlit application
├── chroma_db/              # ChromaDB vector store (auto-created)
├── data/
│   └── documents/          # Your document collection
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # Document processing
│   ├── retriever.py        # Semantic search
│   └── generator.py        # LLM response generation
├── tests/
│   ├── test_questions.py   # Test suite
│   └── test_results.xlsx   # Evaluation results
└── deployment/
    └── deployment_guide.md # Deployment instructions
```

## 💡 Usage

### Web Interface
1. **Upload Documents**: Use the sidebar to upload PDF, TXT, or DOCX files
2. **Process Documents**: Click "Process Documents" to index them in ChromaDB
3. **Ask Questions**: Type your question in the chat input
4. **View Sources**: Each answer includes source document references

### API Usage (Optional)
```python
from src.retriever import RAGRetriever
from src.generator import ResponseGenerator

# Initialize
retriever = RAGRetriever()
generator = ResponseGenerator()

# Ask a question
query = "What is machine learning?"
context = retriever.retrieve(query, top_k=3)
answer = generator.generate(query, context)
print(answer)
```

## 🧪 Testing

Run the evaluation suite:
```bash
python tests/test_questions.py
```

This will:
- Test 20 predefined questions
- Calculate retrieval accuracy
- Measure response times
- Generate `test_results.xlsx`

## 📊 Performance Metrics

- **Retrieval Accuracy**: 60%+ (target)
- **Response Time**: <5 seconds
- **Vector Store**: 20+ documents indexed
- **Chunk Size**: 500-800 characters with 10% overlap

## 🚢 Deployment

### Option 1: Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `GOOGLE_API_KEY` to Secrets
5. Deploy!

### Option 2: Hugging Face Spaces
See `deployment/deployment_guide.md` for detailed instructions

### Option 3: Local Sharing
Use ngrok or localtunnel for temporary public access

## 🔧 Configuration

Edit `src/config.py` to customize:
- Chunk size and overlap
- Number of retrieved documents
- Embedding model
- LLM parameters

## 📝 License

MIT License - feel free to use for educational purposes

## 🙏 Acknowledgments

- ChromaDB for vector database
- Google for Gemini API
- Streamlit for the web framework
- LangChain for RAG utilities

## 📧 Support

For issues or questions:
- Create an issue on GitHub
- Contact: [your-email]
- Course Slack: #rag-project-help

---

**Note**: This is a capstone project demonstrating RAG implementation. Ensure you have sufficient API credits for production use.
