# 📚 Local RAG Exam Assistant

A powerful, privacy-first exam preparation assistant powered by **Llama 3** and **Chroma vector database**. This application enables you to ask questions about your course notes using retrieval-augmented generation (RAG) locally on your machine.

## 🌟 Features

- **Local LLM Processing**: Uses Ollama with Llama 3 model—your data never leaves your machine
- **Vector Search**: Leverages Chroma vector database for semantic search across course notes
- **Multi-Course Support**: Organize and search across multiple courses simultaneously
- **Chat Interface**: Interactive Streamlit-powered chat UI for seamless Q&A
- **Smart References**: Displays source documents with page numbers for every answer
- **LangChain Integration**: Modern LangChain retrieval chains for robust RAG pipeline

## 📋 Requirements

- Python 3.8+
- Ollama with Llama 3 model installed
- Ollama with nomic-embed-text model installed

### Install Ollama

Visit [ollama.ai](https://ollama.ai) and follow the installation instructions for your OS.

### Pull Required Models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ahmeterenyasar/my-exam-assistant.git
cd my-exam-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
my-exam-assistant/
├── app.py                 # Main Streamlit application
├── ingest.py              # PDF ingestion and vectorization
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── chroma_db/            # Vector database (auto-generated)
│   └── .gitkeep         # Placeholder for empty directory
└── data/                 # PDF storage directory
    └── .gitkeep         # Placeholder for empty directory
```

## 📖 Usage

### Step 1: Prepare Your Course Materials

Create subdirectories under `data/` for each course and add PDF files:

```bash
mkdir -p data/Machine_Learning
mkdir -p data/Data_Science
# Add your PDF files to these directories
```

Example structure:
```
data/
├── Machine_Learning/
│   ├── lecture_1.pdf
│   ├── lecture_2.pdf
│   └── notes.pdf
├── Data_Science/
│   ├── module_1.pdf
│   └── module_2.pdf
└── .gitkeep
```

### Step 2: Ingest PDFs into Vector Database

Run the ingestion script to process all PDFs and create embeddings:

```bash
python ingest.py
```

**Options:**
```bash
python ingest.py --reset  # Clear existing database and reingest from scratch
```

This will:
- Discover all PDFs in `data/` subdirectories
- Extract text and split into semantic chunks
- Generate embeddings using Ollama's nomic-embed-text model
- Store vectors in Chroma database
- Create `courses.json` with course metadata

### Step 3: Launch the Chat Interface

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

### Step 4: Ask Questions

1. Select a course from the sidebar dropdown (or "All courses")
2. Type your question in the chat input
3. The assistant searches your notes and provides an answer
4. View the sources used under "Sources Used (References)"

## 🔧 How It Works

### Ingestion Pipeline (`ingest.py`)

1. **PDF Discovery**: Recursively finds all PDFs in `data/` subdirectories
2. **Course Inference**: Determines course name from directory structure
3. **Text Extraction**: Uses PyPDFLoader to extract text from PDFs
4. **Chunking**: Splits documents into 1000-token chunks with 200-token overlap
5. **Embedding**: Generates vector embeddings using Ollama's nomic-embed-text
6. **Storage**: Stores chunks and embeddings in Chroma vector database
7. **Metadata**: Preserves course, filename, and page number information

### RAG Chain (`app.py`)

1. **Retrieval**: Semantic search in vector database (top 4 results)
2. **Filtering**: Optionally filters results by selected course
3. **Document Chain**: Combines retrieved documents into context
4. **LLM Generation**: Sends context + question to Llama 3
5. **Response**: Returns answer with source references

## 🔒 Privacy & Security

- **Local Processing**: All processing happens on your machine
- **No Cloud Services**: Ollama models run locally
- **Data Storage**: Vector database stored in `chroma_db/` directory
- **No Data Transmission**: Your course materials are never sent anywhere

## 🛠️ Configuration

### LLM Model

Edit the model in `app.py` line 50:
```python
llm = Ollama(model="llama3")  # Change to other models if needed
```

### Embedding Model

Edit the model in both `app.py` (line 30) and `ingest.py` (line 63):
```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Retrieval Settings

Adjust number of results in `app.py` line 44:
```python
search_kwargs = {"k": 4}  # Change to return more/fewer results
```

### Chunk Settings

Edit in `ingest.py` line 60:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each text chunk
    chunk_overlap=200     # Overlap between chunks
)
```

## 📦 Dependencies

- **streamlit**: Web application framework
- **langchain-classic**: Legacy chain support for modern LangChain
- **langchain-community**: LLM and vector store integrations
- **langchain-core**: Core LangChain components
- **langchain-text-splitters**: Document chunking utilities
- **chromadb**: Vector database
- **pypdf**: PDF text extraction
- **ollama**: LLM integration

See `requirements.txt` for pinned versions.

## 🐛 Troubleshooting

### "Vector database not found"
Run `python ingest.py` to create the database first.

### "No courses found"
Add PDF files to `data/<course_name>/` directories and run `python ingest.py --reset`.

### Ollama Connection Error
Ensure Ollama is running:
```bash
ollama serve  # Start Ollama server
```

### Slow Responses
- Check if Ollama models are fully downloaded
- Reduce `chunk_size` in `ingest.py` for faster queries
- Increase `chunk_overlap` for better context

### Out of Memory
- Reduce the `k` parameter (number of retrieved documents)
- Use a smaller language model in Ollama
- Process fewer PDFs at once

## 📊 Performance Notes

- **First Query**: May be slower as models are loaded
- **Typical Query Time**: 5-15 seconds depending on your hardware
- **Vector Search**: Sub-second semantic search across all documents
- **LLM Generation**: Depends on model and response length

## 🎯 Future Enhancements

- [ ] Support for other document formats (DOCX, TXT, MD)
- [ ] Multi-language support
- [ ] Custom prompt templates
- [ ] Query history and bookmarking
- [ ] PDF highlighting with source navigation
- [ ] Batch ingestion progress tracking
- [ ] LLM model selection from UI
- [ ] Export answers to PDF

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This tool provides answers based on your course materials. While it aims for accuracy, always verify critical information with official sources or instructors. The assistant should be used as a study aid, not as the sole source of information.

## 💡 Tips for Best Results

1. **Quality PDFs**: Ensure PDFs are text-extractable (not scanned images)
2. **Organized Structure**: Create clear course directories
3. **Consistent Naming**: Use consistent file naming conventions
4. **Relevant Queries**: Ask specific questions for more accurate results
5. **Context**: Include relevant context in your questions
6. **Multiple Sources**: Include notes, slides, and textbook excerpts

## 📞 Support

For issues or questions, please open an issue on the GitHub repository.

---

**Made with ❤️ for exam preparation**

**Last Updated**: April 2026
