# Talk With Your Doc

A powerful AI chatbot built with **LangGraph**, **LangChain**, **Groq**, **FAISS**, and **Streamlit**. Upload a PDF and chat with it using retrieval-augmented generation (RAG) in **50+ languages** — including Urdu, Arabic, French, Spanish, and more. Also includes built-in tools like web search, calculator, and live stock prices.

## Features

- PDF Upload & RAG: Upload any PDF and ask questions about its content
- Multilingual Support: Handles Urdu, Arabic, French, Spanish, and 50+ languages via multilingual embeddings
- Multi-Utility Tools: Web search (DuckDuckGo), calculator, and stock price lookup
- Multi-Conversation Support: Create, switch, and delete past conversations
- Clean Streaming UI: Real-time word-by-word streaming responses in Streamlit
- Persistent History: Conversation history and titles stored in SQLite
- Image-Safe Processing: Automatically strips image references from PDFs for text-only models

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq `llama-3.1-8b-instant` via ChatOpenAI |
| Backend | LangGraph + LangChain |
| Vector Store | FAISS + HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Frontend | Streamlit |
| Database | SQLite (LangGraph checkpointing + thread titles) |
| PDF Processing | PyPDFLoader + RecursiveCharacterTextSplitter |

## Live Demo

🔗 [Try it live on Streamlit Cloud](https://langgraph-pdf-assistant.streamlit.app/)

## Installation

```bash
# Clone the repository
git clone https://github.com/Gotam-Dulhani/Multi-Utility-Chatbot.git
cd Multi-Utility-Chatbot

# Create virtual environment
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1   # Windows
# source myvenv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

## Usage

```bash
streamlit run streamlit_rag_frontend.py
```

1. Open the local URL shown in terminal (usually `http://localhost:8501`)
2. Upload a PDF from the sidebar to index it
3. Ask questions about the document or use tools like search and calculator
4. Manage conversations from the sidebar — new chat, switch history, delete threads

## Project Structure

```
├── langraph_rag_backend.py   # LangGraph backend: LLM, tools, RAG, checkpointer
├── streamlit_rag_frontend.py # Streamlit UI, chat state, sidebar history
├── requirements.txt          # Python dependencies
├── .env                      # API keys (Groq)
├── chatbot.db                # SQLite database for checkpoints & titles
└── README.md                 # Project documentation
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
