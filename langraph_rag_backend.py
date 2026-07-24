from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
def _clean_text(text: str) -> str:
    """Remove image references and non-text artifacts from retrieved chunks."""
    if not text:
        return ""
    text = re.sub(r'(?i)\b(?:image|figure|fig\.?|table|diagram|chart|graph|plot|photo|picture|illustration|screenshot)\s*\d*\s*[:.]?\s*', '', text)
    text = re.sub(r'(?i)(?:see\s+)?(?:the\s+)?(?:above\s+)?(?:below\s+)?(?:image|images)\s*\d*', '', text)
    text = re.sub(r'(?i)(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.svg|\.webp|\.tiff)', '', text)
    text = re.sub(r'(?i)cannot read.*inform the user', '', text)
    text = re.sub(r'(?i)(?:image data|embedded image|vector graphic)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

import re

load_dotenv()

# -------------------
# 1. LLM + embeddings (Groq + local embeddings)
# -------------------
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=10,
)
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# -------------------
# 2. PDF retriever store (global across all threads)
# -------------------
_GLOBAL_RETRIEVER = None
_GLOBAL_METADATA = {}

def _get_retriever(session_state=None):
    """Fetch the global retriever if available."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is not None:
        return _GLOBAL_RETRIEVER
    if session_state and session_state.get("global_retriever"):
        return session_state["global_retriever"]
    return None


def _get_metadata(session_state=None) -> dict:
    """Fetch document metadata from module globals or session state."""
    if _GLOBAL_METADATA:
        return _GLOBAL_METADATA
    if session_state and session_state.get("global_metadata"):
        return session_state["global_metadata"]
    return {}


def sync_retriever_from_session(session_state) -> None:
    """Keep module-level retriever in sync with Streamlit session state."""
    global _GLOBAL_RETRIEVER, _GLOBAL_METADATA
    retriever = session_state.get("global_retriever")
    metadata = session_state.get("global_metadata")
    if retriever is not None:
        _GLOBAL_RETRIEVER = retriever
    if metadata:
        _GLOBAL_METADATA = metadata


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None, session_state=None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.page_content = _clean_text(doc.page_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        metadata = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        global _GLOBAL_RETRIEVER, _GLOBAL_METADATA
        _GLOBAL_RETRIEVER = retriever
        _GLOBAL_METADATA = metadata

        if session_state is not None:
            session_state["global_retriever"] = retriever
            session_state["global_metadata"] = metadata

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
# Note: DuckDuckGoSearchRun does not accept a "region" kwarg directly in its
# constructor. Region/locale is configured via DuckDuckGoSearchAPIWrapper instead.
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

search_wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper, name="search_tool")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF document.

    Args:
        query: The search query to find relevant information in the PDF

    Returns:
        dict: Contains query results with context and metadata from the PDF
    """
    retriever = _get_retriever()
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [_clean_text(doc.page_content) for doc in result]
    chunk_metadata = [doc.metadata for doc in result]
    source_file = _GLOBAL_METADATA.get("filename")

    context = [c for c in context if c]
    total_chars = sum(len(c) for c in context)
    if total_chars > 3000:
        context = context[:2]
        chunk_metadata = chunk_metadata[:2]

    return {
        "query": query,
        "context": context,
        "metadata": chunk_metadata,
        "source_file": source_file,
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config: Optional[RunnableConfig] = None):
    """LLM node that may answer or request a tool call."""
    retriever = _get_retriever()
    doc_meta = _get_metadata()
    has_document = retriever is not None

    if has_document:
        filename = doc_meta.get("filename", "uploaded PDF")
        doc_instruction = (
            f"A PDF document named '{filename}' is indexed and ready. "
            "For ANY question about the document or its topics, you MUST call `rag_tool` "
            f"with the user's question as the query parameter before answering. "
            f"Example: rag_tool(query='What is machine learning?'). "
            "Do NOT tell the user to upload a PDF — the document is already available."
        )
    else:
        doc_instruction = (
            "No PDF is indexed yet. If the user asks about a document, ask them to upload a PDF first."
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            f"{doc_instruction} "
            "If asked about an image, a non-PDF file, or any binary data, "
            "respond only with: 'I cannot process that. Please upload a PDF or ask a text-based question.' "
            "Do NOT attempt to read images or non-text files."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

_conn = sqlite3.connect("chatbot.db", check_same_thread=False)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS thread_titles (thread_id TEXT PRIMARY KEY, title TEXT)"
)
_conn.commit()
_title_conn_lock = False

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def save_thread_title(thread_id: str, title: str) -> None:
    try:
        _conn.execute(
            "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (thread_id, title),
        )
        _conn.commit()
    except Exception:
        pass


def get_thread_title(thread_id: str) -> str | None:
    try:
        row = _conn.execute(
            "SELECT title FROM thread_titles WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def get_all_thread_titles() -> dict[str, str]:
    try:
        rows = _conn.execute("SELECT thread_id, title FROM thread_titles").fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    threads_with_messages = []
    for tid in all_threads:
        try:
            msgs = load_conversation(tid)
            if msgs:
                threads_with_messages.append(tid)
        except Exception:
            pass
    return threads_with_messages


def load_conversation(thread_id: str):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        return [
            msg for msg in messages
            if isinstance(msg, (HumanMessage, AIMessage)) and (msg.content or "").strip()
        ]
    except Exception:
        return []


def thread_has_document(thread_id: str, session_state=None) -> bool:
    if session_state and "thread_retrievers" in session_state:
        return str(thread_id) in session_state["thread_retrievers"]
    return False


def thread_document_metadata(thread_id: str, session_state=None) -> dict:
    return _get_metadata(session_state)