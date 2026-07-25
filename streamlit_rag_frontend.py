import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    save_thread_title,
    get_thread_title,
    get_all_thread_titles,
    load_conversation as backend_load_conversation,
    sync_retriever_from_session,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    # Don't reset thread_titles to preserve conversation history


def clear_all_history():
    """Delete all conversations and reset the database."""
    str_thread_ids = [str(t) for t in st.session_state["chat_threads"]]
    
    # Clear session state
    st.session_state["chat_threads"] = []
    st.session_state["thread_titles"] = {}
    st.session_state["message_history"] = []
    
    # Delete all threads from database
    try:
        import sqlite3
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        for tid in str_thread_ids:
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")
    
    # Start fresh
    reset_chat()
    st.rerun()


def add_thread(thread_id):
    str_thread_id = str(thread_id)
    if str_thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(str_thread_id)


def delete_thread(thread_id):
    """Delete a conversation and its associated data."""
    str_thread_id = str(thread_id)
    
    # Remove from chat_threads
    if str_thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(str_thread_id)
    
    # Remove from thread_titles
    if str_thread_id in st.session_state["thread_titles"]:
        del st.session_state["thread_titles"][str_thread_id]
    
    # Remove from ingested_docs (thread-specific only)
    if str_thread_id in st.session_state["ingested_docs"]:
        del st.session_state["ingested_docs"][str_thread_id]
    
    # Delete from SQLite checkpoint and titles
    try:
        import sqlite3
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str_thread_id,))
        cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (str_thread_id,))
        cursor.execute("DELETE FROM thread_titles WHERE thread_id = ?", (str_thread_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting from database: {e}")
    
    # If the deleted thread was the current thread, switch to another thread
    if str_thread_id == str(st.session_state["thread_id"]):
        if st.session_state["chat_threads"]:
            # Switch to the most recent thread
            new_thread = st.session_state["chat_threads"][-1]
            st.session_state["thread_id"] = new_thread
            # Load conversation for the new thread
            messages = backend_load_conversation(new_thread)
            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})
            st.session_state["message_history"] = temp_messages
        else:
            # No threads left, start fresh
            reset_chat()


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
else:
    # Clean up duplicates on every run
    st.session_state["chat_threads"] = list(dict.fromkeys(st.session_state["chat_threads"]))

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = get_all_thread_titles()

if "global_retriever" not in st.session_state:
    st.session_state["global_retriever"] = None

sync_retriever_from_session(st.session_state)

# Only add thread if it's not already in the list
if str(st.session_state["thread_id"]) not in st.session_state["chat_threads"]:
    add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
# Deduplicate threads and exclude current thread from past conversations
threads = [t for t in list(dict.fromkeys(st.session_state["chat_threads"])) if str(t) != thread_key][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("Talk With Your Doc")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if st.sidebar.button("Clear All History", use_container_width=True):
    clear_all_history()

# Show global PDF status
if st.session_state["global_retriever"] and "global_metadata" in st.session_state:
    global_meta = st.session_state["global_metadata"]
    st.sidebar.success(
        f"Using `{global_meta.get('filename')}` "
        f"({global_meta.get('chunks')} chunks from {global_meta.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    # Check if PDF is already globally indexed
    if st.session_state["global_retriever"] and "global_metadata" in st.session_state:
        current_filename = st.session_state["global_metadata"].get("filename")
        if uploaded_pdf.name == current_filename:
            st.sidebar.info(f"`{uploaded_pdf.name}` already processed globally.")
            sync_retriever_from_session(st.session_state)
        else:
            st.sidebar.info(f"Replacing `{current_filename}` with `{uploaded_pdf.name}`")
    
    try:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
                session_state=st.session_state,
            )
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
    except Exception as e:
        st.sidebar.error(f"Error indexing PDF: {str(e)}")

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        db_title = get_thread_title(str(thread_id))
        if not db_title:
            try:
                msgs = backend_load_conversation(thread_id)
                first_user = next(
                    (m.content for m in msgs if isinstance(m, HumanMessage) and m.content),
                    None,
                )
                db_title = (first_user[:30] + "...") if first_user else str(thread_id)[:8] + "..."
            except Exception:
                db_title = str(thread_id)[:8] + "..."
        
        thread_title = st.session_state["thread_titles"].get(str(thread_id), db_title)
        
        # Create columns for title and delete button
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            if st.button(thread_title, key=f"side-thread-{thread_id}"):
                selected_thread = thread_id
        
        with col2:
            if st.button("🗑️", key=f"delete-{thread_id}", help="Delete conversation"):
                st.session_state["thread_to_delete"] = thread_id

# Handle deletion outside the loop to avoid interference
if "thread_to_delete" in st.session_state and st.session_state["thread_to_delete"]:
    delete_thread(st.session_state["thread_to_delete"])
    del st.session_state["thread_to_delete"]

# ============================ Main Layout ========================
st.title("Talk With Your Doc")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Generate conversation title if this is the first message
    if len(st.session_state["message_history"]) == 1:
        title = user_input[:30] + "..." if len(user_input) > 30 else user_input
        st.session_state["thread_titles"][thread_key] = title
        save_thread_title(thread_key, title)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        try:
            def stream_chunks():
                for message_chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, AIMessage):
                        content = message_chunk.content or ""
                        if content:
                            yield content

            ai_message = st.write_stream(stream_chunks())

        except Exception as e:
            err = str(e)
            if "image" in err.lower() or "unsupported" in err.lower() or "cannot read" in err.lower():
                friendly = "I don't know. Please upload a PDF or ask a text-based question."
                st.markdown(friendly)
                ai_message = friendly
            else:
                friendly = f"Error: {err}"
                st.error(friendly)
                ai_message = friendly

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key, st.session_state)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = backend_load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
