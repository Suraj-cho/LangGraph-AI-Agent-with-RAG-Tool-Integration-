import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from chatbot_backend import (
    chatbot,
    upload_user_pdf,
    retrive_all_threads,
    get_user_vectorstore
)

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
)


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def get_message_text(message) -> str:
    """Convert LangChain message content into plain text."""
    if hasattr(message, "content"):
        content = message.content
    else:
        content = message

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
                elif "content" in item:
                    text_parts.append(str(item["content"]))
                else:
                    text_parts.append(str(item))
            else:
                text_parts.append(str(item))
        return "".join(text_parts)

    if isinstance(content, dict):
        return str(content)

    return str(content)


def add_thread(thread_id: str):
    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = []

    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def load_thread_history(thread_id: str, user_id: str):
    try:
        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            }
        )

        if state is None or not hasattr(state, "values"):
            return []

        messages = state.values.get("messages", [])
        history = []

        for msg in messages:
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            history.append({"role": role, "content": get_message_text(msg)})

        return history
    except Exception:
        return []


def initialize_session():
    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_thread_id()
        add_thread(st.session_state.thread_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_1"

    if st.session_state.thread_id not in st.session_state.chat_threads:
        add_thread(st.session_state.thread_id)


initialize_session()


with st.sidebar:
    st.title("AI Assistant")

    st.subheader("User")
    user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Each user gets their own uploaded PDF knowledge base.",
    )
    st.session_state.user_id = user_id

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a PDF to create or update the user's vector store.",
    )

    if uploaded_file is not None:
        save_dir = Path("uploads")
        save_dir.mkdir(exist_ok=True)

        file_name = f"{st.session_state.user_id}_{uploaded_file.name}"
        save_path = save_dir / file_name
        save_path.write_bytes(uploaded_file.getvalue())

        try:
            result = upload_user_pdf(
                user_id=st.session_state.user_id,
                pdf_path=str(save_path),
            )
            st.success(
                f"PDF uploaded successfully: {result['file_name']} "
                f"({result['pages']} pages, {result['chunks']} chunks)"
            )
        except Exception as e:
            st.error(f"PDF upload failed: {e}")

    st.subheader("Chat Controls")
    if st.button("New Chat"):
        st.session_state.thread_id = generate_thread_id()
        st.session_state.messages = []
        add_thread(st.session_state.thread_id)

    st.subheader("Recent Threads")
    try:
        all_threads = retrive_all_threads()
    except Exception:
        all_threads = []

    if all_threads:
        for thread in sorted(all_threads, reverse=True):
            button_label = str(thread)[:8]
            if st.button(button_label, key=f"thread_{thread}"):
                st.session_state.thread_id = thread
                st.session_state.messages = load_thread_history(
                    thread,
                    st.session_state.user_id,
                )
    else:
        st.caption("No saved chats yet.")


st.title("Chatbot")

if not st.session_state.messages:
    st.info("Ask a question or upload a PDF to start using the RAG assistant.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message here...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_reply = ""

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        for chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id,
                    "user_id": st.session_state.user_id,
                }
            },
            stream_mode="messages",
        ):

            if metadata.get("langgraph_node") != "chat_node":
                continue

            content = get_message_text(chunk)

            if content:
                assistant_reply += content
                message_placeholder.markdown(assistant_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply,
    })
    
if st.sidebar.button("View FAISS Data"):

    vector_store = get_user_vectorstore(
        st.session_state.user_id
    )

    if vector_store:

        for i, doc in enumerate(
            vector_store.docstore._dict.values(), 1
        ):

            st.write(f"### Chunk {i}")
            st.write("Metadata:", doc.metadata)
            st.write(doc.page_content)

    else:
        st.warning("No FAISS data found.")    