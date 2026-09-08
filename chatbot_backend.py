
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sqlite3
from langgraph.prebuilt import ToolNode
import requests
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
import uuid
import json
#-----
# 1.llm
# ----- python langgraph_rag_backend.py
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# LONG TERM MEMORY
# ============================================================

MEMORY_DIR = "chroma_memory"

memory_store = Chroma(
    collection_name="long_term_memory",
    persist_directory=MEMORY_DIR,
    embedding_function=embeddings
)


# ------------------------------------------------------------
# STORE MEMORY
# ------------------------------------------------------------

def store_memory(user_id: str, memory: str, memory_type: str = "user_memory"):

    document = Document(
        page_content=memory,
        metadata={
            "user_id": user_id,
            "memory_type": memory_type
        }
    )

    memory_store.add_documents(
        documents=[document],
        ids=[str(uuid.uuid4())]
    )

    print(f"Memory stored for {user_id}: {memory}")


# ------------------------------------------------------------
# MEMORY EXTRACTOR
# ------------------------------------------------------------

def memory_extractor(messages):

    recent_messages = messages[-6:]

    conversation = ""

    for message in recent_messages:

        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"

        else:
            conversation += f"Assistant: {message.content}\n"


    prompt = f"""
You are a long-term memory extractor.

Analyze this conversation:

{conversation}

Find information about the USER that is useful to remember
for future conversations.

Store things like:
- User preferences
- User's skills
- User's long-term goals
- User's projects
- Technology preferences
- Recurring requirements

Do NOT store:
- Temporary questions
- Normal conversation
- Assistant information
- Information useful only for the current question

Return ONLY valid JSON.

If something should be remembered:

{{
    "should_store": true,
    "memory": "short description"
}}

Otherwise:

{{
    "should_store": false,
    "memory": ""
}}
"""

    response = llm.invoke(prompt)

    try:

        result = json.loads(response.content)

        return result

    except Exception:

        return {
            "should_store": False,
            "memory": ""
        }
        
        
# ============================================================
# 2. PDF INGESTION
# ============================================================

UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstores"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ------------------------------------------------------------
# TEXT SPLITTER
# ------------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# ------------------------------------------------------------
# UPLOAD USER PDF
# ------------------------------------------------------------

def upload_user_pdf(user_id: str, pdf_path: str):

    print(f"Processing PDF for user: {user_id}")

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    # 2. Add metadata
    for doc in docs:

        doc.metadata["user_id"] = user_id

        doc.metadata["source_file"] = os.path.basename(
            pdf_path
        )

    # 3. Split into chunks
    chunks = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} pages")
    print(f"Created {len(chunks)} chunks")

    # 4. User-specific FAISS path
    user_vector_path = os.path.join(
        VECTORSTORE_DIR,
        user_id
    )

    # 5. Existing FAISS
    if os.path.exists(user_vector_path):

        vector_store = FAISS.load_local(
            user_vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Add new PDF chunks
        vector_store.add_documents(chunks)

    # 6. First PDF
    else:

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

    # 7. Save FAISS
    vector_store.save_local(
        user_vector_path
    )

    return {
        "user_id": user_id,
        "file_name": os.path.basename(pdf_path),
        "pages": len(docs),
        "chunks": len(chunks)
    }


# ------------------------------------------------------------
# LOAD USER VECTOR STORE
# ------------------------------------------------------------

def get_user_vectorstore(user_id: str):

    user_vector_path = os.path.join(
        VECTORSTORE_DIR,
        user_id
    )

    if not os.path.exists(user_vector_path):

        return None

    vector_store = FAISS.load_local(
        user_vector_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


#------
# 2. tools
#------

search_tool = DuckDuckGoSearchRun(region="us-en")


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
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=09KOOAGX295DKYWS"
    r = requests.get(url)
    return r.json()

# RAG TOOL

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Search the user's uploaded PDF documents and return
    relevant information from them.

    MUST use this tool when the user asks questions about:
    - their uploaded PDF
    - resume
    - CV
    - cover letter
    - document content
    - skills mentioned in the document
    - experience, education, projects, or qualifications
    """

    # Get user_id
    configurable = config.get(
        "configurable",
        {}
    )

    user_id = configurable.get("user_id")

    if not user_id:

        return {
            "error": "user_id was not provided."
        }

    # Load user's FAISS
    vector_store = get_user_vectorstore(
        user_id
    )

    if vector_store is None:

        return {
            "error": "No documents uploaded for this user."
        }

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    # Retrieve
    result = retriever.invoke(query)

    # Context
    context = [
        doc.page_content
        for doc in result
    ]

    metadata = [
        doc.metadata
        for doc in result
    ]

    return {
        "query": query,
        "user_id": user_id,
        "context": context,
        "metadata": metadata
    }

        
tools = [search_tool, calculator, get_stock_price ,rag_tool]

llm_with_tools = llm.bind_tools(tools)



# ------
# 3. Create State
# ------ 

from langgraph.graph.message import add_messages

class ChatState(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]
    

# ------
# 4. Create Node
# ------

def chat_node(state: ChatState):

    messages = state["messages"]

    system_prompt = SystemMessage(content="""
You are a RAG assistant.

If the user asks anything about their uploaded PDF,
resume, CV, cover letter, skills, education, experience,
projects, or document content, you MUST use the rag_tool.

Do not answer from memory or general knowledge when the
answer should come from the uploaded document.
""")

    response = llm_with_tools.invoke(
        [system_prompt] + messages
    )

    return {"messages": [response]}


# ============================================================
# NEW: LONG-TERM MEMORY RETRIEVAL
# ============================================================

def retrieve_memory(user_id: str, query: str, k: int = 3):

    results = memory_store.similarity_search(
        query=query,
        k=k,
        filter={"user_id": user_id}
    )

    memories = []

    for doc in results:
        memories.append(doc.page_content)

    return memories


def memory_retrieval_node(
    state: ChatState,
    config: RunnableConfig
):

    messages = state["messages"]

    configurable = config.get(
        "configurable",
        {}
    )

    user_id = configurable.get("user_id")

    if not user_id:
        return {}

    # Find latest user message
    query = ""

    for message in reversed(messages):

        if isinstance(message, HumanMessage):
            query = message.content
            break

    if not query:
        return {}

    # Retrieve relevant long-term memories
    memories = retrieve_memory(
        user_id=user_id,
        query=query,
        k=3
    )

    if not memories:
        return {}

    memory_text = "\n".join(
        f"- {memory}"
        for memory in memories
    )

    # Add retrieved memory as system context.
    # It is not another user question.
    from langchain_core.messages import SystemMessage

    memory_message = SystemMessage(
        content=f"""
Relevant long-term memories about the user:

{memory_text}

Instructions:
- Use these memories only when relevant.
- Do not mention the memory system to the user.
- Do not blindly trust a memory if it conflicts with the current user message.
"""
    )

    return {
        "messages": [memory_message]
    }


# ============================================================
# NEW: LONG-TERM MEMORY EXTRACTOR NODE
# ============================================================

def memory_node(
    state: ChatState,
    config: RunnableConfig
):

    messages = state["messages"]

    configurable = config.get(
        "configurable",
        {}
    )

    user_id = configurable.get("user_id")

    if not user_id:
        return {}

    # Extract useful information from the latest conversation
    result = memory_extractor(messages)

    if result.get("should_store"):

        memory = result.get("memory")

        if memory:

            store_memory(
                user_id=user_id,
                memory=memory
            )

    return {}


tool_node = ToolNode(tools)

# -------
# 5. Checkpointer
# -------  
conn= sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ============================================================
# 6. CREATE GRAPH
# ============================================================

graph = StateGraph(ChatState)

# Existing nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Long-term memory nodes
graph.add_node("memory_retrieval", memory_retrieval_node)
graph.add_node("memory_node", memory_node)


# ============================================================
# ROUTING
# ============================================================

def route_after_chat(state: ChatState):

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "memory_node"


# ============================================================
# GRAPH FLOW
# ============================================================

graph.add_edge(
    START,
    "memory_retrieval"
)

graph.add_edge(
    "memory_retrieval",
    "chat_node"
)


# Chat Node → Tools OR Memory
graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "tools": "tools",
        "memory_node": "memory_node"
    }
)


# Tools → Chat Node
graph.add_edge(
    "tools",
    "chat_node"
)


# Memory Node → END
graph.add_edge(
    "memory_node",
    END
)

# Compile Graph
chatbot = graph.compile(
    checkpointer=checkpointer
)


# Thread configuration
config = {
    "configurable": {
        "thread_id": "user_1",
        "user_id": "user_1"
    }
}


def retrive_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
       all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)




# ---------------------------------------------------------------
# Generate graph diagram
png_data = chatbot.get_graph().draw_mermaid_png()

with open("chatbot_workflow.png", "wb") as f:
    f.write(png_data)

print("Workflow saved as chatbot_workflow.png")