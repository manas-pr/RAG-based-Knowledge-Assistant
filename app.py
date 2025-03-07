import os
import faiss
import numpy as np
import openai
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import base64
from langchain_community.retrievers import BM25Retriever


# Function to Set Background Image
def set_background(image_file):
    if os.path.exists(image_file):  # Check if file exists
        with open(image_file, "rb") as image:
            encoded_image = base64.b64encode(image.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{encoded_image}');
            background-size: cover;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Background image not found. Using default background.")

set_background("background.png")

# Sidebar Information
with st.sidebar:
    st.title('🤖💬 Knowledge Assistant')
    st.markdown(
        """
        The Chatbot Assistant is an AI-powered application designed to help users interact with large datasets and get intelligent responses.
        It uses advanced natural language processing (NLP) techniques and machine learning models to understand user queries and retrieve relevant information 
        from a provided dataset, such as a PDF or CSV file.
        """
    )

    # User API Key Input
    st.subheader("🔑 Enter your OpenAI API Key")
    openai_api_key = st.text_input("API Key", type="password")
    
    # Store API Key in Session State
    if openai_api_key:
        st.session_state["api_key"] = openai_api_key

# Ensure API Key is Provided
if "api_key" not in st.session_state or not st.session_state["api_key"]:
    st.error("❌ Please enter your OpenAI API key to proceed.")
    st.stop()

openai.api_key = st.session_state["api_key"]

# Load PDF and Extract Text
pdf_path = "2409.15277v1.pdf"
if not os.path.exists(pdf_path):
    st.error(f"❌ PDF file '{pdf_path}' not found. Please upload it.")
    st.stop()

reader = PdfReader(pdf_path)

texts = []
for i, page in enumerate(reader.pages[:50]):  
    text = page.extract_text()
    if text and text.strip():
        texts.append(Document(page_content=text, metadata={"page": i + 1}))

# Text Splitting for Processing
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_texts = [
    Document(page_content=chunk, metadata=doc.metadata)
    for doc in texts
    for chunk in text_splitter.split_text(doc.page_content)
]

# Generate Embeddings
embeddings = []
for doc in split_texts:
    embedding_response = openai.embeddings.create(
        input=doc.page_content,
        model="text-embedding-ada-002"
    )
    embeddings.append(embedding_response.data[0].embedding)

# Create FAISS Index
embedding_dimension = len(embeddings[0])
faiss_index = faiss.IndexFlatL2(embedding_dimension)
faiss_index.add(np.array(embeddings))

# BM25 Retriever
bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in split_texts])

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS Styling
st.markdown("""
    <style>
        .user-message {
            background-color: #333333;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .assistant-message {
            background-color: #444444;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .stChatMessage {
            padding-left: 0;
            padding-right: 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 21px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>Knowledge Assistant</h1></div>', unsafe_allow_html=True)

# Display Chat Messages
for message in st.session_state.messages:
    message_class = "user-message" if message["role"] == "user" else "assistant-message"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{message_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Process User Input
if prompt := st.chat_input("Ask Here!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    # Generate Query Embedding
    query_response = openai.embeddings.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(query_response.data[0].embedding).reshape(1, -1)

    # FAISS Retrieval
    k = 5
    distances, indices = faiss_index.search(query_embedding, k)
    faiss_docs = [split_texts[i] for i in indices[0]]

    # BM25 Retrieval
    bm25_docs = bm25_retriever.get_relevant_documents(prompt)[:k]

    # Combine Results
    combined_docs = {doc.page_content: doc for doc in (faiss_docs + bm25_docs)}
    retrieved_texts = "\n".join([doc.page_content for doc in combined_docs.values()])

    # Construct Prompt with Context
    prompt_with_context = f"""
    user_prompt:
    {prompt}

    retrieved_context:
    {retrieved_texts}
    """

    # Generate AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        llm_response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt_with_context}]
        )
        full_response = llm_response["choices"][0]["message"]["content"]
        message_placeholder.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)

    # Store Assistant Response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
