import os
import time
import asyncio
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_text_splitters import MarkdownHeaderTextSplitter


PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MD_FILE = os.path.join(BASE_DIR, "wondervector5000.md")

if not os.path.exists(MD_FILE):
    st.error(f"🚨 Error: Markdown file not found at {MD_FILE}. Ensure it is in the same folder as chatbot.py.")
    st.stop()

with open(MD_FILE, "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [("##", "Header 2")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    st.error(f"🚨 Pinecone Error: {e}")
    st.stop()

embeddings = PineconeEmbeddings(
    model="multilingual-e5-large",
    pinecone_api_key=PINECONE_API_KEY
)

index_name = "rag-wondervector5000"
if index_name not in pc.list_indexes().names():
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(name=index_name, dimension=1024, metric="cosine", spec=spec)

namespace = "wondervector5000"
docsearch = PineconeVectorStore.from_documents(
    documents=md_header_splits,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

time.sleep(5)  # Ensure vectors are upserted

try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768",
        temperature=0.0
    )
except Exception as e:
    st.error(f"🚨 ChatGroq Error: {e}")
    st.stop()

retriever = docsearch.as_retriever()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)



st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .big-font { font-size:25px !important; }
    .stTextInput>div>div>input { font-size: 18px; padding: 10px; }
    .stChatMessage { font-size: 16px; padding: 10px; border-radius: 10px; }
    .stMarkdown p { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("RAG Chatbot")
st.write("Ask me anything about the WonderVector5000!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("💬 Your question:", placeholder="Ask me about the WonderVector5000...")

if user_query:
    try:
        answer_with_knowledge = retrieval_chain.invoke({"input": user_query})
        chatbot_response = answer_with_knowledge["answer"]

        st.session_state.chat_history.append(("Question:", user_query))
        st.session_state.chat_history.append(("🤖 RAG Chatbot:", chatbot_response))

    except Exception as e:
        st.error(f"🚨 Error processing query: {e}")

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(f"**{role}** {message}")

st.markdown("---")
st.markdown("Built with Pinecone, LangChain and ChatGroq for EdVentures")
