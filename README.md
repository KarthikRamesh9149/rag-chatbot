# RAG Chatbot  

The **RAG Chatbot** is a **Retrieval-Augmented Generation (RAG) AI assistant** designed to answer questions based on a **knowledge base** stored in **Pinecone**. It integrates **LangChain, GroqCloud LLM, and Pinecone Vector Database** to provide context-aware responses using **retrieval-based QA**.  

This chatbot processes a **markdown knowledge base**, indexes it in **Pinecone**, and retrieves the most relevant information for user queries. Powered by **ChatGroq's Mixtral-8x7B-32768 model**, it ensures high-quality answers tailored to the given dataset.  

---

## Features  

- **RAG-Based Conversational AI** – Uses **retrieval-augmented generation** to fetch accurate responses.  
- **Pinecone Vector Search** – Stores and retrieves knowledge efficiently using embeddings.  
- **GroqCloud LLM Integration** – Leverages `Mixtral-8x7B-32768` for intelligent response generation.  
- **Markdown Knowledge Processing** – Parses and indexes **markdown files** for structured information retrieval.  
- **Interactive Chat UI** – Built with **Streamlit**, featuring real-time Q&A and chat history.  
- **Scalable & Cloud-Compatible** – Can be expanded with additional documents, APIs, and deployment options.  

---

## Tech Stack  

- **Programming Language:** Python  
- **Framework:** Streamlit  
- **LLM API:** GroqCloud (`Mixtral-8x7B-32768`)  
- **Vector Database:** Pinecone  
- **Embeddings Model:** `multilingual-e5-large`  
- **Text Processing:** LangChain + Markdown Header Splitter  
