# RAG Agent

This project is a Retrieval-Augmented Generation (RAG) application prototype that allows you to ingest PDF documents and ask questions that are answered using the content of those documents.

Rather than relying only on a language model’s general knowledge, the system retrieves relevant information from ingested documents and uses it as context when generating answers. This results in responses that are grounded, document-aware, and more reliable.

## What This Project Does

The RAG Pipeline works like this:

1. PDF documents are loaded and split into text chunks
2. Each chunk is converted into a vector embedding
3. Embeddings are stored in a vector database
4. A user query is embedded and used to retrieve the most relevant chunks
5. An LLM generates an answer using only the retrieved context

## Tools & Technologies Used

- **LlamaIndex**
  Used for loading PDF files and splitting them into semantically meaningful text chunks

- **OpenAI**
  Used to generate text embeddings and to produce final answers using a large language model 

- **Qdrant**
  A vector database used to store embeddings and perform similarity search

- **FastAPI**
  Serves as the API layer that handles ingestion and query requests

- **Inngest**
  Orchestrates background workflows like document ingest, embedding, retrieval, and answer generation

- **Streamlit**
  Provides a simple web-based user interface to interact with the system

- **Docker**
  Used to run Qdrant locally in a containerized environment
