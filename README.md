# UK Election 2024 Party Policy Retrieval System

This project builds a scalable Retrieval-Augmented Generation (RAG) pipeline to extract, chunk, embed, and semantically retrieve policy information from the 2024 UK political party manifestos.
It uses Azure Blob Storage for document management, OpenAI embeddings for vectorization, Pinecone for vector storage, and LangChain for retrieval and orchestration.

Users can ask natural language questions about the political manifestos and receive grounded, context-based answers.

The system has been evaluated using RAGAS to assess retrieval quality, answer relevance, and overall system performance.

## Contents
- `create_pinecone_index.py` — Script to create/reset the Pinecone vector index
- `create_chunks_load_embeddings.py` — Load PDFs from Azure, chunk text, embed, and upload to Pinecone
- `RAG_question_answer.py` — Main RAG pipeline: retrieves context and generates answers
- `RAG_Evaluation.ipynb` — Notebook for evaluating RAG retrieval and answer quality

## Technologies Used
- **Pinecone** — Vector database for storing embeddings
- **Azure Blob Storage** — Document storage
- **LangChain** — Framework for retrieval and chaining
- **OpenAI GPT-3.5** — Language model for answer generation
- **Streamlit / Hugging Face Spaces** — Deployment ready

## Workflow
1. PDFs are pulled from Azure Blob Storage.
2. Documents are split into chunks and embedded using OpenAI embeddings.
3. Embeddings are stored in Pinecone.
4. User queries retrieve top-matching document chunks.
5. GPT-3.5 generates answers based on the retrieved context.

## Author
Richard Smith