# UK Election 2024 Party Policy Retrieval System

This project builds a scalable Retrieval-Augmented Generation (RAG) pipeline to extract, chunk, embed, and semantically retrieve policy information from the 2024 UK political party manifestos.
It uses Azure Blob Storage for document management, OpenAI embeddings for vectorization, Pinecone for vector storage, and LangChain for retrieval and orchestration.

Users can ask natural language questions about the political manifestos and receive grounded, context-based answers.

The system has been evaluated using RAGAS to assess retrieval quality, answer relevance, and overall system performance.