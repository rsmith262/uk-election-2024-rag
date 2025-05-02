from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF
import io
import re
from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# for using the .env file
# this try except block uses the .env file if using locally but won't in production and will use environment variables from production instead
# this should still work in huggingface spaces but would not work using streamlit.

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not used in deployment

import os

##########################################################

openai_api = os.getenv("OPENAI_API_KEY")
# import streamlit as st
# openai_api = st.secrets["PINECONE_API_KEY"]

##########################################################
# Azure Blob Storage connection details
connection_string = os.getenv(r"BLOB_STORAGE_CONNECTION_STRING")
container_name = os.getenv("BLOB_STORAGE_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# This function downloads the blob data from Azure Blob Storage and reads it into an in-memory bytes buffer using io.BytesI
def read_blob_to_bytes(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    return io.BytesIO(blob_data)

##########################################################
# create chunks

# extract and clean text
def extract_pdf_text(blob_data):
    doc = fitz.open(stream=blob_data, filetype="pdf")
    full_text = ""
    for page in doc:
        text = page.get_text()
        text = re.sub(r"\n{2,}", "\n\n", text)  # normalise multiple newlines
        text = re.sub(r"Page\s*\d+", "", text)  # Remove standalone page numbers
        text = re.sub(r"\s{2,}", " ", text)  # Remove extra spaces
        full_text += text + "\n"
    return full_text.strip()

# use markdown-style heading splitting
def split_by_headings(text):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "section"),
        ("##", "subsection"),
        ("###", "subsubsection")
    ])
    return splitter.split_text(text)

# split each section into chunks
def split_into_chunks_with_metadata(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    for doc in docs:
        base_metadata = doc.metadata
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["source_section"] = base_metadata.get("section", "Unknown")
            all_chunks.append({
                "content": chunk,
                "metadata": chunk_metadata
            })

    return all_chunks


##########################################################
# generate embeddings

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    api_key=openai_api
)

# Function to generate vector embeddings using OpenAI
def generate_embeddings(chunked_docs):
    text_chunks = [chunk["content"] for chunk in chunked_docs]  # extract the actual text
    embeddings = embed.embed_documents(text_chunks)
    return embeddings

##########################################################
#Store embeddings in pinecone

pc_api_key = os.getenv("PINECONE_API_KEY")
pc_index = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=pc_api_key)

index = pc.Index(pc_index)

# https://docs.pinecone.io/guides/data/upsert-data
# upserting the chunked vectors into pinecone
def store_embeddings_in_pinecone(embeddings, chunked_docs, document_name):
    pinecone_vectors = []

    # this unpacks the chuck_dict into content and metadata
    for i, (embedding, chunk_dict) in enumerate(zip(embeddings, chunked_docs)):
        # text_chunk = chunk_dict["content"]
        metadata = chunk_dict["metadata"]

        # adds general metadata
        metadata["document"] = document_name
        metadata["chunk_id"] = f"{document_name}_{i}"
        metadata["text"] = chunk_dict["content"]  # Adds the actual chunk text for retrieval

        pinecone_vectors.append((metadata["chunk_id"], embedding, metadata))

    index.upsert(vectors=pinecone_vectors)


##########################################################
# run pipeline

# get list of blob names
blob_data = list(container_client.list_blobs())
blob_list = [blob.name for blob in blob_data]

for name in blob_list:
    blob_name = name #takes current blob name
    blob_data = read_blob_to_bytes(blob_name) # reads in blob to memory
    #text_chunks = extract_text_from_pdf(blob_data) # extracts text into chunks 1 page is one chunk
    ##
    raw_text = extract_pdf_text(blob_data)
    structured_docs = split_by_headings(raw_text)
    chunked_docs = split_into_chunks_with_metadata(structured_docs)
    ##
    embeddings = generate_embeddings(chunked_docs) # generates vector embeddings
    store_embeddings_in_pinecone(embeddings, chunked_docs, blob_name) # stores embeddings in pinecone




