from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF
import io
from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings

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

# This function takes the in-memory bytes buffer and opens it directly using PyMuPDF's fitz.open, which supports reading from streams.
def extract_text_from_pdf(blob_data):
    document = fitz.open(stream=blob_data, filetype="pdf")
    text_chunks = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text_chunks.append(page.get_text())
    return text_chunks


##########################################################

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    api_key=openai_api
)

# Function to generate vector embeddings using OpenAI
def generate_embeddings(text_chunks):
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
def store_embeddings_in_pinecone(embeddings, text_chunks, document_name):
    pinecone_vectors = []
    for i, (embedding, text_chunk) in enumerate(zip(embeddings, text_chunks)):
        metadata = {'document': document_name, 'page': str(i + 1), 'text': text_chunk}
        pinecone_vectors.append((f'{document_name}_{i}', embedding, metadata))
    index.upsert(vectors=pinecone_vectors)

##########################################################
# run pipeline

# get list of blob names
blob_data = list(container_client.list_blobs())
blob_list = [blob.name for blob in blob_data]

for name in blob_list:
    blob_name = name #takes current blob name
    blob_data = read_blob_to_bytes(blob_name) # reads in blob to memory
    text_chunks = extract_text_from_pdf(blob_data) # extracts text into chunks 1 page is one chunk
    embeddings = generate_embeddings(text_chunks) # generates vector embeddings
    store_embeddings_in_pinecone(embeddings, text_chunks, blob_name) # stores embeddings in pinecone
