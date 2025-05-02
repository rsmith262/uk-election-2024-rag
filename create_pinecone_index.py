from pinecone import Pinecone, ServerlessSpec
import time

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

pc_api_key = os.getenv("PINECONE_API_KEY")
# import streamlit as st
# pc_api_key = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=pc_api_key)

index_name = os.getenv("PINECONE_INDEX")
dimension = 1536 # this is dependent on type of embeddings being used. I'm using text-embedding-ada-002. This would be different if using other embeddings
metric = "cosine"



##########################################################

# gets list of index names
index_list = [item['name'] for item in pc.list_indexes()]

# https://docs.pinecone.io/reference/api/2025-01/control-plane/create_index
# https://docs.pinecone.io/reference/api/2025-01/control-plane/delete_index

# Create index if it doesn't exist
if index_name in index_list: # creates index if it doesn't already exist
    # If it does exist delete index
    pc.delete_index(name=index_name)

    # recreate blank index
    pc.create_index(
        name = index_name,
        dimension=dimension, # 384 dimension, see comment above
        metric=metric,
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

    # Checks status and waits until it's ready before continuing
    while True:
        status = pc.describe_index(index_name).status
        if status.get("ready", False):
            # break
            #print("Ready")
            break
        time.sleep(2)

else:
    # if it doesnt exist then create index
    pc.create_index(
        name = index_name,
        dimension=dimension, # 384 dimension, see comment above
        metric=metric,
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

    # Checks status and waits until it's ready before continuing
    while True:
        status = pc.describe_index(index_name).status
        if status.get("ready", False):
            # break
            #print("Ready")
            break
        time.sleep(2)


# Add above code to create_cunks_load_embeddings.py code
