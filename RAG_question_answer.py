# https://python.langchain.com/v0.2/docs/integrations/chat/openai/
# https://realpython.com/build-llm-rag-chatbot-with-langchain/#chat-models
# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/

# https://medium.com/@varsha.rainer/building-a-rag-application-from-scratch-using-langchain-openais-whisper-pinecone-6b2fbf22f77f
# https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/

# imports
# Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# openAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings # should be using this instead as one beloe is depricated

# chain imports
from langchain_core.prompts import ChatPromptTemplate # changed this since initial script as it imported from somewhere else
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# getting rid of openMP warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*OpenMP.*")

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

# API keys - move these when finished testing
openai_api = os.getenv("OPENAI_API_KEY")
pinecone_api = os.getenv("PINECONE_API_KEY")


# embeddings
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(
    model=model_name,
    api_key=openai_api
)

# initialise pinecone
pc = Pinecone(api_key=pinecone_api)

# Pinecone index name and namespace
index_name = os.getenv("PINECONE_INDEX")
namespace = ''

# Connect to the Pinecone index
index = pc.Index(index_name)

# initialise the vectorstore
text_field = "text"  
vectorstore = PineconeVectorStore(  
    index, embeddings, text_field  
)

###############

# prmopt template - frames the question and context passed to the llm
template = """
You are an analyst answering questions using excerpts from UK political party manifestos (2024 General Election).

Answer strictly based on the provided context.  
If the context is insufficient, reply: "I don't have any information on this I'm afraid."  
Do not use external knowledge or make assumptions.

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# llm model used
llm = ChatOpenAI(  
    api_key=openai_api,  
    model='gpt-3.5-turbo',  
    temperature=0.0  
)

# parses the outout of the llm into a string format
parser = StrOutputParser()

# retriever gets the relevant chunks from pinecone to be used as context
chain = (
    {"context": vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}), "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# function to handle user queries, the important bit is the chain.invoke() bit
# this is where the query is passed (invoked) along the chain
def handle_query(query):
    response = chain.invoke(query)
    return response

# loop for asking multiple questions
while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = handle_query(query)
    print(f"\nAnswer: {response}\n")