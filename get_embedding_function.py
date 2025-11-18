import getpass
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
#creating an embedding function to using for embedding our text

'''
need a function because we'll ue it in multiple places:
1. To embed the chunks
2. Embedding the query
'''
load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAi API key: ")

def get_embedding_function():
    embedding_func = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=key,
        dimensions=1024
    )
    return embedding_func

#might switch to OpenAI Embedding as it's said to be better than using the smaller sized ollama models