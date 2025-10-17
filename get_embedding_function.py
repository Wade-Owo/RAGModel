from langchain_ollama import OllamaEmbeddings

#creating an embedding function to using for embedding our text

'''
need a function because we'll ue it in multiple places:
1. To embed the chunks
2. Embedding the query
'''

def get_embedding_function():
    embedding_func = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    return embedding_func

#might switch to OpenAI Embedding as it's said to be better than using the smaller sized ollama models