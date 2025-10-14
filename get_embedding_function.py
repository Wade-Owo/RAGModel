from langchain_community.embeddings.ollama import OllamaEmbeddings

#creating an embedding function to using for embedding our text
def get_embedding_function():
    embedding_func = OllamaEmbeddings(
        model="llama2:7b"
    )
    return embedding_func
'''
need a function because we'll ue it in multiple places:
1. To embed the chunks
2. Embedding the query
'''