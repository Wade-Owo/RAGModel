#import for loading
from langchain_community.document_loaders import PyPDFDirectoryLoader
#import for spliting
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

#loading the data from the 'data'
DATA_PATH = './data/'
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# documents = load_documents()
# print(documents[1])

#spliting the documents
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents)
print(chunks[0])