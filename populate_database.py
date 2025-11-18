import argparse
import os
import shutil
import hashlib
#import for loading text
from langchain_community.document_loaders import DirectoryLoader
#import for spliting
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#importing the embeddings function from the "get_embedding_function" file to use here
from get_embedding_function import get_embedding_function
#using chroma as our database to store the vectorized data
from langchain_chroma import Chroma

'''
Need to add the ability to edit an exisiting page
    if the pdf content in a chunk is modified, need to edit/update the chunk ID
    how do we know when we need to update this page?

    1. store hashed data content as metadata
    2. select records with changed hashed metadata and update them using collections.update()
'''

DATA_PATH = './data/'
CHROMA_PATH = 'chroma'

# def main(): 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         clear_database()

#     documents = load_documents()
#     chunks = split_documents(documents)
#     to_chroma(chunks)
#     if chroma_size_check():
#         print("Database Successfully populated")
#     else:
#         print("Unable to populate database, please try again")


# def main(): 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         clear_database()

#     documents = load_documents()
#     chunks = split_documents(documents)
#     to_chroma(chunks)
#     if chroma_size_check():
#         print("Database Successfully populated")
#     else:
#         print("Unable to populate database, please try again")

#loading the data from the 'data'
def load_documents():
    #loads all the txt files in the 'data' directory
    document_loader = DirectoryLoader(DATA_PATH, glob="*.txt", show_progress=True)
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


#adding the chunks into the vector database
def to_chroma(chunks: list[Document]):
    #laoding the database and the embedding function used for it
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = get_chunk_id(chunks)

    #adding the documents

    #going through all the items in the database and get all the ids
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing Documents in the Database: {len(existing_ids)}")

    #only add documents that don't exist
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new documents")
        #list of the new chunks' IDs
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        #adding the new chunks and their IDs into the Chroma DB
        db.add_documents(new_chunks, ids=new_chunks_ids)
        #force save the new additions
        db.persist()
    else:
        print("No new chunks to add")

def get_chunk_id(chunks):
    #create an ID based on the source, page number and chunk index
    last_page_id = None
    cur_chunk_idx = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        cur_page_id = f"{source}:{page}"

        '''
        if the current page is the same as the last, then add to the chunk id because it's the same page

        if it's a different page then start the chunk id at zero

        This avoids the issue of duplication
        '''
        if cur_page_id == last_page_id:
            cur_chunk_idx += 1
        else:
            cur_chunk_idx = 0
        
        #calculating the chunk id
        chunk_id = f"{cur_page_id}:{cur_chunk_idx}"
        last_page_id = cur_page_id

        #adding chunk_id metadata
        chunk.metadata["id"] = chunk_id

    return chunks #with updated metadata that includes the ids

#write function to check that the db is populated, and call after to_chroma
def chroma_size_check():
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    db_count = db._collection.count()
    if db_count > 0:
        return True
    return False

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# if __name__ == '__main__':
#     main()