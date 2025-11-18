import argparse
import os
import copy
import hashlib

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from get_embedding_function import get_embedding_function
from populate_database import load_documents, split_documents, to_chroma, chroma_size_check, clear_database
#using duckduckgo search because google search wasn't working
from ddgs import DDGS

doc_l = []

CHROMA_PATH = "chroma"
DATA_PATH = './data/'

PROMPT_TEMPLATE =  """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    print("Clearing DB...")
    clear_database()
    print("DB Cleared!")

    #performing the search
    links = search_query(query_text)
    print(f"Links: {links}")
    load_docs(links)
    
    # documents = load_documents()
    # chunks = split_documents(documents)
    # to_chroma(chunks)
    # if chroma_size_check():
    #     print("Database Successfully populated")
    # else:
    #     print("Unable to populate database, please try again")
    # query_rag(query_text)

def search_query(query):
    # Construct the search URL (using Google search)
    link_array = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=10) 
            
            for result in results:
                title = result.get('title', 'No title')
                link = result.get('href', '')

                if '.pdf' in link.lower():
                    continue # for now don't do anything with pdfs

                link_array.append(link)
                print(f"Title: {title}")
                print(f"Link: {link}\n")
    
    except Exception as e:
        print(f"Error while searching: {e}")
        import traceback
        traceback.print_exc()

    return link_array

def load_docs(links):
    global doc_l
    try:
        loader = AsyncHtmlLoader(links)
        docs = loader.load()
        html2text = Html2TextTransformer()
        documents_transformed = html2text.transform_documents(docs)
        
        for i, d in enumerate(documents_transformed):
            url = d.metadata.get("source", "unknown")
            clean_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
            file = f"{DATA_PATH}/{clean_name}.txt"

            with open(file, "w", encoding="utf-8") as f:
                f.write(d.page_content)
            
            print(f"Saved: {file}")

        doc_l.append(copy.deepcopy(documents_transformed))
    except Exception as e:
        print(f"Error while loading {e}")
        import traceback
        traceback.print_exc()
    
    return docs

def query_rag(query_text: str):
    #preparing the db
    embedding_func = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

    #searching for similarity in the db and getting top k results
    results = db.similarity_search_with_score(query_text, k=3)
    print(results)

    context_info = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_temp = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_temp.format(context=context_info, question=query_text)

    print(prompt)

    model = Ollama(model="llama2")
    response_text = model.invoke(prompt)

    #citing where the LLM got it's answers

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_res = f"Response: {response_text}\n Sources: {sources}"

    print(formatted_res)
    return response_text

if __name__ == '__main__':
    main()