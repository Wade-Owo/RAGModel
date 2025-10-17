import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

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
    query_rag(query_text)

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