from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


app = Flask(__name__)


cached_llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
    Context: {context}
    Answer:
    [/INST]
    """
)



vector_db = Milvus.from_documents(
    docs,
    embeddings,
    collection_name = 'collection_1',
    connection_args={"host": "127.0.0.1", "port": "19530"},
)



@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"answer": response}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    milvus_store.from_documents(chunks)

    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        collection_name = 'collection_1',
        connection_args={"host": "127.0.0.1", "port": "19530"},
        )

    milvus_store.persist()
    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response



@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    print("Creating chain")
    retriever = milvus_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    print(result)
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )
    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


def start_app():
    app.run(host="0.0.0.0", port=8085, debug=True)


if __name__ == "__main__":
        start_app()