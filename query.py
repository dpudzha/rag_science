"""Retrieve relevant chunks and answer questions."""
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import RetrievalQA
from config import *


def load_vectorstore() -> FAISS:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )


def ask(question: str):
    vectorstore = load_vectorstore()
    qa = build_qa_chain(vectorstore)
    result = qa.invoke(question)

    print(f"\n📌 Answer:\n{result['result']}\n")
    print("📄 Sources:")
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  - {source}: {snippet}...")


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What are the main findings?"
    ask(question)