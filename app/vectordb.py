from langchain_chroma import Chroma

def create_vectordb(documents, embedding):
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    return vectordb
