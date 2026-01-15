def create_retriever(vectordb):
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
