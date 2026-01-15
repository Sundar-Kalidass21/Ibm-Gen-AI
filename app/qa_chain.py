from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from app.config import WATSONX_URL, PROJECT_ID, API_KEY, LLM_MODEL

def build_qa_chain(retriever):
    llm = WatsonxLLM(
        model_id=LLM_MODEL,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        apikey=API_KEY,
        temperature=0.2
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
