from langchain_ibm import WatsonxEmbeddings


from app.config import WATSONX_URL, PROJECT_ID, API_KEY, EMBEDDING_MODEL

def get_embeddings():
    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        apikey=API_KEY
    )
