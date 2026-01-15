import os

WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "YOUR_PROJECT_ID")
API_KEY = os.getenv("WATSONX_API_KEY", "YOUR_API_KEY")

EMBEDDING_MODEL = "ibm/slate-embedding-english-rtrvr"
LLM_MODEL = "mistralai/mixtral-8x7b-instruct-v01"
