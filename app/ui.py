import gradio as gr
from app.loaders import load_pdf
from app.splitter import split_documents
from app.embeddings import get_embeddings
from app.vectordb import create_vectordb
from app.retriever import create_retriever
from app.qa_chain import build_qa_chain

def rag_pipeline(pdf_file, question):
    documents = load_pdf(pdf_file.name)
    split_docs = split_documents(documents)
    embeddings = get_embeddings()
    vectordb = create_vectordb(split_docs, embeddings)
    retriever = create_retriever(vectordb)
    qa_chain = build_qa_chain(retriever)

    result = qa_chain.invoke({"query": question})
    return result["result"]

def launch_ui():
    interface = gr.Interface(
        fn=rag_pipeline,
        inputs=[
            gr.File(label="Upload PDF"),
            gr.Textbox(
                value="What this paper is talking about?",
                label="Ask a question"
            )
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Quest Analytics - AI RAG Assistant",
        description="LangChain + Watsonx RAG Assistant (Python 3.12)"
    )
    interface.launch()
