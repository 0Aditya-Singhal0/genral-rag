import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.vectorstores import Weaviate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv
import weaviate
from typing import List

load_dotenv()

app = FastAPI(title="Weaviate RAG Application")

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

weaviate_client = weaviate.Client(WEAVIATE_URL)


# Initialize weaviate schema
def init_weaviate_schema():
    schema = {
        "class": "Document",
        "description": "A collection of documents with text and filename.",
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
                "description": "Content of the document",
            },
            {
                "name": "filename",
                "dataType": ["text"],
                "description": "Name of the file",
            },
        ],
        "vectorizer": "text2vec-transformers",  # Configure the vectorizer
        "vectorIndexType": "flat",  # Configure the vector index type
        "moduleConfig": {
            "text2vec-transformers": {
                "poolingStrategy": "mean",
                "vectorizeClassName": False,
            },
            "reranker-transformers": {"enabled": True},
        },
    }

    if not weaviate_client.is_ready():
        weaviate_client.connect()

    if not weaviate_client.schema.exists("Document"):
        weaviate_client.schema.create_class(schema)
        print("Schema created successfully.")
    else:
        print("Schema already exists.")
        response = weaviate_client.query.aggregate("Document").with_meta_count().do()
        count = response["data"]["Aggregate"]["Document"][0]["meta"]["count"]
        if count == 0:
            print("Schema is empty. Deleting and recreating.")
            weaviate_client.schema.delete_class("Document")
            weaviate_client.schema.create_class(schema)
            print("Schema recreated successfully.")


init_weaviate_schema()


# Initialize VectorStore
vector_store = Weaviate(
    client=weaviate.Client(WEAVIATE_URL),
    index_name="Document",
    text_key="text",
    attributes=["filename"],
)

# Initialize LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# Initialize RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
)


@app.post("/add_files/", summary="Add files to the vector database")
async def add_files(files: List[UploadFile] = File(...)):
    """
    Upload and add files to the Weaviate vector database.
    """
    for file in files:
        content = await file.read()

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a valid text file.",
            )

        # Add to vector store
        vector_store.add_texts(
            [text], ids=[file.filename], metadatas=[{"filename": file.filename}]
        )
    return {"status": "Files added successfully"}


@app.post("/remove_files/", summary="Remove files from the vector database")
async def remove_files(file_ids: List[str]):
    """
    Remove files from the Weaviate vector database by their IDs.
    """
    try:
        weaviate_client.batch.delete_objects(
            class_name="Document",
            where={"operator": "In", "path": ["filename"], "valueText": file_ids},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "Files removed successfully"}


@app.post("/query/", summary="Query the RAG system")
async def query(prompt: str):
    """
    Query the RAG system with a prompt. Each prompt is treated independently (0 memory).
    """
    try:
        response = qa.run(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
