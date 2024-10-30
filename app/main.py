import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.vectorstores import Weaviate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from enum import Enum
import weaviate
import logging
import asyncio
import uuid


class IdentifierType(str, Enum):
    filename = "filename"
    uuid = "uuid"


class RemoveFilesRequest(BaseModel):
    identifiers: List[str]
    identifier_type: IdentifierType


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Weaviate RAG Application")

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

weaviate_client = weaviate.Client(WEAVIATE_URL)


def delete_all_items():
    if not weaviate_client.is_ready():
        weaviate_client.connect()
    # Get the schema information
    schema_data = weaviate_client.schema.get()

    # Iterate through collections
    for collection_name in schema_data["classes"]:
        # Delete all objects in this collection
        weaviate_client.schema.delete_class(collection_name["class"])


# delete_all_items()


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
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizePropertyName": False,
                    }
                },
            },
            {
                "name": "filename",
                "dataType": ["string"],
                "description": "Name of the file",
            },
        ],
        "vectorizer": "text2vec-transformers",  # Configure the vectorizer
        "vectorIndexType": "flat",  # Configure the vector index type
        "moduleConfig": {
            "reranker-transformers": {"enabled": True},
            "text2vec-transformers": {
                "model": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                "options": {"waitForModel": True},
            },
        },
    }

    if not weaviate_client.is_ready():
        weaviate_client.connect()

    if not weaviate_client.schema.exists("Document"):
        weaviate_client.schema.create_class(schema)
        logger.info("Schema created successfully.")
    else:
        logger.info("Schema already exists.")
        response = weaviate_client.query.aggregate("Document").with_meta_count().do()
        count = response["data"]["Aggregate"]["Document"][0]["meta"]["count"]
        if count == 0:
            logger.info("Schema is empty. Deleting and recreating.")
            weaviate_client.schema.delete_class("Document")
            weaviate_client.schema.create_class(schema)
            logger.info("Schema recreated successfully.")


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
    documents = []
    metadata = []
    ids = []

    for file in files:
        content = await file.read()

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"File {file.filename} is not a valid text file.")
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a valid text file.",
            )

        documents.append(text)
        metadata.append({"filename": file.filename})
        ids.append(uuid.uuid4())

    # Run add_texts in a separate thread to avoid blocking
    try:
        await asyncio.to_thread(
            vector_store.add_texts, texts=documents, ids=ids, metadatas=metadata
        )
        logger.info(f"Added {len(files)} files to the vector store.")
    except Exception as e:
        logger.error(f"Error adding files: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to add files to the vector store."
        )

    return {"status": "Files added successfully"}


@app.post("/remove_files/", summary="Remove files from the vector database")
async def remove_files(request: RemoveFilesRequest):
    """
    Remove files from the Weaviate vector database by their filenames or UUIDs.
    """
    identifiers = request.identifiers
    identifier_type = request.identifier_type

    try:
        if identifier_type == IdentifierType.uuid:
            # Delete by UUIDs
            for obj_id in identifiers:
                weaviate_client.data_object.delete(uuid=obj_id, class_name="Document")
        elif identifier_type == IdentifierType.filename:
            # Use ContainsAny operator to delete objects with matching filenames
            weaviate_client.batch.delete_objects(
                class_name="Document",
                where={
                    "operator": "ContainsAny",
                    "path": ["filename"],
                    "valueStringArray": identifiers,
                },
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid identifier_type")
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
