import uuid
import asyncio
import weaviate
from enum import Enum
from typing import List
from logger import logger
from config import settings

from langchain_openai import OpenAI
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Weaviate
from fastapi.security.api_key import APIKeyHeader, APIKey
from langchain.chains.retrieval_qa.base import RetrievalQA
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status

# Security: API Key Authentication
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def get_api_key(api_key_header: str = Depends(api_key_header)) -> APIKey:
    if settings.SECURITY_ENABLED:
        if api_key_header == settings.API_KEY:
            return api_key_header
        else:
            logger.warning("Unauthorized access attempt.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
    else:
        return api_key_header


# Initialize FastAPI app
app = FastAPI(
    title="Weaviate RAG Application",
    description="A production-ready FastAPI application integrating Weaviate, LangChain, and OpenAI for Retrieval-Augmented Generation (RAG).",
    version="1.0.0",
)

# Initialize Weaviate Client
weaviate_client = weaviate.Client(settings.WEAVIATE_URL)


# Exception Handling
class WeaviateConnectionError(Exception):
    pass


@app.on_event("startup")
async def startup_event():
    try:
        if not weaviate_client.is_ready():
            weaviate_client.connect()
            logger.info("Connected to Weaviate successfully.")
        init_weaviate_schema()
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise WeaviateConnectionError("Cannot connect to Weaviate database.")


# Define Enums and Models
class IdentifierType(str, Enum):
    filename = "filename"
    uuid = "uuid"


class RemoveFilesRequest(BaseModel):
    identifiers: List[str] = Field(..., example=["file1.txt", "file2.txt"])
    identifier_type: IdentifierType


# Initialize Weaviate Schema
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
        "vectorizer": "text2vec-transformers",
        "vectorIndexType": "flat",
        "moduleConfig": {
            "reranker-transformers": {"enabled": True},
            "text2vec-transformers": {
                "model": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                "options": {"waitForModel": True},
            },
        },
    }

    if not weaviate_client.schema.exists("Document"):
        weaviate_client.schema.create_class(schema)
        logger.info("Weaviate schema 'Document' created successfully.")
    else:
        logger.info("Weaviate schema 'Document' already exists.")
        response = weaviate_client.query.aggregate("Document").with_meta_count().do()
        count = response["data"]["Aggregate"]["Document"][0]["meta"]["count"]
        if count == 0:
            logger.info("Schema 'Document' is empty. Reinitializing.")
            weaviate_client.schema.delete_class("Document")
            weaviate_client.schema.create_class(schema)
            logger.info("Weaviate schema 'Document' recreated successfully.")


# Dependency Injection for VectorStore, LLM, and QA
def get_vector_store() -> Weaviate:
    return Weaviate(
        client=weaviate_client,
        index_name="Document",
        text_key="text",
        attributes=["filename"],
    )


def get_llm() -> OpenAI:
    return OpenAI(openai_api_key=settings.OPENAI_API_KEY, temperature=0)


def get_retrieval_qa(
    llm: OpenAI = Depends(get_llm), vector_store: Weaviate = Depends(get_vector_store)
) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )


# API Endpoints
@app.post(
    "/add_files/",
    summary="Add files to the vector database",
    dependencies=[Depends(get_api_key)],
)
async def add_files(files: List[UploadFile] = File(...)):
    """
    Upload and add files to the Weaviate vector database.
    """
    if not files:
        logger.warning("No files uploaded in the request.")
        raise HTTPException(status_code=400, detail="No files uploaded.")

    documents = []
    metadata = []
    ids = []

    for file in files:
        content = await file.read()
        if not content:
            logger.warning(f"File {file.filename} is empty.")
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is empty.",
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"File {file.filename} is not a valid UTF-8 text file.")
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a valid UTF-8 text file.",
            )

        documents.append(text)
        metadata.append({"filename": file.filename})
        ids.append(str(uuid.uuid4()))

    vector_store = get_vector_store()

    try:
        await asyncio.to_thread(
            vector_store.add_texts, texts=documents, ids=ids, metadatas=metadata
        )
        logger.info(f"Successfully added {len(files)} files to the vector store.")
    except Exception as e:
        logger.exception("Error adding files to the vector store.")
        raise HTTPException(
            status_code=500, detail="Failed to add files to the vector store."
        )

    return {"status": "Files added successfully", "added_count": len(files)}


@app.post(
    "/remove_files/",
    summary="Remove files from the vector database",
    dependencies=[Depends(get_api_key)],
)
async def remove_files(request: RemoveFilesRequest):
    """
    Remove files from the Weaviate vector database by their filenames or UUIDs.
    """
    identifiers = request.identifiers
    identifier_type = request.identifier_type

    if not identifiers:
        logger.warning("No identifiers provided for removal.")
        raise HTTPException(status_code=400, detail="No identifiers provided.")

    try:
        if identifier_type == IdentifierType.uuid:
            # Validate UUIDs
            for obj_id in identifiers:
                try:
                    uuid.UUID(obj_id)
                except ValueError:
                    logger.error(f"Invalid UUID format: {obj_id}")
                    raise HTTPException(
                        status_code=400, detail=f"Invalid UUID: {obj_id}"
                    )
            # Delete by UUIDs
            for obj_id in identifiers:
                weaviate_client.data_object.delete(uuid=obj_id, class_name="Document")
            logger.info(f"Deleted {len(identifiers)} objects by UUID.")
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
            logger.info(f"Deleted objects with filenames: {identifiers}")
        else:
            logger.error(f"Invalid identifier_type: {identifier_type}")
            raise HTTPException(status_code=400, detail="Invalid identifier_type")
    except weaviate.exceptions.WeaviateException as e:
        logger.exception("Weaviate exception occurred during file removal.")
        raise HTTPException(
            status_code=500, detail="Failed to remove files from the vector store."
        )
    except Exception as e:
        logger.exception("Unexpected error during file removal.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    return {"status": "Files removed successfully", "removed_count": len(identifiers)}


@app.post(
    "/query/", summary="Query the RAG system", dependencies=[Depends(get_api_key)]
)
async def query(
    prompt: str,
    qa: RetrievalQA = Depends(get_retrieval_qa),
):
    """
    Query the RAG system with a prompt. Each prompt is treated independently (0 memory).
    """
    if not prompt.strip():
        logger.warning("Empty prompt received.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        response = await asyncio.to_thread(qa.run, prompt)
        logger.info("Successfully processed query.")
    except Exception as e:
        logger.exception("Error processing the query.")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

    return {"response": response}


@app.get("/health", summary="Health Check")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "OK"}
