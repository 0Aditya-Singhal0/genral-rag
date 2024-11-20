import uuid
import asyncio
import weaviate
from enum import Enum
from typing import List
from logger import logger
from config import settings

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import ChatCohere, CohereEmbeddings
from fastapi.security.api_key import APIKeyHeader, APIKey
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate as LangchainWeaviate
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status

# Initialize Weaviate Client
weaviate_client = weaviate.Client(settings.WEAVIATE_URL)


# Exception Handling
class WeaviateConnectionError(Exception):
    pass


# Define Enums and Models
class IdentifierType(str, Enum):
    filename = "filename"
    uuid = "uuid"


class RemoveFilesRequest(BaseModel):
    identifiers: List[str] = Field(...)
    identifier_type: IdentifierType


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
            {
                "name": "short_summary",
                "dataType": ["text"],
                "description": "A brief summary of the document",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True,  # Skip vectorization for this property
                        "vectorizePropertyName": False,
                    }
                },
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


def get_embeddings():
    if settings.EMBEDDING_PROVIDER.lower() == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for OpenAI embeddings.")
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
    elif settings.EMBEDDING_PROVIDER.lower() == "cohere":
        if not settings.COHERE_API_KEY:
            raise ValueError("Cohere API key is required for Cohere embeddings.")
        return CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_EMBEDDING_MODEL,
        )
    elif settings.EMBEDDING_PROVIDER.lower() == "local":
        return
    else:
        logger.error(f"Unsupported Embedding provider: {settings.EMBEDDING_PROVIDER}")
        raise ValueError(
            f"Unsupported Embedding provider: {settings.EMBEDDING_PROVIDER}"
        )


def get_vector_store() -> LangchainWeaviate:
    return LangchainWeaviate(
        client=weaviate_client,
        index_name="Document",
        text_key="text",
        attributes=["filename", "short_summary"],
        embedding=get_embeddings(),
    )


def get_llm() -> ChatOpenAI | ChatCohere:
    if settings.LLM_PROVIDER.lower() == "openai":
        return ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
            model=settings.OPENAI_MODEL,
        )
    elif settings.LLM_PROVIDER.lower() == "cohere":
        return ChatCohere(
            cohere_api_key=settings.COHERE_API_KEY,
            temperature=0,
            model=settings.COHERE_MODEL,
        )
    else:
        logger.error(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def get_retrieval_qa(
    llm: ChatOpenAI | ChatCohere = Depends(get_llm),
    vector_store: LangchainWeaviate = Depends(get_vector_store),
) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )


# Initialize FastAPI app
app = FastAPI(
    title="Weaviate RAG Application",
    description="A production-ready FastAPI application integrating Weaviate, LangChain, and OpenAI for Retrieval-Augmented Generation (RAG).",
    version="1.0.0",
)

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size as needed
        chunk_overlap=200,  # Adjust overlap as needed
    )

    for file in files:
        content = await file.read()
        if not content:
            logger.warning(f"File {file.filename} is empty.")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"File {file.filename} is not a valid UTF-8 text file.")
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a valid UTF-8 text file.",
            )
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(chunk)
            metadata.append({"filename": file.filename})
            ids.append(str(uuid.uuid4()))

    vector_store = get_vector_store()

    try:
        await asyncio.to_thread(
            vector_store.add_texts, texts=documents, ids=ids, metadatas=metadata
        )
        logger.info(f"Successfully added {len(files)} files to the vector store.")
    except Exception as e:
        logger.exception(f"Error adding files to the vector store -> {e}")
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
        logger.exception(
            f"Weaviate exception occurred during file removal -> {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Failed to remove files from the vector store."
        )
    except Exception as e:
        logger.exception(f"Unexpected error during file removal -> {e}")
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

    # Inspect and log the context (you can modify based on what you're sending)
    retriever_context = qa.retriever.get_relevant_documents(prompt)
    logger.info(f"Context retrieved: {retriever_context}")

    try:
        response = await asyncio.to_thread(qa.run, prompt)
        logger.info("Successfully processed query.")
    except Exception as e:
        logger.exception(f"Error processing the query -> {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

    return {"response": response}


@app.get("/health", summary="Health Check")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "OK"}
