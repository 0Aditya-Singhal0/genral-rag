import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
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

# Initialize Weaviate Client
weaviate_client = weaviate.Client(url=WEAVIATE_URL)


# Define the schema if not already present
def define_weaviate_schema():
    schema = {
        "classes": [
            {
                "class": "Document",
                "description": "A class to hold documents for RAG",
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Content of the document",
                    },
                    {
                        "name": "filename",
                        "dataType": ["string"],
                        "description": "Name of the file",
                    },
                ],
            }
        ]
    }

    existing_schema = weaviate_client.schema.get()
    if "classes" not in existing_schema:
        weaviate_client.schema.create(schema)
    else:
        classes = existing_schema["classes"]
        class_names = [cls["class"] for cls in classes]
        if "Document" not in class_names:
            weaviate_client.schema.create(schema)


define_weaviate_schema()

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize VectorStore
vector_store = Weaviate(
    client=weaviate_client,
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
        # Here, you should process the content to extract text.
        # For simplicity, assuming the file is a text file.
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
