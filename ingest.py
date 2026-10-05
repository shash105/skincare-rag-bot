from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# loading API key from .env file
load_dotenv(override=True)

# Debugging: Print the API key status
#print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
#print("API key starts with:", os.getenv("OPENAI_API_KEY", "")[:7])
#print("API key ends with:", os.getenv("OPENAI_API_KEY", "")[-4:)


# made ingest.py reuasable by creating a function to create the vector database
def create_vector_database():

    pdf_files = list(Path("data").glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Embeddings created and stored successfully!")