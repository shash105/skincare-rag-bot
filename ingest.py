from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# loading API key from .env file
load_dotenv()

# load PDF
loader = PyPDFLoader("data/niacinamide.pdf")
documents = loader.load()

print(f"Number of pages: {len(documents)}")

#splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

# calling the function to split into chunks
chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

# creating embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# storing embeddings in chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("Embeddings created and stored in Chroma.")