from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv(override=True)

# creating the same embedding model as in ingest.py when stroing the documents
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Connecting to the existing Chroma database
vector_store = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# creating a retriever from the vector store
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

#asking a question
question = "What does niacinamide do for the skin?"

# retrieving the 3 most relevant chunks
results = retriever.invoke(question)

print(f"Found {len(results)} relevant chunks.")

for i, result in enumerate(results):
    print(f"\nChunk {i+1}:")
    print(result.page_content)