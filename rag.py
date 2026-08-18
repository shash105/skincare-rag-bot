from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv(override=True)

# creating embeddings 
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# connecting to existing Chroma database
vector_store = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# creating retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# creating the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# asking a question
question = "What does niacinamide do for the skin?"

# retrieving relevant chunks
results = retriever.invoke(question)

# combining the chunks into a single context
context = "\n\n".join(
    document.page_content for document in results
)

# creating the prompt
prompt = f"""
You are a skincare information assistant.

Answer the question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say: "I couldn't fund that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

# sending the prompt to GPT
response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)
