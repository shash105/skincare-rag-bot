import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv(override=True)

# page setup

st.set_page_config(
    page_title="Skincare Ingredient Bot",
    page_icon="🧴"
)

st.title("🧴 Skincare Ingredient Bot")

st.write(
    "Ask questions about skincare ingredients based on "
    "the information in the provided documents."
)

# connecting to the vector db

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# LLM creation

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# user question
question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What does niacinamide do"
)

# generating answer

if question:
    results = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content for document in results
    )

    prompt = f"""
You are a skincare information assistant. 

Answer the question using ONLY the information provided in the context below. 

If the answer cannot be found in the context, 
say that you couldn't find the information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    st.subheader("Answer")
    st.write(response.content)
