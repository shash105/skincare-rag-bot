import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv(override=True)

# Debugging: Print the API key status
#st.write("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
#st.write("API key starts with:", os.getenv("OPENAI_API_KEY", "")[:7])
#st.write("API key ends with:", os.getenv("OPENAI_API_KEY", "")[-4:])
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

# chat history

if "messages" not in st.session_state:
    st.session_state.messages = []

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

# displaying chat history
for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.write(messages["content"])

# user question
question = st.chat_input("Ask a skincare question...")

# generating answer

if question:

    # adding user's message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # displaying user's message
    with st.chat_message("user"):
        st.write(question)

    
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
    answer = response.content

    # adding assistant reponse to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # displaying assistant's response
    with st.chat_message("assistant"):
        st.write(answer)

        #sources
        st.subheader("Sources")
        for i, document in enumerate(results):
            source = document.metadata.get(
                "source",
                "Unknown source")
            page = document.metadata.get(
                "page",
                "Unknown page")
            st.write(
                f"{i + 1}. {source} (Page: {page})"
            )
