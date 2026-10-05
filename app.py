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
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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

    # rewrite questio using chat history
    conversation_history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in st.session_state.messages[:-1]
    )

    rewrite_prompt = f"""
    You are a helping a skincare question-answering system. 

    Look at the conversation history and the user's latest question.

    Rewrite the latest question into a standalone question that 
    can be understood without the conversation history.

    If the question is already clear and standalong, return it unchanged.

    Do not answer the question. Only return the rewritten question.

    Conversation history:
    {conversation_history}

    Latest question:
    {question}

    Standalone question:
    """

    rewritten_question = llm.invoke(rewrite_prompt).content.strip()
    st.caption(f"Rewritten question: {rewritten_question}")

    
    results_with_scores = vector_store.similarity_search_with_score(
        rewritten_question,
        k=3
    )

    SIMILARITY_THRESHOLD = 0.8
    results = [
        document
        for document, score in results_with_scores
        if score <= SIMILARITY_THRESHOLD
    ]


    if results:
        context = "\n\n".join(
        document.page_content for document in results
        )
    else:
        context = "No relevant information was found in the provided documents."

    prompt = f"""
    You are an informational skincare assistant.

    Your job is to answer the user's question using ONLY the information
    contained in the provided context.

    Follow these rules:
    1. Do not use outside knowledge or make up information.
    2. If the context does not contain enough information to answer the 
       question, say:
       "I couldn't find enough information to answer that in the provided documents."
    3. If the context only patially answers the question, clearly explain what information
       is supported by the documents and what information is not available.
    4. Do not diagnose skin conditions.
    5. Do not claim that a skincare ingredient will definitely treat or cure a medical condition.
    6. Keep the answer clear and easy to understand.
    7. For medical or safety-related questions. provide general information from
       the documents rather than personalised medical advice.

    Context:
    {context}

    Question:
    {rewritten_question}

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
        if results:

            with st.expander("Sources", expanded=False):

                seen_sources = set()
                for document in results:
                    source = document.metadata.get(
                        "source",
                        "Unknown source"
                    )

                    page = document.metadata.get(
                        "page",
                        None
                    )

                    source_name = source.split("\\")[-1]
                    if page is not None:
                        page = page + 1

                        source_key = (source_name, page)
                        if source_key in seen_sources:
                            continue

                    seen_sources.add(source_key)

                    if page is not None:
                        st.markdown(
                            f"**{source_name}** - Page {page}"
                        )
                    else:
                        st.markdown(
                            f"**{source_name}**"
                        )

            
            
