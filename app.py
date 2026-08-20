import os
import streamlit as st

# Bridge Streamlit Cloud's Secrets manager into an environment variable.
# This MUST run before any src/ imports, since those create the LLM client
# at import time and read the key immediately.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from langchain_core.messages import HumanMessage, AIMessage
from src.llm import get_ai_response
from src.rag import process_pdf, get_rag_response, summarize_pdf, generate_quiz
from src.agent import run_agent
from src.multi_agent import run_multi_agent

def initialize_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content) 

def handle_user_input():
    user_input = st.chat_input("Ask me anything...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.vector_store is not None:
                    response = get_rag_response(
                        st.session_state.vector_store,
                        st.session_state.chat_history,
                        user_input
                    )
                else:
                    response = get_ai_response(
                        st.session_state.chat_history,
                        user_input
                    )
                st.markdown(response)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

# UI
st.set_page_config(page_title="AI Study Assistant", page_icon="🎓", layout="wide")
st.title("🎓 AI Study Assistant")
st.caption("Your personal AI tutor — ask me anything!")

# Sidebar
with st.sidebar:
    st.header("📄 Upload Study Material")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Reading and processing PDF..."):
                st.session_state.vector_store = process_pdf(uploaded_file)
                st.session_state.chat_history = []
            st.success("PDF processed!")

    # Show features only when PDF is loaded
    if st.session_state.get("vector_store"):
        st.info("📚 PDF mode active")
        st.divider()

        # Summarize button
        if st.button("📝 Summarize PDF"):
            with st.spinner("Generating summary..."):
                summary = summarize_pdf(st.session_state.vector_store)
                st.session_state.chat_history.append(
                    HumanMessage(content="Please summarize this document.")
                )
                st.session_state.chat_history.append(
                    AIMessage(content=summary)
                )
            st.rerun()

        st.divider()

        # Quiz generation
        st.subheader("🧠 Generate Quiz")
        num_questions = st.slider("Number of questions", 3, 10, 5)
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz(st.session_state.vector_store, num_questions)
                st.session_state.chat_history.append(
                    HumanMessage(content=f"Generate a quiz with {num_questions} questions.")
                )
                st.session_state.chat_history.append(
                    AIMessage(content=quiz)
                )
            st.rerun()
    else:
        st.info("💬 Normal chat mode")
st.divider()
st.subheader("🤖 AI Agent Mode")
st.caption("Agent can search web, explain topics, generate quizzes")

agent_input = st.text_input("Ask the agent anything...")
if st.button("Run Agent"):
    if agent_input:
        with st.spinner("Agent thinking..."):
            agent_response = run_agent(agent_input)
        st.session_state.chat_history.append(
                HumanMessage(content=agent_input)
            )
        st.session_state.chat_history.append(
                AIMessage(content=agent_response)
            )
        st.rerun()     
st.divider()
st.subheader("🤖 Multi-Agent System")
st.caption("Supervisor routes your request to the right specialist agent")

multi_input = st.text_input("Ask the multi-agent system...")
if st.button("Run Multi-Agent"):
    if multi_input:
        with st.spinner("Multi-agent system thinking..."):
            multi_response = run_multi_agent(multi_input)
        
        st.session_state.chat_history.append(
            HumanMessage(content=f"[Multi-Agent] {multi_input}")
        )
        st.session_state.chat_history.append(
            AIMessage(content=multi_response)
        )
        st.rerun()        

initialize_chat()
display_chat_history()
handle_user_input()
