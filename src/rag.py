from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from src.llm import llm
import tempfile
import os

# 1. Embeddings model (free, runs locally)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # ADD THIS CHECK
    if len(chunks) == 0:
        os.unlink(tmp_path)
        raise ValueError("No text found in PDF. Make sure it's not a scanned image PDF.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)

    return vector_store

def get_rag_response(vector_store, chat_history, user_input):
    # 5. Find relevant chunks
    relevant_docs = vector_store.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 6. Build prompt with context
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a helpful AI Study Assistant.
Answer questions based on the following context from the PDF.
If the answer is not in the context, say 'I could not find this in the PDF.'

CONTEXT:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    return response.content

def summarize_pdf(vector_store):
    # Get a broad sample of chunks from the document
    relevant_docs = vector_store.similarity_search(
        "main topics overview summary", 
        k=10
    )
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a helpful AI Study Assistant.
Summarize the following document content clearly and concisely.
Structure your summary with:
- Main Topic
- Key Points (bullet points)
- Important Concepts
- Conclusion

DOCUMENT CONTENT:
{context}"""),
        ("human", "Please provide a comprehensive summary of this document.")
    ])

    chain = prompt | llm
    response = chain.invoke({})
    return response.content


def generate_quiz(vector_store, num_questions=5):
    relevant_docs = vector_store.similarity_search(
        "key concepts important facts definitions", 
        k=10
    )
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a helpful AI Study Assistant.
Generate exactly {num_questions} multiple choice questions from the document.

Format each question EXACTLY like this:
Q1: [Question here]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Answer: [Correct letter]
Explanation: [Why this is correct]

DOCUMENT CONTENT:
{context}"""),
        ("human", f"Generate {num_questions} multiple choice questions from this document.")
    ])

    chain = prompt | llm
    response = chain.invoke({})
    return response.content