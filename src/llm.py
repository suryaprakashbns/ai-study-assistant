import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def clean_response(text):
    if not isinstance(text, str):
        return str(text)
    if "<think>" in text:
        if "</think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        else:
            return "I had trouble forming a complete response — please try asking again."
    return text.strip()

# 1. Create LLM with multi-model fallbacks to handle Groq rate limits (8,000 TPM limit)
# If the primary model hits its token limit, it will automatically route to the fallback models.
primary_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
fallback_model_1 = "openai/gpt-oss-120b"
fallback_model_2 = "openai/gpt-oss-20b"

primary_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=primary_model,
    temperature=0.1,
    max_tokens=1024,
)

fallback_llm_1 = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=fallback_model_1,
    temperature=0.1,
    max_tokens=1024,
)

fallback_llm_2 = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=fallback_model_2,
    temperature=0.1,
    max_tokens=1024,
)

llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])

# 2. Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful AI Study Assistant.
Help students understand concepts clearly and simply.
When explaining, use examples and analogies."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 3. Create the chain
chain = prompt | llm

def get_ai_response(chat_history, user_input):
    # Only keep the last 6 messages (3 conversation turns) to avoid exceeding Groq TPM limits
    trimmed_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    response = chain.invoke({
        "chat_history": trimmed_history,
        "user_input": user_input
    })
    return clean_response(response.content)

