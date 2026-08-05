from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Create the LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.1,
   
)

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
    response = chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })
    return response.content