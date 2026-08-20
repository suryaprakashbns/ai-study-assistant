from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
       model="qwen/qwen3.6-27b",
    temperature=0.1,
    
)

search = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """Search the internet for current information about any topic or news."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def explain_topic(topic: str) -> str:
    """Explain any academic or study topic clearly with examples."""
    from src.llm import llm as base_llm
    from langchain_core.messages import HumanMessage
    try:
        response = base_llm.invoke([
            HumanMessage(content=f"Explain {topic} clearly with examples. Be concise.")
        ])
        return response.content
    except Exception as e:
        return f"Explanation failed: {str(e)}"

@tool
def create_quiz(topic: str) -> str:
    """Create a 3 question multiple choice quiz about any topic."""
    from src.llm import llm as base_llm
    from langchain_core.messages import HumanMessage
    try:
        response = base_llm.invoke([
            HumanMessage(content=f"""Generate 3 multiple choice questions about {topic}.
Format:
Q1: [Question]
A) B) C) D)
Answer: [Letter]""")
        ])
        return response.content
    except Exception as e:
        return f"Quiz generation failed: {str(e)}"

tools = [search_web, explain_topic, create_quiz]

agent_executor = create_react_agent(llm, tools)

def run_agent(user_input: str) -> str:
    try:
        print(f"\n{'='*50}")
        print(f"USER: {user_input}")
        print(f"{'='*50}")

        response = agent_executor.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        messages = response["messages"]
        print(f"\nTotal messages: {len(messages)}")

        for i, msg in enumerate(messages):
            print(f"\nMessage {i}: {type(msg).__name__}")
            if hasattr(msg, "content") and msg.content:
                print(f"Content: {str(msg.content)[:200]}")

        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                    return msg.content

        return messages[-1].content

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return f"Agent error: {str(e)}"
    
    
    
    
    
    
    
    
    
    
    
    
    
