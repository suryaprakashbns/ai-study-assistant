from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Literal
from dotenv import load_dotenv
import os

load_dotenv()

from src.llm import clean_response

# ============================================
# 1. SHARED LLM
# ============================================
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3.6-27b",
    temperature=0.1
)

# ============================================
# 2. TOOLS FOR EACH AGENT
# ============================================
search = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """Search the internet for current information."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def explain_topic(topic: str) -> str:
    """Explain any academic topic clearly with examples."""
    response = llm.invoke([
        HumanMessage(content=f"Explain {topic} clearly with examples. Be concise.")
    ])
    return clean_response(response.content)

@tool
def create_quiz(topic: str) -> str:
    """Create a multiple choice quiz about any topic."""
    response = llm.invoke([
        HumanMessage(content=f"""Generate 3 multiple choice questions about {topic}.
Format:
Q1: [Question]
A) B) C) D)
Answer: [Letter]""")
    ])
    return clean_response(response.content)

# ============================================
# 3. SPECIALIZED AGENTS
# ============================================
research_agent = create_react_agent(llm, [search_web])
study_agent = create_react_agent(llm, [explain_topic])
quiz_agent = create_react_agent(llm, [create_quiz])

# ============================================
# 4. GRAPH STATE
# ============================================
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_agent: str
    final_answer: str

# ============================================
# 5. SUPERVISOR NODE
# ============================================
def supervisor_node(state: AgentState) -> AgentState:
    user_message = state["messages"][-1].content.lower()

    print(f"\nSUPERVISOR: Analyzing request: {user_message}")

    if any(word in user_message for word in ["search", "news", "latest", "current", "find", "lookup"]):
        next_agent = "research"
    elif any(word in user_message for word in ["quiz", "test", "questions", "mcq", "practice"]):
        next_agent = "quiz"
    else:
        next_agent = "study"

    print(f"SUPERVISOR: Routing to {next_agent} agent")

    return {
        "messages": state["messages"],
        "next_agent": next_agent,
        "final_answer": ""
    }

# ============================================
# 6. AGENT NODES
# ============================================
def research_node(state: AgentState) -> AgentState:
    print("RESEARCH AGENT: Searching the web...")
    user_input = state["messages"][-1].content

    response = research_agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    final = clean_response(response["messages"][-1].content)
    return {
        "messages": state["messages"],
        "next_agent": state["next_agent"],
        "final_answer": final
    }

def study_node(state: AgentState) -> AgentState:
    print("STUDY AGENT: Explaining topic...")
    user_input = state["messages"][-1].content

    response = study_agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    final = clean_response(response["messages"][-1].content)
    return {
        "messages": state["messages"],
        "next_agent": state["next_agent"],
        "final_answer": final
    }

def quiz_node(state: AgentState) -> AgentState:
    print("QUIZ AGENT: Creating quiz...")
    user_input = state["messages"][-1].content

    response = quiz_agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    final = clean_response(response["messages"][-1].content)
    return {
        "messages": state["messages"],
        "next_agent": state["next_agent"],
        "final_answer": final
    }

# ============================================
# 7. ROUTING FUNCTION
# ============================================
def route_to_agent(state: AgentState) -> Literal["research", "study", "quiz"]:
    return state["next_agent"]

# ============================================
# 8. BUILD THE GRAPH
# ============================================
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research", research_node)
workflow.add_node("study", study_node)
workflow.add_node("quiz", quiz_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "research": "research",
        "study": "study",
        "quiz": "quiz"
    }
)
workflow.add_edge("research", END)
workflow.add_edge("study", END)
workflow.add_edge("quiz", END)

multi_agent = workflow.compile()

# ============================================
# 9. RUN FUNCTION
# ============================================
def run_multi_agent(user_input: str) -> str:
    try:
        print(f"\n{'='*50}")
        print(f"MULTI-AGENT SYSTEM STARTED")
        print(f"Input: {user_input}")
        print(f"{'='*50}")

        result = multi_agent.invoke({
            "messages": [HumanMessage(content=user_input)],
            "next_agent": "",
            "final_answer": ""
        })

        print(f"\nFINAL ANSWER: {result['final_answer'][:100]}")
        return clean_response(result["final_answer"])

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return f"Multi-agent error: {str(e)}"
