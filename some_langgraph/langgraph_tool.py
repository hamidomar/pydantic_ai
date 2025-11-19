# langgraph_tool_agent.py
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema import HumanMessage

# -------------------------
# 1) Agent state (typed)
# -------------------------
class AgentState(TypedDict):
    # messages is a list that should append (not replace) across nodes
    messages: Annotated[List, operator.add]


# -------------------------
# 2) Tools (classic examples)
# -------------------------
@tool
def get_weather(city: str) -> str:
    """Mock weather tool - replace with a real API in production."""
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 22°C"
    }
    return weather_data.get(city, "Weather data not available")

@tool
def calculate_sum(numbers: List[float]) -> float:
    """Return sum of numbers."""
    return sum(numbers)

@tool
def search_wikipedia(query: str) -> str:
    """Mock Wikipedia search result."""
    return f"Wikipedia search results for '{query}': This is sample content about {query}."

# register tools in a list
tools = [get_weather, calculate_sum, search_wikipedia]


# -------------------------
# 3) LLM init and bind to tools
# -------------------------
# Note: the article binds the tools to the LLM so it can emit tool_calls.
llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)   # allows the model to return tool_calls


# -------------------------
# 4) ToolNode wrapper
# -------------------------
tool_node = ToolNode(tools)


# -------------------------
# 5) Agent node (calls model)
# -------------------------
def call_model(state: AgentState):
    """
    The agent node sends the conversation messages to the LLM-with-tools.
    The LLM may return a tool call which the graph will route to the ToolNode.
    """
    messages = state["messages"]
    # If using LangChain message types, the model expects e.g. [HumanMessage(...)]
    # Here ensure we pass the messages the model expects.
    # If the messages are plain strings, adapt accordingly.
    response = llm_with_tools.invoke(messages)  # may emit tool_calls
    # wrap response back into messages list; keep appending
    return {"messages": [response]}


# -------------------------
# 6) Routing logic: send to tools if there's a tool call
# -------------------------
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    # Depending on model response object shape, check for tool calls attribute/name.
    # Example from the article: last_message.tool_calls
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# -------------------------
# 7) Build graph
# -------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

# conditional edge: when should_continue returns "tools", go to tools node; else END
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)

# after tools run, route back to agent so model can continue (tool results are merged into state)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# -------------------------
# 8) Runner helper
# -------------------------
def run_agent(query: str):
    # seed messages in a shape the model expects. Using HumanMessage to match examples.
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    result = app.invoke(initial_state)
    # The agent's final reply should be in the last messages element; adapt if your LLM returns differently.
    last = result["messages"][-1]
    # If last is an object, access .content; otherwise return string representation
    return getattr(last, "content", str(last))


# -------------------------
# 9) Example usage
# -------------------------
if __name__ == "__main__":
    queries = [
        "What's the weather like in New York?",
        "Calculate the sum of 10, 25, and 37.",
        "Search Wikipedia for information about artificial intelligence."
    ]

    for q in queries:
        print(f"Query: {q}")
        try:
            print("Response:", run_agent(q))
        except Exception as e:
            print("Error running agent (check LLM/tool bindings):", e)
        print("-" * 50)
