from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Initialize the model - will automatically use ADC
llm = ChatVertexAI(
    model="gemini-2.5-flash",
    project="YOUR_PROJECT_ID",
    location="us-central1",  # or your preferred region
    temperature=0
)

# Define your state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create your graph
graph_builder = StateGraph(State)

# Add your nodes and edges
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()