from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI

def call_llm(state):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    resp = llm.invoke(state["topic"])
    return {"text": resp.content}

graph = StateGraph(dict)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
