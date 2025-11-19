# file: langgraph_stream_sync.py
from langgraph.graph import StateGraph, START, END
from langgraph.nodes import LLMNode    # LangGraph LLM node (high-level)
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# NOTE: APIs/names below mirror the LangGraph docs shape.
# If your install exposes slightly different names, adapt imports accordingly.

def build_graph():
    graph = StateGraph()
    # A simple LLM node that accepts `topic` and emits a story.
    llm = LLMNode(
        name="write_story",
        llm=ChatOpenAI(streaming=True, temperature=0.7, model="gpt-4o-mini"),  # streaming enabled
        prompt_template="Write a short, vivid story about: {topic}"
    )
    graph.add_node(START, "start")
    graph.add_node(llm, "write_story")
    graph.add_node(END, "end")
    graph.add_edge(START, llm)
    graph.add_edge(llm, END)
    return graph

if __name__ == "__main__":
    graph = build_graph()
    inputs = {"topic": "a lighthouse keeper who finds a secret map"}
    # stream_mode "updates" returns incremental updates (tokens / partial messages)
    for chunk in graph.stream(inputs, stream_mode="updates"):
        # chunk is usually a dict-like event describing what changed.
        # Print it raw so you can inspect structure (token, node, state, etc.)
        print("CHUNK:", chunk)
