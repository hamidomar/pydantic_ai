# minimal_langgraph_tool_demo.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Simple shared state
class State(dict):
    pass

# --- Tool implementation (classic) ---
# A trivial tool that "transforms" text (here: uppercasing)
def to_upper(text: str) -> dict:
    # Tool returns a dict that will be merged into state
    return {"translated": text.upper()}

# Register tools in a ToolNode
tools = {"to_upper": to_upper}
tool_node = ToolNode(tools)

# --- Graph nodes ---
def provide_text(state: State):
    # just pass the input text into state under "text"
    return {"text": state["input_text"]}

def call_tool(state: State):
    # Ask the ToolNode to run the "to_upper" tool with the provided text
    return {"tool": {"tool_name": "to_upper", "args": state["text"]}}

def final_node(state: State):
    # Tool injected "translated" into state; return final shape
    return {"result": state["translated"]}

# --- Build graph ---
graph = StateGraph(State)
graph.add_node("provide_text", provide_text)
graph.add_node("call_tool", tool_node)   # tool node plugged in
graph.add_node("final", final_node)

graph.set_entry_point("provide_text")
graph.add_edge("provide_text", "call_tool")
graph.add_edge("call_tool", "final")
graph.add_edge("final", END)

app = graph.compile()

# --- Run example ---
if __name__ == "__main__":
    out = app.run({"input_text": "hello langgraph tools!"})
    print(out)
