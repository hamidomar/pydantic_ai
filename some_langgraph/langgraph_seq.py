from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI


# -------------------------------
# Shared State
# -------------------------------
class State(dict):
    pass


# -------------------------------
# Tools
# -------------------------------
def count_words(text: str) -> dict:
    return {"word_count": len(text.split())}

tools = {
    "count_words": count_words
}

tool_node = ToolNode(tools)


# -------------------------------
# LLM
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")


# -------------------------------
# Nodes
# -------------------------------
def summarize(state: State):
    """Use LLM to summarize input text."""
    text = state["input_text"]
    summary = llm.invoke(f"Summarize this in 1–2 sentences:\n\n{text}").content
    return {"summary": summary}


def count_summary_words(state: State):
    """Send summary to the tool node."""
    return {
        "tool": {
            "tool_name": "count_words",
            "args": state["summary"]
        }
    }


def final_answer(state: State):
    summary = state["summary"]
    count = state["word_count"]

    answer = llm.invoke(
        f"Here is a summary: {summary}\n"
        f"It contains {count} words.\n"
        "Write a clean final answer."
    ).content

    return {"final": answer}


# -------------------------------
# Build Graph
# -------------------------------
graph = StateGraph(State)

graph.add_node("summarize", summarize)
graph.add_node("count_words", tool_node)
graph.add_node("final_answer", final_answer)

graph.set_entry_point("summarize")
graph.add_edge("summarize", "count_words")
graph.add_edge("count_words", "final_answer")
graph.add_edge("final_answer", END)

app = graph.compile(checkpointer=MemorySaver())


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    result = app.run({"input_text": "LangGraph helps create agentic workflows."})
    print(result)
