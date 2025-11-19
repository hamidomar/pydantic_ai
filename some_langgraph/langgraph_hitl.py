# file: langgraph_human_in_loop.py
# Minimal human-in-the-loop LangGraph example
# pip install langgraph langchain openai

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI

# --- Simple document store (replace with real retriever/vectorstore) ---
DOCS = [
    "LangGraph is a state-machine-first orchestration library for LLM workflows.",
    "Use StateGraph.add_node(...) to add nodes which receive and return dict states.",
    "LangGraph supports synchronous and asynchronous execution and can be combined with LangChain retrievers."
]

# --- Node implementations ---
def retrieve_fn(state: dict):
    """
    Retrieve relevant docs for a question.
    Replace with a vectorstore retriever for real RAG.
    """
    query = state.get("question", "")
    # naive "retrieval": return docs that contain any keyword from query
    hits = [d for d in DOCS if any(tok.lower() in d.lower() for tok in query.split() if len(tok) > 3)]
    # fallback: return top-N if nothing matched
    if not hits:
        hits = DOCS[:2]
    state["retrieved_docs"] = hits
    return {"retrieved_docs": hits}

def generate_fn(state: dict):
    """
    Call LLM to produce an answer using retrieved docs and the question.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # swap model as needed
    question = state["question"]
    context = "\n\n".join(state.get("retrieved_docs", []))
    prompt = (
        "You are a helpful assistant. Use the context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer succinctly."
    )
    # langchain ChatOpenAI supports __call__ or .generate; using .invoke style safe fallback
    resp = llm.call_as_llm if hasattr(llm, "call_as_llm") else llm
    # simple call (this shape works for many LangChain chat models)
    answer = llm(prompt) if callable(llm) else llm.generate([prompt])
    # handle both return shapes:
    if isinstance(answer, str):
        text = answer
    else:
        # try to extract content from typical LangChain ChatModel response
        try:
            text = answer.generations[0][0].text
        except Exception:
            text = str(answer)
    state["draft_answer"] = text.strip()
    return {"answer": state["draft_answer"]}

def finalize_fn(state: dict):
    """Final node that returns the final accepted answer."""
    return {"final_answer": state.get("final_answer", state.get("draft_answer"))}

# --- Build the graph: basic retrieve -> generate -> finalize (we control human loop outside) ---
graph = StateGraph(dict)
graph.add_node("retrieve", retrieve_fn)
graph.add_node("generate", generate_fn)
graph.add_node("finalize", finalize_fn)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "finalize")
graph.add_edge("finalize", END)

# --- Orchestration: run graph, ask human, optionally revise and re-run ---
def human_in_the_loop(question: str):
    state = {"question": question}
    while True:
        # run the graph: this will execute retrieve -> generate -> finalize
        result = graph.run(state)
        draft = result.get("draft_answer") or result.get("final_answer") or "<no answer>"
        print("\n--- LLM DRAFT ANSWER ---\n")
        print(draft)
        print("\n--- HUMAN REVIEW ---")
        choice = input("Approve (a) / Edit (e) / Retry retrieval (r) / Quit (q): ").strip().lower()

        if choice == "a":
            # accept answer and finalize
            state["final_answer"] = draft
            final = graph.run(state).get("final_answer")
            print("\n--- FINAL ANSWER (accepted) ---\n")
            print(final)
            return final
        elif choice == "e":
            # ask the human to provide an edited answer (human becomes oracle)
            edited = input("Paste edited answer (or type new instructions for the model):\n")
            # if the human provided new answer, take it as final
            state["final_answer"] = edited
            final = graph.run(state).get("final_answer")
            print("\n--- FINAL ANSWER (human edited) ---\n")
            print(final)
            return final
        elif choice == "r":
            # ask for a refinement of the question / retrieval hints and loop
            hint = input("Enter retrieval hint or revised question (leave blank to repeat):\n").strip()
            if hint:
                state["question"] = hint
            # loop will re-run retrieve/generate with updated question
            print("Re-running retrieval/generation...\n")
            continue
        elif choice == "q":
            print("Aborting. No final answer produced.")
            return None
        else:
            print("Unknown option - please choose a/e/r/q. Re-running for convenience.\n")
            continue

# --- Example usage ---
if __name__ == "__main__":
    q = input("Enter your question about LangGraph: ").strip()
    human_in_the_loop(q)
