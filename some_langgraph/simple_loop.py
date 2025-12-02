"""
================================================================================
LANGGRAPH AGENTIC LOOP PATTERN DEMO
================================================================================

This script demonstrates the four core aspects of an agentic loop:

1. REPETITION
   - The agent repeats a task (research + refine) until quality is sufficient
   - LangGraph handles this via edges that loop back to earlier nodes

2. LOOP CONDITION
   - A conditional edge decides: continue looping or exit?
   - Based on state (e.g., quality score, iteration count)

3. STATE MANAGEMENT
   - State updates accumulate across iterations
   - Each loop iteration can read previous results and build on them

4. LOOP EXIT
   - When condition is met, graph transitions to END
   - Final state contains the accumulated result

================================================================================
THE SCENARIO
================================================================================

We'll build a "Research Refinement Agent" that:
- Takes a research question
- Generates an answer (using Vertex AI / Gemini)
- Evaluates the answer quality (using LLM-as-judge)
- If quality < threshold: refines and tries again
- If quality >= threshold OR max iterations reached: exits with final answer

This is a common agentic pattern: iterate until good enough.

================================================================================
AUTHENTICATION SETUP
================================================================================

Uses Google Application Default Credentials (ADC) with a service account.

Before running, set these environment variables:

    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    export GOOGLE_CLOUD_PROJECT="your-project-id"

================================================================================
"""

import os
import re
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ==============================================================================
# LLM SETUP (Vertex AI with Google ADC)
# ==============================================================================
#
# We create the LLM client once, reuse across nodes.
# Credentials are picked up automatically from GOOGLE_APPLICATION_CREDENTIALS.
#
# ==============================================================================

def create_llm():
    """
    Create the Vertex AI LLM client.
    
    Uses ADC - no explicit credentials in code.
    Just set GOOGLE_APPLICATION_CREDENTIALS env var.
    """
    return ChatVertexAI(
        model_name="gemini-1.5-flash",
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        temperature=0.7,
        max_tokens=1024,
    )


# Global LLM instance (created in main)
llm = None


# ==============================================================================
# STATE SCHEMA
# ==============================================================================
#
# State is the data that flows through the graph.
# Each node can read state and return updates.
#
# For an agentic loop, we typically track:
# - The task/input
# - Current result/output
# - Loop control variables (iteration count, quality score, etc.)
# - History of attempts (optional, for debugging/learning)
#
# ==============================================================================

class AgentState(TypedDict):
    """
    State for our research refinement agent.
    
    FIELDS:
    
    question: str
        The research question we're trying to answer.
        Set once at the start, doesn't change.
    
    current_answer: str
        The current best answer.
        Updated each iteration as we refine.
    
    feedback: str
        Critique from the evaluator.
        Used by generator to improve next iteration.
    
    quality_score: float
        How good is the current answer? (0.0 to 1.0)
        Evaluated after each generation.
        Loop continues until this meets threshold.
    
    iteration: int
        How many times have we looped?
        Used to prevent infinite loops (max iterations).
    
    refinement_history: list[str]
        History of all answers we've generated.
        Shows how the answer evolved.
        Useful for debugging and understanding the process.
    
    status: str
        Why did we exit? ("success" or "max_iterations")
        Set when loop terminates.
    """
    question: str
    current_answer: str
    feedback: str
    quality_score: float
    iteration: int
    refinement_history: list[str]
    status: str


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Quality threshold - loop exits when score >= this
QUALITY_THRESHOLD = 8.0  # Out of 10

# Maximum iterations - safety limit to prevent infinite loops
MAX_ITERATIONS = 3


# ==============================================================================
# NODE 1: GENERATE/REFINE ANSWER
# ==============================================================================
#
# This node does the "work" of the agent.
# First iteration: generates initial answer
# Subsequent iterations: refines based on feedback from evaluator
#
# REPETITION happens because the graph can route back to this node.
#
# ==============================================================================

def generate_answer(state: AgentState) -> AgentState:
    """
    Generate or refine an answer using the LLM.
    
    WHAT THIS DEMONSTRATES:
    - REPETITION: This node gets called multiple times
    - STATE MANAGEMENT: Reads previous answer + feedback, writes new answer
    
    FIRST ITERATION:
    - Generates initial answer to the question
    
    SUBSEQUENT ITERATIONS:
    - Uses feedback from evaluator to improve the answer
    """
    iteration = state["iteration"]
    question = state["question"]
    previous_answer = state.get("current_answer", "")
    feedback = state.get("feedback", "")
    
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration + 1}: Generating Answer")
    print(f"{'='*60}")
    
    if iteration == 0:
        # First iteration: generate initial answer
        print(f"Question: {question}")
        print("Action: Generating initial answer with LLM...")
        
        messages = [
            SystemMessage(content="""You are a knowledgeable research assistant. 
Provide a comprehensive, well-structured answer to the question.
Include specific details, examples, and evidence where relevant.
Keep your answer focused and around 200-300 words."""),
            HumanMessage(content=question)
        ]
    else:
        # Subsequent iterations: refine based on feedback
        print(f"Previous score: {state['quality_score']:.1f}/10")
        print(f"Feedback: {feedback[:100]}...")
        print("Action: Refining answer based on feedback...")
        
        messages = [
            SystemMessage(content="""You are a knowledgeable research assistant.
Your previous answer received feedback. Improve it based on the critique.
Make specific improvements addressing each point of feedback.
Keep the improved answer focused and around 200-300 words."""),
            HumanMessage(content=f"""Original question: {question}

Your previous answer:
{previous_answer}

Feedback received:
{feedback}

Please provide an improved answer that addresses the feedback:""")
        ]
    
    # Call the LLM
    response = llm.invoke(messages)
    new_answer = response.content
    
    print(f"\nGenerated answer:\n{'-'*40}")
    print(new_answer[:500] + "..." if len(new_answer) > 500 else new_answer)
    print(f"{'-'*40}")
    
    # ===== STATE UPDATE =====
    # Return only the fields we're changing
    # LangGraph merges this with existing state
    return {
        "current_answer": new_answer,
        "iteration": iteration + 1,
        "refinement_history": state.get("refinement_history", []) + [new_answer]
    }


# ==============================================================================
# NODE 2: EVALUATE QUALITY
# ==============================================================================
#
# This node uses LLM-as-judge to assess the answer quality.
# Sets the quality_score that the LOOP CONDITION will check.
# Also provides feedback for the next refinement iteration.
#
# ==============================================================================

def evaluate_quality(state: AgentState) -> AgentState:
    """
    Evaluate the quality of the current answer using LLM-as-judge.
    
    WHAT THIS DEMONSTRATES:
    - STATE MANAGEMENT: Reads current_answer, writes quality_score + feedback
    - Prepares data for LOOP CONDITION
    
    The LLM evaluates on multiple criteria:
    - Accuracy and completeness
    - Clarity and structure
    - Use of evidence/examples
    - Relevance to the question
    """
    iteration = state["iteration"]
    question = state["question"]
    current_answer = state["current_answer"]
    
    print(f"\n--- Evaluating Quality (LLM-as-Judge) ---")
    
    # LLM-as-judge prompt
    messages = [
        SystemMessage(content="""You are an expert evaluator assessing answer quality.

Evaluate the answer on these criteria:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-organized and easy to understand?
4. Evidence: Does it include relevant examples or details?

Provide:
1. A score from 1-10 (where 10 is excellent)
2. Specific feedback for improvement

Format your response EXACTLY like this:
SCORE: [number]
FEEDBACK: [your detailed feedback]"""),
        HumanMessage(content=f"""Question: {question}

Answer to evaluate:
{current_answer}

Please evaluate this answer:""")
    ]
    
    # Call the LLM for evaluation
    response = llm.invoke(messages)
    evaluation = response.content
    
    print(f"\nEvaluator response:\n{evaluation}")
    
    # Parse the score and feedback
    score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', evaluation)
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', evaluation, re.DOTALL)
    
    if score_match:
        quality_score = float(score_match.group(1))
    else:
        # Default score if parsing fails
        quality_score = 5.0
        print("Warning: Could not parse score, defaulting to 5.0")
    
    feedback = feedback_match.group(1).strip() if feedback_match else "No specific feedback provided."
    
    print(f"\n{'='*40}")
    print(f"Quality score: {quality_score:.1f}/10")
    print(f"Threshold: {QUALITY_THRESHOLD:.1f}/10")
    
    if quality_score >= QUALITY_THRESHOLD:
        print("✓ Quality threshold MET")
    else:
        print(f"✗ Below threshold, need {QUALITY_THRESHOLD - quality_score:.1f} more points")
    print(f"{'='*40}")
    
    return {
        "quality_score": quality_score,
        "feedback": feedback
    }


# ==============================================================================
# LOOP CONDITION: SHOULD WE CONTINUE?
# ==============================================================================
#
# This is the decision point of the agentic loop.
# Returns a string that determines the next node.
#
# LOOP CONDITION logic:
# - If quality >= threshold: exit (go to "finalize")
# - If iterations >= max: exit (go to "finalize")
# - Otherwise: continue (go back to "generate")
#
# ==============================================================================

def should_continue(state: AgentState) -> Literal["generate", "finalize"]:
    """
    Decide whether to continue looping or exit.
    
    WHAT THIS DEMONSTRATES:
    - LOOP CONDITION: The decision logic for the loop
    - Returns edge name ("generate" to loop, "finalize" to exit)
    
    TWO EXIT CONDITIONS:
    1. Quality threshold met (success!)
    2. Max iterations reached (give up, return best effort)
    
    The return value must match an edge name in the graph.
    """
    quality = state["quality_score"]
    iteration = state["iteration"]
    
    print(f"\n--- Loop Decision ---")
    print(f"Quality: {quality:.2f} (threshold: {QUALITY_THRESHOLD:.2f})")
    print(f"Iteration: {iteration} (max: {MAX_ITERATIONS})")
    
    # Check exit conditions
    if quality >= QUALITY_THRESHOLD:
        print("Decision: EXIT - Quality threshold reached!")
        return "finalize"
    
    if iteration >= MAX_ITERATIONS:
        print("Decision: EXIT - Max iterations reached")
        return "finalize"
    
    print("Decision: CONTINUE - Need more refinement")
    return "generate"


# ==============================================================================
# NODE 3: FINALIZE
# ==============================================================================
#
# This node runs when the loop exits.
# Sets final status and prepares the result.
#
# LOOP EXIT happens when we reach this node.
#
# ==============================================================================

def finalize(state: AgentState) -> AgentState:
    """
    Finalize the result after loop exits.
    
    WHAT THIS DEMONSTRATES:
    - LOOP EXIT: This node is reached when loop condition says "stop"
    - Sets final status based on why we exited
    """
    quality = state["quality_score"]
    iteration = state["iteration"]
    
    print(f"\n{'='*60}")
    print("FINALIZING RESULT")
    print(f"{'='*60}")
    
    if quality >= QUALITY_THRESHOLD:
        status = "success"
        print(f"✓ Success! Achieved quality {quality:.2f} in {iteration} iterations")
    else:
        status = "max_iterations"
        print(f"⚠ Max iterations reached. Best quality: {quality:.2f}")
    
    return {
        "status": status
    }


# ==============================================================================
# BUILD THE GRAPH
# ==============================================================================
#
# Graph structure:
#
#     START
#       |
#       v
#   [generate] <----+
#       |           |
#       v           |
#   [evaluate]      |
#       |           |
#       v           |
#   {decision}------+  (if quality < threshold AND iterations < max)
#       |
#       v (if quality >= threshold OR iterations >= max)
#   [finalize]
#       |
#       v
#      END
#
# The loop is created by the conditional edge from evaluate -> generate
#
# ==============================================================================

def build_agent_graph():
    """
    Build the agentic loop graph.
    
    KEY LANGGRAPH CONCEPTS:
    
    1. add_node(): Adds a node (function) to the graph
    
    2. add_edge(): Unconditional edge (always go from A to B)
    
    3. add_conditional_edges(): The magic for loops!
       - Takes a condition function
       - Function returns a string
       - String maps to next node name
       - This is how we implement LOOP CONDITION
    """
    # Create graph builder with our state schema
    builder = StateGraph(AgentState)
    
    # ===== ADD NODES =====
    # These are the "actions" the agent can take
    
    builder.add_node("generate", generate_answer)   # Generate/refine answer
    builder.add_node("evaluate", evaluate_quality)  # Evaluate quality
    builder.add_node("finalize", finalize)          # Final processing
    
    # ===== ADD EDGES =====
    
    # START -> generate (begin with generating an answer)
    builder.add_edge(START, "generate")
    
    # generate -> evaluate (always evaluate after generating)
    builder.add_edge("generate", "evaluate")
    
    # ===== THE LOOP: CONDITIONAL EDGE =====
    #
    # This is where the loop magic happens!
    #
    # add_conditional_edges takes:
    # - source node: "evaluate"
    # - condition function: should_continue
    # - path map: {return_value: next_node}
    #
    # When should_continue returns "generate" -> loop back
    # When should_continue returns "finalize" -> exit loop
    #
    builder.add_conditional_edges(
        "evaluate",           # After evaluation...
        should_continue,      # ...run this function...
        {
            "generate": "generate",   # If returns "generate", go to generate node
            "finalize": "finalize"    # If returns "finalize", go to finalize node
        }
    )
    
    # finalize -> END (we're done)
    builder.add_edge("finalize", END)
    
    # Compile the graph
    graph = builder.compile()
    
    return graph


# ==============================================================================
# DEMONSTRATION
# ==============================================================================

def main():
    """
    Run the agentic loop demonstration.
    
    WHAT YOU'LL SEE:
    
    1. REPETITION: generate node called multiple times
    2. LOOP CONDITION: decision made after each evaluation
    3. STATE MANAGEMENT: answer improves across iterations
    4. LOOP EXIT: finalize node called when done
    """
    global llm  # Use global LLM instance
    
    print("\n" + "="*70)
    print("LANGGRAPH AGENTIC LOOP PATTERN DEMO")
    print("="*70)
    print(f"""
    SCENARIO: Research Refinement Agent (with Vertex AI / Gemini)
    
    The agent will:
    1. Generate an answer to a research question (LLM call)
    2. Evaluate the answer quality (LLM-as-judge)
    3. If quality < {QUALITY_THRESHOLD}/10: refine and try again
    4. If quality >= {QUALITY_THRESHOLD}/10 OR iterations >= {MAX_ITERATIONS}: exit
    
    This demonstrates the agentic loop pattern:
    - REPETITION: Same nodes execute multiple times
    - LOOP CONDITION: Conditional edge decides continue/exit
    - STATE MANAGEMENT: State accumulates across iterations
    - LOOP EXIT: Graph terminates when condition met
    """)
    
    # ===== INITIALIZE LLM =====
    print("-"*70)
    print("Initializing Vertex AI LLM (using Google ADC)...")
    print("-"*70)
    
    # Check environment
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not set")
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("WARNING: GOOGLE_CLOUD_PROJECT not set")
    
    llm = create_llm()
    print(f"✓ LLM initialized: {llm.model_name}")
    
    # Build the graph
    graph = build_agent_graph()
    
    # Initial state
    initial_state = {
        "question": "What are the key factors that led to the success of the Apollo 11 mission?",
        "current_answer": "",
        "feedback": "",
        "quality_score": 0.0,
        "iteration": 0,
        "refinement_history": [],
        "status": ""
    }
    
    print("-"*70)
    print("STARTING AGENT LOOP")
    print("-"*70)
    
    # Run the graph
    # This will loop until exit condition is met
    final_state = graph.invoke(initial_state)
    
    # ===== DISPLAY RESULTS =====
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print(f"\nStatus: {final_state['status']}")
    print(f"Total iterations: {final_state['iteration']}")
    print(f"Final quality score: {final_state['quality_score']:.2f}")
    
    print(f"\n--- Final Answer ---")
    print(final_state['current_answer'])
    
    print(f"\n--- Refinement History ({len(final_state['refinement_history'])} versions) ---")
    for i, answer in enumerate(final_state['refinement_history']):
        print(f"\nVersion {i+1}:")
        print(f"  {answer[:100]}..." if len(answer) > 100 else f"  {answer}")
    
    # ===== RECAP =====
    
    print("\n" + "="*70)
    print("PATTERN RECAP")
    print("="*70)
    print(f"""
    1. REPETITION:
       └── 'generate' node was called {final_state['iteration']} times
    
    2. LOOP CONDITION:
       └── 'should_continue' checked quality after each evaluation
       └── Returned "generate" to loop, "finalize" to exit
    
    3. STATE MANAGEMENT:
       └── quality_score: 0.0 -> {final_state['quality_score']:.2f}
       └── iteration: 0 -> {final_state['iteration']}
       └── refinement_history grew with each iteration
    
    4. LOOP EXIT:
       └── Status: {final_state['status']}
       └── Triggered by: {"quality threshold met" if final_state['status'] == 'success' else "max iterations reached"}
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()