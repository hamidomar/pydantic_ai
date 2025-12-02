"""
================================================================================
LANGGRAPH AGENTIC ORCHESTRATOR PATTERN DEMO
================================================================================

This script demonstrates the four core aspects of an agentic orchestrator:

1. WORKFLOW COORDINATION
   - An orchestrator agent coordinates multiple specialist agents
   - Each specialist handles a specific subtask
   - Orchestrator decides which agents to invoke and in what order

2. COMMUNICATION
   - Agents share data through a common state object
   - Each agent reads inputs from state and writes outputs to state
   - Clear contracts: what each agent expects and produces

3. STATE MANAGEMENT
   - Central state tracks the overall workflow progress
   - Accumulates results from each agent
   - Tracks which steps have completed

4. ERROR HANDLING
   - Graceful handling when individual agents fail
   - Retry logic for transient failures
   - Fallback strategies when agents can't complete their task

================================================================================
THE SCENARIO
================================================================================

We'll build a "Research Report Orchestrator" that coordinates three specialists:

1. RESEARCHER AGENT
   - Takes a topic and generates key findings
   - Outputs: research_findings

2. ANALYST AGENT  
   - Takes research findings and provides analysis/insights
   - Outputs: analysis

3. WRITER AGENT
   - Takes findings + analysis and produces a final report
   - Outputs: final_report

The ORCHESTRATOR:
- Decides the workflow sequence
- Routes data between agents
- Handles failures gracefully

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
import time
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ==============================================================================
# LLM SETUP (Vertex AI with Google ADC)
# ==============================================================================

def create_llm():
    """
    Create the Vertex AI LLM client.
    
    Uses ADC - credentials picked up from GOOGLE_APPLICATION_CREDENTIALS.
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
# The state is the CENTRAL HUB for communication between agents.
# 
# WORKFLOW COORDINATION: State tracks which steps are done
# COMMUNICATION: Agents read/write to shared state fields
# STATE MANAGEMENT: All workflow data lives here
# ERROR HANDLING: Error info stored in state for recovery
#
# ==============================================================================

class OrchestratorState(TypedDict):
    """
    Central state for the orchestrator workflow.
    
    WORKFLOW FIELDS:
    
    topic: str
        The research topic (input to the workflow)
    
    research_findings: str
        Output from Researcher Agent
        Input to Analyst Agent and Writer Agent
    
    analysis: str
        Output from Analyst Agent
        Input to Writer Agent
    
    final_report: str
        Output from Writer Agent
        Final deliverable of the workflow
    
    COORDINATION FIELDS:
    
    current_step: str
        Which step are we on? ("research", "analyze", "write", "done")
        Used by orchestrator to track progress
    
    completed_steps: list[str]
        Which steps have successfully completed?
        Useful for resuming after failures
    
    ERROR HANDLING FIELDS:
    
    error: str
        If an agent fails, error message stored here
    
    retry_count: int
        How many times have we retried the current step?
    
    workflow_status: str
        Overall status: "in_progress", "completed", "failed"
    """
    # Input
    topic: str
    
    # Agent outputs (COMMUNICATION happens through these)
    research_findings: str
    analysis: str
    final_report: str
    
    # Workflow tracking (COORDINATION)
    current_step: str
    completed_steps: list[str]
    
    # Error handling
    error: str
    retry_count: int
    workflow_status: str


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MAX_RETRIES = 2  # How many times to retry a failed agent


# ==============================================================================
# SPECIALIST AGENT 1: RESEARCHER
# ==============================================================================
#
# The Researcher agent takes a topic and produces key findings.
#
# COMMUNICATION:
# - Reads: topic
# - Writes: research_findings
#
# ==============================================================================

def researcher_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Researcher Agent: Gathers information on a topic.
    
    WHAT THIS DEMONSTRATES:
    - COMMUNICATION: Reads 'topic' from state, writes 'research_findings'
    - ERROR HANDLING: Wrapped in try/except, errors stored in state
    
    In a real system, this might:
    - Search the web
    - Query databases
    - Call external APIs
    """
    topic = state["topic"]
    
    print(f"\n{'='*60}")
    print("RESEARCHER AGENT")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print("Action: Researching topic...")
    
    try:
        messages = [
            SystemMessage(content="""You are a research specialist. 
Your job is to gather key facts, data points, and findings about a topic.

Provide 4-5 key findings, each with specific details.
Format as a numbered list.
Focus on factual, verifiable information."""),
            HumanMessage(content=f"Research this topic and provide key findings: {topic}")
        ]
        
        response = llm.invoke(messages)
        research_findings = response.content
        
        print(f"\nResearch findings:\n{'-'*40}")
        print(research_findings[:500] + "..." if len(research_findings) > 500 else research_findings)
        print(f"{'-'*40}")
        
        # ===== COMMUNICATION: Write output to state =====
        return {
            "research_findings": research_findings,
            "current_step": "analyze",
            "completed_steps": state.get("completed_steps", []) + ["research"],
            "error": "",  # Clear any previous error
            "retry_count": 0
        }
        
    except Exception as e:
        # ===== ERROR HANDLING: Store error in state =====
        print(f"ERROR in Researcher Agent: {e}")
        return {
            "error": f"Researcher failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1
        }


# ==============================================================================
# SPECIALIST AGENT 2: ANALYST
# ==============================================================================
#
# The Analyst agent takes research findings and provides insights.
#
# COMMUNICATION:
# - Reads: research_findings
# - Writes: analysis
#
# ==============================================================================

def analyst_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Analyst Agent: Analyzes research findings and provides insights.
    
    WHAT THIS DEMONSTRATES:
    - COMMUNICATION: Reads 'research_findings', writes 'analysis'
    - Depends on output from Researcher Agent
    """
    research_findings = state["research_findings"]
    topic = state["topic"]
    
    print(f"\n{'='*60}")
    print("ANALYST AGENT")
    print(f"{'='*60}")
    print(f"Analyzing findings on: {topic}")
    print("Action: Generating insights...")
    
    try:
        messages = [
            SystemMessage(content="""You are an analysis specialist.
Your job is to take research findings and provide deeper insights.

For each finding:
- Explain its significance
- Identify patterns or connections
- Note any implications

Be insightful and add value beyond the raw findings."""),
            HumanMessage(content=f"""Topic: {topic}

Research findings to analyze:
{research_findings}

Provide your analysis and insights:""")
        ]
        
        response = llm.invoke(messages)
        analysis = response.content
        
        print(f"\nAnalysis:\n{'-'*40}")
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
        print(f"{'-'*40}")
        
        # ===== COMMUNICATION: Write output to state =====
        return {
            "analysis": analysis,
            "current_step": "write",
            "completed_steps": state.get("completed_steps", []) + ["analyze"],
            "error": "",
            "retry_count": 0
        }
        
    except Exception as e:
        # ===== ERROR HANDLING =====
        print(f"ERROR in Analyst Agent: {e}")
        return {
            "error": f"Analyst failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1
        }


# ==============================================================================
# SPECIALIST AGENT 3: WRITER
# ==============================================================================
#
# The Writer agent produces the final report.
#
# COMMUNICATION:
# - Reads: topic, research_findings, analysis
# - Writes: final_report
#
# ==============================================================================

def writer_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Writer Agent: Produces the final report.
    
    WHAT THIS DEMONSTRATES:
    - COMMUNICATION: Reads from multiple agents' outputs
    - Synthesizes all previous work into final deliverable
    """
    topic = state["topic"]
    research_findings = state["research_findings"]
    analysis = state["analysis"]
    
    print(f"\n{'='*60}")
    print("WRITER AGENT")
    print(f"{'='*60}")
    print(f"Writing report on: {topic}")
    print("Action: Composing final report...")
    
    try:
        messages = [
            SystemMessage(content="""You are a professional report writer.
Your job is to synthesize research findings and analysis into a polished report.

Structure your report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (from research)
3. Analysis & Insights
4. Conclusions

Write in a clear, professional tone."""),
            HumanMessage(content=f"""Topic: {topic}

Research Findings:
{research_findings}

Analysis:
{analysis}

Please write the final report:""")
        ]
        
        response = llm.invoke(messages)
        final_report = response.content
        
        print(f"\nFinal Report:\n{'-'*40}")
        print(final_report[:800] + "..." if len(final_report) > 800 else final_report)
        print(f"{'-'*40}")
        
        # ===== COMMUNICATION: Write final output =====
        return {
            "final_report": final_report,
            "current_step": "done",
            "completed_steps": state.get("completed_steps", []) + ["write"],
            "workflow_status": "completed",
            "error": "",
            "retry_count": 0
        }
        
    except Exception as e:
        # ===== ERROR HANDLING =====
        print(f"ERROR in Writer Agent: {e}")
        return {
            "error": f"Writer failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1
        }


# ==============================================================================
# ERROR HANDLER NODE
# ==============================================================================
#
# This node handles errors from any agent.
# Implements retry logic and fallback strategies.
#
# ERROR HANDLING:
# - Checks retry count
# - Decides: retry or fail workflow
#
# ==============================================================================

def error_handler(state: OrchestratorState) -> OrchestratorState:
    """
    Error Handler: Decides how to handle agent failures.
    
    WHAT THIS DEMONSTRATES:
    - ERROR HANDLING: Centralized error recovery logic
    - Retry logic with max attempts
    - Graceful failure when retries exhausted
    """
    error = state["error"]
    retry_count = state["retry_count"]
    current_step = state["current_step"]
    
    print(f"\n{'='*60}")
    print("ERROR HANDLER")
    print(f"{'='*60}")
    print(f"Error: {error}")
    print(f"Current step: {current_step}")
    print(f"Retry count: {retry_count}/{MAX_RETRIES}")
    
    if retry_count <= MAX_RETRIES:
        print(f"Action: Will retry {current_step} step")
        # Keep current_step the same so we retry it
        return {
            "error": ""  # Clear error for retry
        }
    else:
        print("Action: Max retries exceeded, failing workflow")
        return {
            "workflow_status": "failed",
            "current_step": "failed"
        }


# ==============================================================================
# ORCHESTRATOR: ROUTING LOGIC
# ==============================================================================
#
# The orchestrator decides which agent to invoke next.
# This is the WORKFLOW COORDINATION logic.
#
# Routes based on:
# - Current step in the workflow
# - Whether there's an error to handle
#
# ==============================================================================

def orchestrator_router(state: OrchestratorState) -> Literal["researcher", "analyst", "writer", "error_handler", "end"]:
    """
    Orchestrator: Routes to the appropriate agent.
    
    WHAT THIS DEMONSTRATES:
    - WORKFLOW COORDINATION: Decides the flow of work
    - ERROR HANDLING: Routes to error handler when needed
    
    This is the "brain" of the orchestrator pattern.
    It examines state and decides what happens next.
    """
    current_step = state.get("current_step", "research")
    error = state.get("error", "")
    workflow_status = state.get("workflow_status", "in_progress")
    
    print(f"\n--- Orchestrator Decision ---")
    print(f"Current step: {current_step}")
    print(f"Error: {'Yes' if error else 'No'}")
    print(f"Status: {workflow_status}")
    
    # Check for errors first
    if error:
        print("Decision: Route to ERROR HANDLER")
        return "error_handler"
    
    # Check if workflow is done or failed
    if workflow_status in ["completed", "failed"] or current_step in ["done", "failed"]:
        print("Decision: Route to END")
        return "end"
    
    # Route based on current step
    if current_step == "research":
        print("Decision: Route to RESEARCHER")
        return "researcher"
    elif current_step == "analyze":
        print("Decision: Route to ANALYST")
        return "analyst"
    elif current_step == "write":
        print("Decision: Route to WRITER")
        return "writer"
    else:
        print("Decision: Unknown step, route to END")
        return "end"


# ==============================================================================
# BUILD THE ORCHESTRATOR GRAPH
# ==============================================================================
#
# Graph structure:
#
#                    START
#                      |
#                      v
#              +---------------+
#              |  ORCHESTRATOR |  (conditional routing)
#              +---------------+
#               /    |    |    \
#              v     v    v     v
#         [research] [analyst] [writer] [error_handler]
#              \     |    |     /
#               \    |    |    /
#                v   v    v   v
#              +---------------+
#              |  ORCHESTRATOR |  (loops back for next step)
#              +---------------+
#                      |
#                      v
#                     END
#
# ==============================================================================

def build_orchestrator_graph():
    """
    Build the orchestrator workflow graph.
    
    KEY CONCEPTS:
    
    1. WORKFLOW COORDINATION:
       - Orchestrator node routes to specialist agents
       - Conditional edges based on current_step
    
    2. COMMUNICATION:
       - All agents share the same state
       - Each agent reads/writes specific fields
    
    3. STATE MANAGEMENT:
       - State flows through the entire workflow
       - Accumulates outputs from each agent
    
    4. ERROR HANDLING:
       - Error handler node for recovery
       - Routing logic checks for errors
    """
    builder = StateGraph(OrchestratorState)
    
    # ===== ADD SPECIALIST AGENT NODES =====
    builder.add_node("researcher", researcher_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("error_handler", error_handler)
    
    # ===== ORCHESTRATOR ROUTING =====
    # From START, use orchestrator to decide first agent
    builder.add_conditional_edges(
        START,
        orchestrator_router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    # After each agent, route back through orchestrator
    # This creates the coordination loop
    
    builder.add_conditional_edges(
        "researcher",
        orchestrator_router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    builder.add_conditional_edges(
        "analyst",
        orchestrator_router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    builder.add_conditional_edges(
        "writer",
        orchestrator_router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    builder.add_conditional_edges(
        "error_handler",
        orchestrator_router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    # Compile
    graph = builder.compile()
    
    return graph


# ==============================================================================
# DEMONSTRATION
# ==============================================================================

def main():
    """
    Run the orchestrator pattern demonstration.
    
    WHAT YOU'LL SEE:
    
    1. WORKFLOW COORDINATION:
       - Orchestrator routes: research -> analyze -> write
       - Each agent called in sequence
    
    2. COMMUNICATION:
       - Researcher output feeds into Analyst
       - Both feed into Writer
       - All through shared state
    
    3. STATE MANAGEMENT:
       - State accumulates through workflow
       - Track completed_steps growing
    
    4. ERROR HANDLING:
       - If an agent fails, error_handler activates
       - Retry logic demonstrated
    """
    global llm
    
    print("\n" + "="*70)
    print("LANGGRAPH AGENTIC ORCHESTRATOR PATTERN DEMO")
    print("="*70)
    print("""
    SCENARIO: Research Report Orchestrator
    
    Three specialist agents coordinated by an orchestrator:
    
    1. RESEARCHER AGENT
       └── Gathers key findings on a topic
       
    2. ANALYST AGENT
       └── Analyzes findings, provides insights
       
    3. WRITER AGENT
       └── Synthesizes into final report
    
    ORCHESTRATOR coordinates the workflow:
    - Routes to appropriate agent based on current step
    - Handles errors with retry logic
    - Manages overall workflow state
    
    This demonstrates:
    - WORKFLOW COORDINATION: Orchestrator routes between agents
    - COMMUNICATION: Agents share data through state
    - STATE MANAGEMENT: Central state tracks everything
    - ERROR HANDLING: Graceful failure recovery
    """)
    
    # ===== INITIALIZE LLM =====
    print("-"*70)
    print("Initializing Vertex AI LLM...")
    print("-"*70)
    
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not set")
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("WARNING: GOOGLE_CLOUD_PROJECT not set")
    
    llm = create_llm()
    print(f"✓ LLM initialized: {llm.model_name}")
    
    # Build the graph
    graph = build_orchestrator_graph()
    
    # ===== INITIAL STATE =====
    initial_state = {
        "topic": "The impact of artificial intelligence on healthcare diagnostics",
        "research_findings": "",
        "analysis": "",
        "final_report": "",
        "current_step": "research",
        "completed_steps": [],
        "error": "",
        "retry_count": 0,
        "workflow_status": "in_progress"
    }
    
    print("\n" + "-"*70)
    print("STARTING ORCHESTRATED WORKFLOW")
    print("-"*70)
    print(f"Topic: {initial_state['topic']}")
    
    # ===== RUN THE WORKFLOW =====
    start_time = time.time()
    final_state = graph.invoke(initial_state)
    elapsed_time = time.time() - start_time
    
    # ===== DISPLAY RESULTS =====
    print("\n" + "="*70)
    print("WORKFLOW RESULTS")
    print("="*70)
    
    print(f"\nWorkflow Status: {final_state['workflow_status']}")
    print(f"Completed Steps: {final_state['completed_steps']}")
    print(f"Total Time: {elapsed_time:.2f} seconds")
    
    if final_state['workflow_status'] == "completed":
        print(f"\n{'='*70}")
        print("FINAL REPORT")
        print(f"{'='*70}")
        print(final_state['final_report'])
    else:
        print(f"\nWorkflow failed. Last error: {final_state.get('error', 'Unknown')}")
    
    # ===== RECAP =====
    print("\n" + "="*70)
    print("PATTERN RECAP")
    print("="*70)
    print(f"""
    1. WORKFLOW COORDINATION:
       └── Orchestrator routed through: {' -> '.join(final_state['completed_steps'])}
       └── Each agent invoked in correct sequence
    
    2. COMMUNICATION:
       └── Researcher wrote: research_findings ({len(final_state.get('research_findings', ''))} chars)
       └── Analyst wrote: analysis ({len(final_state.get('analysis', ''))} chars)
       └── Writer wrote: final_report ({len(final_state.get('final_report', ''))} chars)
       └── All through shared state object
    
    3. STATE MANAGEMENT:
       └── Central state tracked entire workflow
       └── completed_steps: {final_state['completed_steps']}
       └── workflow_status: {final_state['workflow_status']}
    
    4. ERROR HANDLING:
       └── Error handler ready for failures
       └── Max retries: {MAX_RETRIES}
       └── Final error state: '{final_state.get('error', 'None')}'
    """)
    print("="*70 + "\n")


# ==============================================================================
# BONUS: SIMULATE ERROR SCENARIO
# ==============================================================================

def demo_error_handling():
    """
    Demonstrate error handling by simulating a failure.
    
    This shows how the orchestrator handles agent failures.
    """
    print("\n" + "="*70)
    print("ERROR HANDLING DEMO")
    print("="*70)
    print("""
    This demo shows what happens when an agent fails:
    1. Error is captured in state
    2. Orchestrator routes to error_handler
    3. Error handler decides: retry or fail
    """)
    
    # Create a state that simulates an error
    error_state = {
        "topic": "Test topic",
        "research_findings": "Some findings",
        "analysis": "",
        "final_report": "",
        "current_step": "analyze",
        "completed_steps": ["research"],
        "error": "Simulated API timeout error",
        "retry_count": 1,
        "workflow_status": "in_progress"
    }
    
    print(f"\nSimulated error state:")
    print(f"  - Current step: {error_state['current_step']}")
    print(f"  - Error: {error_state['error']}")
    print(f"  - Retry count: {error_state['retry_count']}")
    
    # Show what the orchestrator would do
    decision = orchestrator_router(error_state)
    print(f"\nOrchestrator decision: route to '{decision}'")
    
    # Show what error handler would do
    if decision == "error_handler":
        result = error_handler(error_state)
        print(f"Error handler result: {result}")


if __name__ == "__main__":
    main()
    
    # Uncomment to see error handling demo:
    # demo_error_handling()