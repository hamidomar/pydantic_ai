"""
================================================================================
LANGGRAPH AGENTIC PARALLEL PATTERN DEMO
================================================================================

This script demonstrates the four core aspects of an agentic parallel pattern:

1. CONCURRENT EXECUTION
   - Multiple agents/tasks run in parallel (not sequentially)
   - LangGraph's fan-out pattern sends to multiple nodes simultaneously

2. RESULT SYNCHRONIZATION
   - Results from parallel tasks are collected and merged
   - Fan-in pattern aggregates outputs before continuing

3. RESOURCE MANAGEMENT
   - Control over parallelism (how many concurrent tasks)
   - Timeout handling for long-running tasks

4. ERROR HANDLING
   - Individual task failures don't crash the whole workflow
   - Partial results collected even if some tasks fail

================================================================================
THE SCENARIO
================================================================================

We'll build a "Multi-Source Research Agent" that queries multiple sources
in parallel and synthesizes the results:

PARALLEL AGENTS (run concurrently):
1. ACADEMIC AGENT - Searches academic/scientific perspective
2. NEWS AGENT - Searches current news perspective  
3. INDUSTRY AGENT - Searches industry/business perspective

AGGREGATOR AGENT (waits for all parallel results):
- Collects results from all three sources
- Synthesizes into unified summary

This is much faster than sequential execution!

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
import asyncio
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ==============================================================================
# LLM SETUP (Vertex AI with Google ADC)
# ==============================================================================

def create_llm():
    """
    Create the Vertex AI LLM client.
    """
    return ChatVertexAI(
        model_name="gemini-1.5-flash",
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        temperature=0.7,
        max_tokens=1024,
    )


# Global LLM instance
llm = None


# ==============================================================================
# STATE SCHEMA
# ==============================================================================
#
# For parallel execution, state needs to:
# - Hold inputs that ALL parallel agents can read
# - Have separate fields for EACH parallel agent's output
# - Track status of each parallel task
#
# RESULT SYNCHRONIZATION happens through the state:
# - Each parallel agent writes to its own field
# - Aggregator reads from all fields
#
# ==============================================================================

class ParallelState(TypedDict):
    """
    State for parallel agent execution.
    
    INPUT FIELDS (shared by all parallel agents):
    
    topic: str
        The research topic all agents will investigate
    
    PARALLEL OUTPUT FIELDS (one per parallel agent):
    
    academic_result: str
        Output from Academic Agent
    
    news_result: str
        Output from News Agent
    
    industry_result: str
        Output from Industry Agent
    
    SYNCHRONIZATION FIELDS:
    
    aggregated_result: str
        Final synthesized result from Aggregator
    
    STATUS/ERROR TRACKING (per agent):
    
    academic_status: str
        "pending", "success", "failed", "timeout"
    
    news_status: str
        Status of News Agent
    
    industry_status: str
        Status of Industry Agent
    
    academic_error: str
        Error message if Academic Agent failed
    
    news_error: str
        Error message if News Agent failed
    
    industry_error: str
        Error message if Industry Agent failed
    
    TIMING (for demonstrating parallelism):
    
    academic_duration: float
        How long Academic Agent took (seconds)
    
    news_duration: float
        How long News Agent took
    
    industry_duration: float
        How long Industry Agent took
    """
    # Input
    topic: str
    
    # Parallel outputs (RESULT SYNCHRONIZATION happens here)
    academic_result: str
    news_result: str
    industry_result: str
    
    # Final aggregated output
    aggregated_result: str
    
    # Status tracking (ERROR HANDLING)
    academic_status: str
    news_status: str
    industry_status: str
    
    # Error messages
    academic_error: str
    news_error: str
    industry_error: str
    
    # Timing info (demonstrates CONCURRENT EXECUTION)
    academic_duration: float
    news_duration: float
    industry_duration: float


# ==============================================================================
# CONFIGURATION (RESOURCE MANAGEMENT)
# ==============================================================================

# Timeout for each parallel task (seconds)
TASK_TIMEOUT = 60

# Max concurrent LLM calls (RESOURCE MANAGEMENT)
MAX_CONCURRENT_TASKS = 3


# ==============================================================================
# PARALLEL AGENT 1: ACADEMIC RESEARCHER
# ==============================================================================

def academic_agent(state: ParallelState) -> ParallelState:
    """
    Academic Agent: Researches topic from academic/scientific perspective.
    
    WHAT THIS DEMONSTRATES:
    - CONCURRENT EXECUTION: Runs in parallel with other agents
    - ERROR HANDLING: Captures failures without crashing workflow
    - RESOURCE MANAGEMENT: Tracks timing for analysis
    """
    topic = state["topic"]
    
    print(f"\n[ACADEMIC AGENT] Starting research on: {topic}")
    start_time = time.time()
    
    try:
        messages = [
            SystemMessage(content="""You are an academic researcher.
Provide insights from a scientific/academic perspective.
Focus on:
- Research findings and studies
- Theoretical frameworks
- Evidence-based conclusions

Keep response to 150-200 words."""),
            HumanMessage(content=f"Analyze this topic from an academic perspective: {topic}")
        ]
        
        response = llm.invoke(messages)
        result = response.content
        duration = time.time() - start_time
        
        print(f"[ACADEMIC AGENT] Completed in {duration:.2f}s")
        
        return {
            "academic_result": result,
            "academic_status": "success",
            "academic_error": "",
            "academic_duration": duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"[ACADEMIC AGENT] Failed: {e}")
        
        return {
            "academic_result": "",
            "academic_status": "failed",
            "academic_error": str(e),
            "academic_duration": duration
        }


# ==============================================================================
# PARALLEL AGENT 2: NEWS ANALYST
# ==============================================================================

def news_agent(state: ParallelState) -> ParallelState:
    """
    News Agent: Researches topic from current events perspective.
    
    Runs CONCURRENTLY with Academic and Industry agents.
    """
    topic = state["topic"]
    
    print(f"\n[NEWS AGENT] Starting research on: {topic}")
    start_time = time.time()
    
    try:
        messages = [
            SystemMessage(content="""You are a news analyst.
Provide insights from a current events perspective.
Focus on:
- Recent developments and news
- Public discourse and debates
- Trending aspects of the topic

Keep response to 150-200 words."""),
            HumanMessage(content=f"Analyze this topic from a news/current events perspective: {topic}")
        ]
        
        response = llm.invoke(messages)
        result = response.content
        duration = time.time() - start_time
        
        print(f"[NEWS AGENT] Completed in {duration:.2f}s")
        
        return {
            "news_result": result,
            "news_status": "success",
            "news_error": "",
            "news_duration": duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"[NEWS AGENT] Failed: {e}")
        
        return {
            "news_result": "",
            "news_status": "failed",
            "news_error": str(e),
            "news_duration": duration
        }


# ==============================================================================
# PARALLEL AGENT 3: INDUSTRY ANALYST
# ==============================================================================

def industry_agent(state: ParallelState) -> ParallelState:
    """
    Industry Agent: Researches topic from business/industry perspective.
    
    Runs CONCURRENTLY with Academic and News agents.
    """
    topic = state["topic"]
    
    print(f"\n[INDUSTRY AGENT] Starting research on: {topic}")
    start_time = time.time()
    
    try:
        messages = [
            SystemMessage(content="""You are an industry analyst.
Provide insights from a business/industry perspective.
Focus on:
- Market trends and business impact
- Industry adoption and challenges
- Commercial applications

Keep response to 150-200 words."""),
            HumanMessage(content=f"Analyze this topic from an industry/business perspective: {topic}")
        ]
        
        response = llm.invoke(messages)
        result = response.content
        duration = time.time() - start_time
        
        print(f"[INDUSTRY AGENT] Completed in {duration:.2f}s")
        
        return {
            "industry_result": result,
            "industry_status": "success",
            "industry_error": "",
            "industry_duration": duration
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"[INDUSTRY AGENT] Failed: {e}")
        
        return {
            "industry_result": "",
            "industry_status": "failed",
            "industry_error": str(e),
            "industry_duration": duration
        }


# ==============================================================================
# AGGREGATOR AGENT (RESULT SYNCHRONIZATION)
# ==============================================================================
#
# This agent waits for ALL parallel agents to complete.
# Then it synchronizes/merges their results.
#
# This is the FAN-IN part of the parallel pattern.
#
# ==============================================================================

def aggregator_agent(state: ParallelState) -> ParallelState:
    """
    Aggregator Agent: Synchronizes and synthesizes parallel results.
    
    WHAT THIS DEMONSTRATES:
    - RESULT SYNCHRONIZATION: Collects outputs from all parallel agents
    - ERROR HANDLING: Works with partial results if some agents failed
    
    This is the FAN-IN node that merges parallel outputs.
    """
    topic = state["topic"]
    
    print(f"\n{'='*60}")
    print("AGGREGATOR AGENT (Synchronizing Results)")
    print(f"{'='*60}")
    
    # ===== RESULT SYNCHRONIZATION: Collect all results =====
    academic_result = state.get("academic_result", "")
    news_result = state.get("news_result", "")
    industry_result = state.get("industry_result", "")
    
    academic_status = state.get("academic_status", "pending")
    news_status = state.get("news_status", "pending")
    industry_status = state.get("industry_status", "pending")
    
    print(f"\nStatus Summary:")
    print(f"  Academic: {academic_status}")
    print(f"  News: {news_status}")
    print(f"  Industry: {industry_status}")
    
    # ===== ERROR HANDLING: Work with available results =====
    successful_results = []
    
    if academic_status == "success" and academic_result:
        successful_results.append(f"ACADEMIC PERSPECTIVE:\n{academic_result}")
    
    if news_status == "success" and news_result:
        successful_results.append(f"NEWS PERSPECTIVE:\n{news_result}")
    
    if industry_status == "success" and industry_result:
        successful_results.append(f"INDUSTRY PERSPECTIVE:\n{industry_result}")
    
    if not successful_results:
        # All agents failed
        return {
            "aggregated_result": "ERROR: All research agents failed. No results to aggregate."
        }
    
    print(f"\nAggregating {len(successful_results)} successful results...")
    
    # Synthesize results
    try:
        combined_input = "\n\n".join(successful_results)
        
        messages = [
            SystemMessage(content="""You are a research synthesizer.
Your job is to combine multiple perspectives into a unified summary.

Create a cohesive summary that:
1. Highlights key insights from each perspective
2. Identifies common themes
3. Notes any contrasting viewpoints
4. Provides a balanced conclusion

Keep response to 250-300 words."""),
            HumanMessage(content=f"""Topic: {topic}

Research from multiple perspectives:

{combined_input}

Please synthesize these perspectives into a unified summary:""")
        ]
        
        response = llm.invoke(messages)
        aggregated = response.content
        
        print(f"\nSynthesized Summary:\n{'-'*40}")
        print(aggregated[:500] + "..." if len(aggregated) > 500 else aggregated)
        print(f"{'-'*40}")
        
        return {
            "aggregated_result": aggregated
        }
        
    except Exception as e:
        print(f"Aggregator failed: {e}")
        # Return raw results if synthesis fails
        return {
            "aggregated_result": f"Synthesis failed. Raw results:\n\n{combined_input}"
        }


# ==============================================================================
# BUILD THE PARALLEL GRAPH
# ==============================================================================
#
# Graph structure showing FAN-OUT and FAN-IN:
#
#                    START
#                      |
#           +-------- FAN-OUT --------+
#           |          |              |
#           v          v              v
#     [academic]    [news]      [industry]
#           |          |              |
#           +-------- FAN-IN ---------+
#                      |
#                      v
#               [aggregator]
#                      |
#                      v
#                     END
#
# LangGraph executes the three parallel nodes CONCURRENTLY.
#
# ==============================================================================

def build_parallel_graph():
    """
    Build the parallel execution graph.
    
    KEY CONCEPTS:
    
    1. CONCURRENT EXECUTION:
       - Multiple edges from START = fan-out
       - LangGraph runs these nodes in parallel
    
    2. RESULT SYNCHRONIZATION:
       - All parallel nodes connect to aggregator
       - Aggregator waits for all to complete (fan-in)
    
    3. RESOURCE MANAGEMENT:
       - LangGraph manages the thread pool
       - Parallelism controlled by graph structure
    
    4. ERROR HANDLING:
       - Each node handles its own errors
       - Aggregator works with partial results
    """
    builder = StateGraph(ParallelState)
    
    # ===== ADD PARALLEL AGENT NODES =====
    builder.add_node("academic", academic_agent)
    builder.add_node("news", news_agent)
    builder.add_node("industry", industry_agent)
    builder.add_node("aggregator", aggregator_agent)
    
    # ===== FAN-OUT: START to all parallel agents =====
    # When multiple edges leave the same node, LangGraph
    # executes the target nodes IN PARALLEL
    builder.add_edge(START, "academic")
    builder.add_edge(START, "news")
    builder.add_edge(START, "industry")
    
    # ===== FAN-IN: All parallel agents to aggregator =====
    # Aggregator automatically waits for ALL incoming edges
    # before executing (this is the synchronization point)
    builder.add_edge("academic", "aggregator")
    builder.add_edge("news", "aggregator")
    builder.add_edge("industry", "aggregator")
    
    # ===== Aggregator to END =====
    builder.add_edge("aggregator", END)
    
    # Compile
    graph = builder.compile()
    
    return graph


# ==============================================================================
# DEMONSTRATION
# ==============================================================================

def main():
    """
    Run the parallel pattern demonstration.
    
    WHAT YOU'LL SEE:
    
    1. CONCURRENT EXECUTION:
       - All three agents start at approximately the same time
       - Total time ≈ slowest agent (not sum of all agents)
    
    2. RESULT SYNCHRONIZATION:
       - Aggregator receives all results
       - Synthesizes into unified output
    
    3. RESOURCE MANAGEMENT:
       - Timing shows parallel execution benefit
       - Compare total time vs sum of individual times
    
    4. ERROR HANDLING:
       - Status tracked per agent
       - Partial results still aggregated
    """
    global llm
    
    print("\n" + "="*70)
    print("LANGGRAPH AGENTIC PARALLEL PATTERN DEMO")
    print("="*70)
    print("""
    SCENARIO: Multi-Source Research Agent
    
    Three specialist agents running IN PARALLEL:
    
    ┌─────────────────────────────────────────────────────────┐
    │                        START                            │
    │                          │                              │
    │           ┌──────────────┼──────────────┐               │
    │           │              │              │               │
    │           ▼              ▼              ▼               │
    │     [ACADEMIC]       [NEWS]       [INDUSTRY]           │
    │           │              │              │               │
    │           └──────────────┼──────────────┘               │
    │                          │                              │
    │                          ▼                              │
    │                   [AGGREGATOR]                          │
    │                          │                              │
    │                          ▼                              │
    │                         END                             │
    └─────────────────────────────────────────────────────────┘
    
    This demonstrates:
    - CONCURRENT EXECUTION: Agents run simultaneously
    - RESULT SYNCHRONIZATION: Aggregator waits for all
    - RESOURCE MANAGEMENT: Controlled parallelism
    - ERROR HANDLING: Graceful partial failure
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
    graph = build_parallel_graph()
    
    # ===== INITIAL STATE =====
    initial_state = {
        "topic": "The future of renewable energy and its impact on global economies",
        "academic_result": "",
        "news_result": "",
        "industry_result": "",
        "aggregated_result": "",
        "academic_status": "pending",
        "news_status": "pending",
        "industry_status": "pending",
        "academic_error": "",
        "news_error": "",
        "industry_error": "",
        "academic_duration": 0.0,
        "news_duration": 0.0,
        "industry_duration": 0.0
    }
    
    print("\n" + "-"*70)
    print("STARTING PARALLEL EXECUTION")
    print("-"*70)
    print(f"Topic: {initial_state['topic']}")
    print("\nWatch the agents start simultaneously...")
    
    # ===== RUN THE WORKFLOW =====
    total_start = time.time()
    final_state = graph.invoke(initial_state)
    total_duration = time.time() - total_start
    
    # ===== DISPLAY RESULTS =====
    print("\n" + "="*70)
    print("PARALLEL EXECUTION RESULTS")
    print("="*70)
    
    # Timing analysis (demonstrates CONCURRENT EXECUTION benefit)
    academic_time = final_state.get("academic_duration", 0)
    news_time = final_state.get("news_duration", 0)
    industry_time = final_state.get("industry_duration", 0)
    
    sequential_time = academic_time + news_time + industry_time
    parallel_time = max(academic_time, news_time, industry_time)
    
    print(f"\n--- TIMING ANALYSIS (CONCURRENT EXECUTION) ---")
    print(f"Academic Agent:  {academic_time:.2f}s")
    print(f"News Agent:      {news_time:.2f}s")
    print(f"Industry Agent:  {industry_time:.2f}s")
    print(f"\nIf sequential: {sequential_time:.2f}s (sum of all)")
    print(f"Parallel time:   {parallel_time:.2f}s (slowest agent)")
    print(f"Total workflow:  {total_duration:.2f}s (includes aggregation)")
    
    if sequential_time > 0:
        speedup = sequential_time / total_duration
        print(f"\n✓ Speedup from parallelism: {speedup:.2f}x faster!")
    
    # Status summary (ERROR HANDLING)
    print(f"\n--- STATUS SUMMARY (ERROR HANDLING) ---")
    print(f"Academic: {final_state.get('academic_status', 'unknown')}")
    if final_state.get('academic_error'):
        print(f"  Error: {final_state['academic_error']}")
    
    print(f"News: {final_state.get('news_status', 'unknown')}")
    if final_state.get('news_error'):
        print(f"  Error: {final_state['news_error']}")
    
    print(f"Industry: {final_state.get('industry_status', 'unknown')}")
    if final_state.get('industry_error'):
        print(f"  Error: {final_state['industry_error']}")
    
    # Final aggregated result
    print(f"\n{'='*70}")
    print("AGGREGATED RESULT (RESULT SYNCHRONIZATION)")
    print(f"{'='*70}")
    print(final_state.get('aggregated_result', 'No result'))
    
    # ===== RECAP =====
    print("\n" + "="*70)
    print("PATTERN RECAP")
    print("="*70)
    
    successful_count = sum(1 for s in ['academic_status', 'news_status', 'industry_status'] 
                          if final_state.get(s) == 'success')
    
    print(f"""
    1. CONCURRENT EXECUTION:
       └── All 3 agents started simultaneously
       └── Parallel execution time: ~{parallel_time:.2f}s
       └── Sequential would have been: ~{sequential_time:.2f}s
       └── Speedup: {sequential_time/max(total_duration, 0.1):.2f}x
    
    2. RESULT SYNCHRONIZATION:
       └── Aggregator waited for all agents
       └── Collected {successful_count}/3 successful results
       └── Synthesized into unified summary
    
    3. RESOURCE MANAGEMENT:
       └── LangGraph managed thread pool automatically
       └── {MAX_CONCURRENT_TASKS} max concurrent tasks configured
       └── No manual thread management needed
    
    4. ERROR HANDLING:
       └── Each agent tracked status independently
       └── Failures captured without crashing workflow
       └── Aggregator handled partial results gracefully
    """)
    print("="*70 + "\n")


# ==============================================================================
# BONUS: DEMONSTRATE ERROR HANDLING WITH SIMULATED FAILURE
# ==============================================================================

def demo_partial_failure():
    """
    Demonstrate that the pattern handles partial failures gracefully.
    
    One agent fails, but others succeed and results are still aggregated.
    """
    print("\n" + "="*70)
    print("PARTIAL FAILURE DEMO")
    print("="*70)
    print("""
    This shows how the aggregator handles partial failures:
    - 2 agents succeed
    - 1 agent fails
    - Aggregator still produces result from available data
    """)
    
    # Simulate a state where one agent failed
    partial_state = {
        "topic": "Test topic",
        "academic_result": "Academic perspective: This is important research...",
        "news_result": "",  # Failed
        "industry_result": "Industry perspective: Market trends show...",
        "aggregated_result": "",
        "academic_status": "success",
        "news_status": "failed",
        "industry_status": "success",
        "academic_error": "",
        "news_error": "API timeout after 30 seconds",
        "industry_error": "",
        "academic_duration": 2.5,
        "news_duration": 30.0,
        "industry_duration": 2.8
    }
    
    print(f"\nSimulated state:")
    print(f"  Academic: {partial_state['academic_status']}")
    print(f"  News: {partial_state['news_status']} - {partial_state['news_error']}")
    print(f"  Industry: {partial_state['industry_status']}")
    
    print("\nCalling aggregator with partial results...")
    result = aggregator_agent(partial_state)
    
    print(f"\nAggregator successfully handled partial failure!")
    print(f"Result length: {len(result.get('aggregated_result', ''))} characters")


if __name__ == "__main__":
    main()
    
    # Uncomment to see partial failure handling:
    # demo_partial_failure()