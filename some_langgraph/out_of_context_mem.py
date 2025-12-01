"""
================================================================================
LANGGRAPH SESSIONS TUTORIAL
================================================================================

This script teaches four core LangGraph concepts:

1. SESSION ID (thread_id)
   - How LangGraph identifies different conversations
   - Each thread_id = one independent conversation

2. CHECKPOINTING
   - Automatic state saving after each graph step
   - Uses MemorySaver here (in-memory, no database needed)

3. STATE RESTORATION
   - When you use the same thread_id, state loads automatically
   - No manual save/load code needed - LangGraph handles it

4. CONTEXT MANAGEMENT
   - Preventing token overflow in long conversations
   - We trim old messages, keeping recent ones + system message

================================================================================
AUTHENTICATION SETUP
================================================================================

This uses Google Application Default Credentials (ADC) with a service account.

Before running, set these environment variables:

    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    export GOOGLE_CLOUD_PROJECT="your-project-id"

The ChatVertexAI client will automatically pick up these credentials.
No explicit auth code needed - that's the beauty of ADC.

================================================================================
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ==============================================================================
# AUTHENTICATION VERIFICATION (Optional but helpful)
# ==============================================================================
#
# This just checks that your environment is set up correctly.
# Not strictly necessary - ChatVertexAI will fail with clear errors anyway.
# But nice to catch issues early.
#
# ==============================================================================

def verify_google_auth():
    """
    Quick check that Google ADC is configured.
    
    WHAT THIS CHECKS:
    - GOOGLE_APPLICATION_CREDENTIALS is set
    - The file actually exists
    - GOOGLE_CLOUD_PROJECT is set
    
    WHY THIS EXISTS:
    - Catches setup issues before you waste time debugging
    - Gives clear guidance on what's missing
    """
    print("\n--- Checking Google ADC Configuration ---")
    
    # Check credentials file
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("✗ GOOGLE_APPLICATION_CREDENTIALS not set")
        print("  Fix: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
        return False
    
    if not os.path.exists(creds_path):
        print(f"✗ Credentials file not found: {creds_path}")
        return False
    
    print(f"✓ Credentials file: {creds_path}")
    
    # Check project ID
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("✗ GOOGLE_CLOUD_PROJECT not set")
        print("  Fix: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return False
    
    print(f"✓ Project ID: {project_id}")
    print("--- Authentication looks good! ---\n")
    return True


# ==============================================================================
# CONCEPT 1: STATE SCHEMA
# ==============================================================================
#
# LangGraph needs to know what "state" looks like.
# For a chatbot, state is simply: a list of messages.
#
# The `add_messages` annotation tells LangGraph:
# "When updating this field, APPEND new messages, don't replace."
#
# WHY THIS MATTERS:
# Without add_messages: {"messages": [new_msg]} would DELETE all history
# With add_messages: {"messages": [new_msg]} ADDS to history
#
# This annotation is how LangGraph knows to accumulate messages
# rather than overwriting them each time.
#
# ==============================================================================

class ChatState(TypedDict):
    """
    The state of our conversation.
    
    STATE IS SIMPLE:
    Just one field: messages (a list of chat messages)
    
    THE ANNOTATION IS KEY:
    Annotated[list[BaseMessage], add_messages]
    
    This tells LangGraph:
    "When a node returns {"messages": [x]}, APPEND x to the list"
    
    Without this, each node would replace the entire message history.
    With this, messages accumulate naturally.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ==============================================================================
# CONCEPT 4: CONTEXT MANAGEMENT
# ==============================================================================
#
# THE PROBLEM:
# LLMs have token limits. A conversation with 1000 messages will overflow.
#
# THE SOLUTION:
# Trim old messages before sending to the LLM.
#
# OUR STRATEGY (simple and effective):
# - Always keep the system message (defines AI behavior)
# - Keep only the last N messages after that
#
# WHERE THIS HAPPENS:
# Inline, inside the chat node, right before the LLM call.
#
# IMPORTANT DISTINCTION:
# - What LLM sees: trimmed messages (fits in context window)
# - What gets checkpointed: FULL history (nothing lost)
#
# ==============================================================================

# How many recent messages to keep (excluding system message)
# Adjust based on your model's context window
MAX_MESSAGES = 10


def trim_messages_for_context(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Trim messages to prevent context overflow.
    
    THE ALGORITHM:
    1. If first message is SystemMessage, keep it (always)
    2. Keep only the last MAX_MESSAGES of everything else
    
    EXAMPLE (MAX_MESSAGES = 10):
    Input: [System, Human1, AI1, Human2, AI2, ... Human50, AI50]
    Output: [System, Human46, AI46, Human47, AI47, Human48, AI48, Human49, AI49, Human50, AI50]
    
    The LLM sees 11 messages (system + 10 recent).
    The checkpoint still has all 101 messages.
    
    WHY KEEP SYSTEM MESSAGE:
    System messages define behavior ("You are a helpful assistant...")
    Losing this would change the AI's personality mid-conversation.
    """
    if not messages:
        return messages
    
    # Check if first message is a system message
    if isinstance(messages[0], SystemMessage):
        system_msg = messages[0]
        other_messages = messages[1:]
        
        # Keep system + last N messages
        if len(other_messages) > MAX_MESSAGES:
            trimmed = other_messages[-MAX_MESSAGES:]
            print(f"    [Context trimmed: {len(other_messages)} -> {len(trimmed)} messages]")
            return [system_msg] + trimmed
        return [system_msg] + other_messages
    else:
        # No system message - just keep last N
        if len(messages) > MAX_MESSAGES:
            trimmed = messages[-MAX_MESSAGES:]
            print(f"    [Context trimmed: {len(messages)} -> {len(trimmed)} messages]")
            return trimmed
        return messages


# ==============================================================================
# THE CHAT NODE
# ==============================================================================
#
# This is where the actual LLM call happens.
#
# FLOW:
# 1. Receive state (contains all messages)
# 2. Trim messages (prevent context overflow)
# 3. Call LLM with trimmed messages
# 4. Return AI response (gets APPENDED via add_messages)
#
# THE RETURN VALUE:
# We return {"messages": [response]}
# Because of add_messages, this APPENDS to state, not replaces.
#
# ==============================================================================

def create_chat_node(llm):
    """
    Factory function to create the chat node.
    
    WHY A FACTORY:
    We need to inject the LLM into the node.
    This keeps the node function signature clean for LangGraph.
    """
    
    def chat_node(state: ChatState) -> ChatState:
        """
        The heart of the chatbot.
        
        WHAT HAPPENS HERE:
        
        1. GET STATE
           all_messages = state["messages"]
           This includes EVERYTHING - full history
        
        2. TRIM FOR CONTEXT
           trimmed = trim_messages_for_context(all_messages)
           Now we have just what the LLM can handle
        
        3. CALL LLM
           response = llm.invoke(trimmed)
           LLM sees trimmed version
        
        4. RETURN RESPONSE
           return {"messages": [response]}
           add_messages annotation means this APPENDS
        
        RESULT:
        - Full history in state: grows by 1 message
        - LLM context: stays bounded
        - Checkpoint: saves full history
        """
        # Step 1: Get all messages from state
        all_messages = state["messages"]
        
        # Step 2: Trim for context management
        trimmed_messages = trim_messages_for_context(all_messages)
        
        # Step 3: Call the LLM with trimmed context
        response = llm.invoke(trimmed_messages)
        
        # Step 4: Return response (will be APPENDED due to add_messages)
        return {"messages": [response]}
    
    return chat_node


# ==============================================================================
# CONCEPT 2: CHECKPOINTING
# ==============================================================================
#
# A CHECKPOINTER saves state after each graph step.
#
# AVAILABLE CHECKPOINTERS:
# - MemorySaver: In-memory (lost on restart) - great for learning
# - SqliteSaver: Persists to SQLite file
# - PostgresSaver: Persists to PostgreSQL database
#
# WHY WE USE MemorySaver:
# - Zero setup (no database needed)
# - Perfect for understanding concepts
# - Behavior is identical to persistent checkpointers
#
# HOW IT WORKS:
# 1. Attach checkpointer when compiling graph
# 2. Every invoke() automatically saves state
# 3. State is keyed by thread_id (see Concept 3)
#
# YOU NEVER CALL save() or load():
# LangGraph handles this automatically.
#
# ==============================================================================

def build_graph(llm):
    """
    Build and compile the LangGraph.
    
    GRAPH STRUCTURE:
    
        START
          |
          v
        [chat]  <-- Our only node (trims context, calls LLM)
          |
          v
         END
    
    The simplicity is intentional.
    The interesting part is the CHECKPOINTER, not the graph structure.
    
    COMPILATION:
    graph = builder.compile(checkpointer=checkpointer)
    
    This is where checkpointing gets attached.
    After this, every invoke() automatically checkpoints.
    """
    # Create graph builder with our state schema
    builder = StateGraph(ChatState)
    
    # Add the chat node
    builder.add_node("chat", create_chat_node(llm))
    
    # Define edges: START -> chat -> END
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    
    # =========================================
    # THE KEY PART: ATTACHING THE CHECKPOINTER
    # =========================================
    #
    # MemorySaver = in-memory dictionary storage
    # 
    # After this, every invoke() will:
    # 1. Check for existing state (by thread_id)
    # 2. Load it if found
    # 3. Run the graph
    # 4. Save new state
    #
    # All automatic. No manual save/load.
    #
    checkpointer = MemorySaver()
    
    # Compile with checkpointer
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph


# ==============================================================================
# CONCEPT 3: SESSION ID (thread_id) AND STATE RESTORATION
# ==============================================================================
#
# thread_id is how LangGraph knows WHICH conversation you mean.
#
# THE RULE:
# Same thread_id = same conversation = state RESTORED
# Different thread_id = different conversation = fresh state
#
# HOW TO USE IT:
# config = {"configurable": {"thread_id": "my-session-123"}}
# graph.invoke(inputs, config=config)
#
# WHAT HAPPENS AUTOMATICALLY:
# 1. LangGraph sees thread_id in config
# 2. Checks checkpointer for existing state with that thread_id
# 3. If found: loads it, appends your new messages
# 4. If not found: starts fresh
# 5. After invoke(): saves new state under that thread_id
#
# NO MANUAL SAVE/LOAD CODE NEEDED
#
# ==============================================================================

def chat_with_session(graph, thread_id: str, user_message: str, system_prompt: str = None):
    """
    Send a message in a specific session.
    
    PARAMETERS:
    - graph: The compiled LangGraph (with checkpointer)
    - thread_id: The session identifier
    - user_message: What the user says
    - system_prompt: Only used for NEW conversations
    
    THE thread_id MAGIC:
    
    Call 1: chat_with_session(graph, "alice", "Hi")
            -> Creates new session, saves state
    
    Call 2: chat_with_session(graph, "bob", "Hello")
            -> Creates DIFFERENT session, saves state
    
    Call 3: chat_with_session(graph, "alice", "Remember me?")
            -> RESTORES Alice's session, AI remembers "Hi"
    
    STATE RESTORATION IS AUTOMATIC:
    You just pass the thread_id. LangGraph does the rest.
    """
    
    # ===== THE CONFIG =====
    # This is how you identify the session
    # thread_id is the key that LangGraph uses to look up state
    config = {
        "configurable": {
            "thread_id": thread_id  # <-- THE SESSION IDENTIFIER
        }
    }
    
    # Build input messages
    messages = []
    
    # Check if this is a new conversation
    # If new AND system_prompt provided, include it
    existing_state = graph.get_state(config)
    
    if not existing_state.values and system_prompt:
        # New conversation - include system prompt
        messages.append(SystemMessage(content=system_prompt))
    
    # Add user's message
    messages.append(HumanMessage(content=user_message))
    
    # ===== INVOKE THE GRAPH =====
    #
    # What happens here:
    # 1. LangGraph sees thread_id in config
    # 2. Loads existing state for that thread_id (if any)
    # 3. Appends our new messages to state
    # 4. Runs the graph (chat node)
    # 5. Saves updated state back to checkpointer
    #
    # ALL AUTOMATIC.
    #
    result = graph.invoke(
        {"messages": messages},
        config=config
    )
    
    # Extract AI response (last message in result)
    ai_response = result["messages"][-1].content
    
    return ai_response


def show_session_history(graph, thread_id: str):
    """
    Display all messages in a session.
    
    PURPOSE:
    Proves that state IS being checkpointed.
    All messages are preserved.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    if not state.values:
        print(f"\n[Session '{thread_id}' has no history]")
        return
    
    messages = state.values.get("messages", [])
    
    print(f"\n{'='*60}")
    print(f"SESSION HISTORY: {thread_id}")
    print(f"Total messages: {len(messages)}")
    print(f"{'='*60}")
    
    for i, msg in enumerate(messages):
        role = type(msg).__name__.replace("Message", "").upper()
        # Truncate long messages for display
        content = msg.content
        if len(content) > 80:
            content = content[:80] + "..."
        print(f"{i+1:2d}. [{role:8s}] {content}")
    
    print(f"{'='*60}\n")


# ==============================================================================
# DEMONSTRATION
# ==============================================================================
#
# This demo shows all four concepts working together:
#
# 1. SESSION ID: Two different sessions (alice and bob)
# 2. CHECKPOINTING: State saves automatically after each message
# 3. STATE RESTORATION: Returning to a session continues seamlessly
# 4. CONTEXT MANAGEMENT: Ready to trim when messages exceed limit
#
# ==============================================================================

def main():
    """
    Demonstrate LangGraph session management.
    
    WHAT YOU'LL SEE:
    
    1. Two separate conversations start (alice about cooking, bob about space)
    2. We switch between them
    3. Each session remembers its context
    4. Histories prove checkpointing worked
    """
    
    print("\n" + "="*70)
    print("LANGGRAPH SESSIONS TUTORIAL")
    print("="*70)
    
    # ===== STEP 0: Verify Authentication =====
    print("\n[SETUP] Verifying Google ADC...")
    if not verify_google_auth():
        print("\n⚠ Authentication not configured. Please set environment variables.")
        print("Continuing anyway - you'll see auth errors when calling the LLM.\n")
    
    # ===== STEP 1: Create LLM and Graph =====
    print("[SETUP] Creating LLM and building graph...")
    
    # Initialize LLM using Vertex AI
    # Credentials come from GOOGLE_APPLICATION_CREDENTIALS (ADC)
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",  # Fast and cheap, good for demos
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        temperature=0.7,
    )
    
    # Build the graph with checkpointer
    graph = build_graph(llm)
    print("✓ Graph built with MemorySaver checkpointer attached\n")
    
    # ===== STEP 2: Demonstrate Two Independent Sessions =====
    
    print("-"*70)
    print("DEMO: Two Independent Sessions")
    print("-"*70)
    
    # ----- Alice's Session: Cooking -----
    print("\n>>> [SESSION: alice] Starting conversation about cooking...")
    response = chat_with_session(
        graph,
        thread_id="session_alice",
        user_message="Hi! I want to learn to make fresh pasta.",
        system_prompt="You are a friendly Italian cooking instructor. Keep responses to 2-3 sentences."
    )
    print(f"<<< AI: {response}")
    
    # ----- Bob's Session: Space -----
    print("\n>>> [SESSION: bob] Starting conversation about space...")
    response = chat_with_session(
        graph,
        thread_id="session_bob",
        user_message="What are black holes?",
        system_prompt="You are an enthusiastic astronomy educator. Keep responses to 2-3 sentences."
    )
    print(f"<<< AI: {response}")
    
    # ===== STEP 3: Return to Alice - State Should Be Restored =====
    
    print("\n" + "-"*70)
    print("DEMO: State Restoration")
    print("-"*70)
    
    print("\n>>> [SESSION: alice] Continuing... (AI should remember pasta)")
    response = chat_with_session(
        graph,
        thread_id="session_alice",  # Same thread_id!
        user_message="What type of flour should I use?"
    )
    print(f"<<< AI: {response}")
    
    # ===== STEP 4: Return to Bob - Different State Restored =====
    
    print("\n>>> [SESSION: bob] Continuing... (AI should remember black holes)")
    response = chat_with_session(
        graph,
        thread_id="session_bob",  # Same thread_id!
        user_message="How do they form?"
    )
    print(f"<<< AI: {response}")
    
    # ===== STEP 5: A Few More Messages to Show History Growing =====
    
    print("\n>>> [SESSION: alice] Adding more to conversation...")
    response = chat_with_session(
        graph,
        thread_id="session_alice",
        user_message="How long should I knead the dough?"
    )
    print(f"<<< AI: {response}")
    
    # ===== STEP 6: Show Conversation Histories =====
    
    print("\n" + "-"*70)
    print("PROOF: Checkpointed Histories")
    print("-"*70)
    
    show_session_history(graph, "session_alice")
    show_session_history(graph, "session_bob")
    
    # ===== RECAP =====
    
    print("="*70)
    print("WHAT JUST HAPPENED - RECAP")
    print("="*70)
    print("""
    CONCEPT 1 - SESSION ID (thread_id):
    ├── 'session_alice' = cooking conversation
    ├── 'session_bob' = space conversation  
    └── Each completely independent
    
    CONCEPT 2 - CHECKPOINTING:
    ├── MemorySaver attached at compile()
    ├── Every invoke() saved state automatically
    └── No manual save() calls anywhere
    
    CONCEPT 3 - STATE RESTORATION:
    ├── Returned to 'session_alice' -> pasta context restored
    ├── Returned to 'session_bob' -> black hole context restored
    └── All automatic via thread_id lookup
    
    CONCEPT 4 - CONTEXT MANAGEMENT:
    ├── trim_messages_for_context() ready
    ├── Would activate when messages > MAX_MESSAGES
    ├── LLM sees trimmed version
    └── Full history stays checkpointed
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()