"""
================================================================================
LANGGRAPH CODE GENERATION WITH SELF-CORRECTION DEMO
================================================================================

This script demonstrates a self-correcting code generation agent that:

1. GENERATES CODE
   - Takes a coding question + documentation context
   - Uses structured output to produce: description, imports, code

2. VALIDATES CODE
   - Runs actual Python exec() to test imports
   - Runs actual Python exec() to test code execution
   - Captures real error messages

3. SELF-CORRECTS
   - If code fails, error message is fed back to LLM
   - LLM retries with knowledge of what went wrong
   - Loop continues until success or max iterations

4. STRUCTURED OUTPUT
   - Uses Pydantic to enforce output schema
   - Guarantees we get {prefix, imports, code} structure

================================================================================
THE FLOW
================================================================================

    START
      │
      ▼
┌─────────────┐
│  GENERATE   │◄────────────────────┐
│             │                     │
│  LLM call   │                     │
│  with docs  │                     │
└──────┬──────┘                     │
       │                            │
       ▼                            │
┌─────────────┐                     │
│ CHECK_CODE  │                     │
│             │                     │
│ exec() test │                     │
└──────┬──────┘                     │
       │                            │
       ▼                            │
┌─────────────┐     error="yes"     │
│  DECISION   │─────────────────────┘
│             │     AND iterations < max
└──────┬──────┘
       │ error="no" OR iterations >= max
       ▼
      END

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
from typing import List, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# ==============================================================================
# LLM SETUP (Vertex AI with Google ADC)
# ==============================================================================

def create_llm():
    """
    Create the Vertex AI LLM client.
    """
    return ChatVertexAI(
        model_name="gemini-1.5-pro",  # Using Pro for better code generation
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        temperature=0,  # Deterministic for code generation
        max_tokens=2048,
    )


# Global LLM instance
llm = None


# ==============================================================================
# STRUCTURED OUTPUT SCHEMA
# ==============================================================================
#
# We use Pydantic to define the expected output structure.
# This ensures the LLM returns code in a predictable format.
#
# WHY STRUCTURED OUTPUT:
# - Guarantees we get separate imports vs code (easier to test)
# - Includes description/explanation
# - No need to parse free-form text
#
# ==============================================================================

class CodeSolution(BaseModel):
    """
    Schema for code solutions.
    
    The LLM will be forced to return output matching this structure.
    This is achieved via .with_structured_output() on the LLM.
    """
    prefix: str = Field(
        description="Description of the problem and the approach taken"
    )
    imports: str = Field(
        description="Python import statements (just the imports, nothing else)"
    )
    code: str = Field(
        description="Python code block (without imports, they go in 'imports' field)"
    )


# ==============================================================================
# DOCUMENTATION CONTEXT
# ==============================================================================
#
# In the original tutorial, they load real LangChain docs.
# For this demo, we'll use a smaller, self-contained context
# about a fictional simple API to keep it focused.
#
# In production, this could be:
# - Loaded from files
# - Retrieved from a vector database
# - Scraped from documentation sites
#
# ==============================================================================

DOCUMENTATION_CONTEXT = """
# DataProcessor API Documentation

## Overview
DataProcessor is a simple Python library for data transformation.

## Installation
```python
from data_processor import DataProcessor, Pipeline, Transformer
```

## Classes

### DataProcessor
Main class for processing data.

```python
processor = DataProcessor()
result = processor.process(data, transform_type="uppercase")
```

Methods:
- `process(data: str, transform_type: str) -> str`: Transforms the input data
  - transform_type options: "uppercase", "lowercase", "reverse", "strip"

### Pipeline  
Chain multiple transformations together.

```python
pipeline = Pipeline()
pipeline.add_step("uppercase")
pipeline.add_step("reverse")
result = pipeline.run("hello world")
```

Methods:
- `add_step(transform_type: str)`: Add a transformation step
- `run(data: str) -> str`: Execute all steps in order

### Transformer
Functional interface for single transformations.

```python
from data_processor import transform
result = transform("hello", "uppercase")  # Returns "HELLO"
```

## Examples

Example 1: Simple uppercase
```python
from data_processor import DataProcessor
proc = DataProcessor()
result = proc.process("hello", "uppercase")
print(result)  # Output: HELLO
```

Example 2: Pipeline with multiple steps
```python
from data_processor import Pipeline
pipe = Pipeline()
pipe.add_step("uppercase")
pipe.add_step("reverse")
result = pipe.run("hello")
print(result)  # Output: OLLEH
```
"""

# We also need to create the mock library so exec() works
MOCK_LIBRARY_CODE = '''
"""Mock data_processor library for testing"""

class DataProcessor:
    """Main class for processing data."""
    
    def process(self, data: str, transform_type: str = "uppercase") -> str:
        """Transform the input data."""
        if transform_type == "uppercase":
            return data.upper()
        elif transform_type == "lowercase":
            return data.lower()
        elif transform_type == "reverse":
            return data[::-1]
        elif transform_type == "strip":
            return data.strip()
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")


class Pipeline:
    """Chain multiple transformations."""
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, transform_type: str):
        """Add a transformation step."""
        self.steps.append(transform_type)
        return self
    
    def run(self, data: str) -> str:
        """Execute all steps in order."""
        processor = DataProcessor()
        result = data
        for step in self.steps:
            result = processor.process(result, step)
        return result


class Transformer:
    """Functional transformer."""
    
    @staticmethod
    def apply(data: str, transform_type: str) -> str:
        return DataProcessor().process(data, transform_type)


def transform(data: str, transform_type: str) -> str:
    """Functional interface for transformations."""
    return DataProcessor().process(data, transform_type)
'''


# ==============================================================================
# STATE SCHEMA
# ==============================================================================
#
# State tracks:
# - The question being asked
# - Conversation history (including errors for feedback)
# - Current code generation
# - Error status for control flow
# - Iteration count to prevent infinite loops
#
# ==============================================================================

class GraphState(TypedDict):
    """
    State for the code generation graph.
    
    FIELDS:
    
    question: str
        The user's coding question
    
    messages: List
        Conversation history - crucial for self-correction!
        Contains: original question, past attempts, error messages
        LLM sees this history and learns from mistakes
    
    generation: CodeSolution
        The current code solution (Pydantic object)
        Has: prefix, imports, code
    
    error: str
        Control flow flag: "yes" if code failed, "no" if passed
        Used by decide_to_finish to route the graph
    
    iterations: int
        How many generation attempts we've made
        Prevents infinite loops
    """
    question: str
    messages: List
    generation: CodeSolution
    error: str
    iterations: int


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Maximum retry attempts
MAX_ITERATIONS = 3


# ==============================================================================
# CODE GENERATION PROMPT
# ==============================================================================

CODE_GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a coding assistant with expertise in the DataProcessor library.

Here is the full documentation for the library:
-------
{context}
-------

Your task:
1. Answer the user's coding question based on the documentation above
2. Ensure any code you provide can be executed
3. All imports must be valid
4. Structure your answer with:
   - prefix: A description of your approach
   - imports: The import statements (just imports, nothing else)
   - code: The functioning code (without imports)

IMPORTANT: 
- Only use classes/functions documented above
- Make sure imports match what's in the documentation
- Test your logic mentally before responding"""
    ),
    ("placeholder", "{messages}"),
])


# ==============================================================================
# NODE 1: GENERATE CODE
# ==============================================================================
#
# This node calls the LLM to generate a code solution.
# 
# KEY INSIGHT: If there was a previous error (error="yes"), the error
# message is already in the messages list. The LLM sees this and can
# learn from the mistake.
#
# ==============================================================================

def generate(state: GraphState) -> GraphState:
    """
    Generate a code solution using the LLM.
    
    WHAT THIS DEMONSTRATES:
    - Structured output with Pydantic
    - Context stuffing (docs passed in prompt)
    - Error feedback via message history
    
    If this is a retry (error="yes"), we add a "try again" message
    to make it explicit that the LLM should fix the previous attempt.
    """
    print(f"\n{'='*60}")
    print(f"GENERATING CODE (Iteration {state['iterations'] + 1})")
    print(f"{'='*60}")
    
    messages = state["messages"].copy()
    error = state.get("error", "")
    iterations = state["iterations"]
    
    # If retrying after an error, add instruction to try again
    if error == "yes":
        print("Previous attempt failed. Adding retry instruction...")
        messages.append((
            "user",
            "The previous code failed. Please try again, fixing the error. "
            "Make sure imports are correct and code executes properly."
        ))
    
    # Create the chain with structured output
    code_gen_chain = CODE_GEN_PROMPT | llm.with_structured_output(CodeSolution)
    
    # Generate the solution
    print(f"Question: {state['question']}")
    print("Calling LLM...")
    
    try:
        solution = code_gen_chain.invoke({
            "context": DOCUMENTATION_CONTEXT,
            "messages": messages
        })
        
        print(f"\n--- Generated Solution ---")
        print(f"Prefix: {solution.prefix[:100]}...")
        print(f"Imports:\n{solution.imports}")
        print(f"Code:\n{solution.code[:200]}...")
        print(f"--------------------------")
        
        # Add assistant response to message history
        messages.append((
            "assistant",
            f"{solution.prefix}\n\nImports:\n{solution.imports}\n\nCode:\n{solution.code}"
        ))
        
        return {
            "generation": solution,
            "messages": messages,
            "iterations": iterations + 1
        }
        
    except Exception as e:
        print(f"ERROR in generation: {e}")
        # Return a dummy solution that will fail validation
        return {
            "generation": CodeSolution(
                prefix="Generation failed",
                imports="",
                code=f"raise Exception('Generation failed: {e}')"
            ),
            "messages": messages,
            "iterations": iterations + 1,
            "error": "yes"
        }


# ==============================================================================
# NODE 2: CHECK CODE
# ==============================================================================
#
# This node ACTUALLY RUNS the generated code using Python's exec().
#
# Two tests:
# 1. exec(imports) - Can we import?
# 2. exec(imports + code) - Does it run without errors?
#
# If either fails, we capture the error message and add it to
# the message history so the LLM can learn from it.
#
# ==============================================================================

def check_code(state: GraphState) -> GraphState:
    """
    Validate the generated code by actually executing it.
    
    WHAT THIS DEMONSTRATES:
    - Real code validation (not just LLM self-assessment)
    - Error capture and feedback
    - Two-stage testing (imports, then execution)
    
    We use exec() which runs Python code from a string.
    The mock library is injected into the execution namespace.
    """
    print(f"\n{'='*60}")
    print("CHECKING CODE")
    print(f"{'='*60}")
    
    messages = state["messages"].copy()
    solution = state["generation"]
    iterations = state["iterations"]
    
    imports = solution.imports
    code = solution.code
    
    # Create execution namespace with our mock library
    exec_globals = {}
    
    # First, set up the mock library in the namespace
    try:
        exec(MOCK_LIBRARY_CODE, exec_globals)
    except Exception as e:
        print(f"ERROR setting up mock library: {e}")
        # This shouldn't happen, but handle it gracefully
    
    # ===== TEST 1: Check imports =====
    print("\n--- Test 1: Checking imports ---")
    print(f"Imports to test:\n{imports}")
    
    try:
        exec(imports, exec_globals)
        print("✓ Imports: PASSED")
    except Exception as e:
        print(f"✗ Imports: FAILED - {e}")
        
        # Add error to messages for LLM feedback
        error_msg = f"Your solution failed the import test. Error: {e}"
        messages.append(("user", error_msg))
        
        return {
            "generation": solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }
    
    # ===== TEST 2: Check code execution =====
    print("\n--- Test 2: Checking code execution ---")
    print(f"Code to test:\n{code[:200]}...")
    
    try:
        # Execute imports + code together
        full_code = imports + "\n" + code
        exec(full_code, exec_globals)
        print("✓ Execution: PASSED")
    except Exception as e:
        print(f"✗ Execution: FAILED - {e}")
        
        # Add error to messages for LLM feedback
        error_msg = f"Your solution failed the code execution test. Error: {e}"
        messages.append(("user", error_msg))
        
        return {
            "generation": solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }
    
    # Both tests passed!
    print("\n✓ ALL TESTS PASSED")
    return {
        "generation": solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no"
    }


# ==============================================================================
# DECISION FUNCTION: SHOULD WE CONTINUE?
# ==============================================================================
#
# This is the conditional edge that controls the loop.
#
# Exit conditions:
# 1. error == "no" (success!)
# 2. iterations >= MAX_ITERATIONS (give up)
#
# Otherwise: loop back to generate and try again
#
# ==============================================================================

def decide_to_finish(state: GraphState) -> Literal["end", "generate"]:
    """
    Decide whether to finish or retry.
    
    WHAT THIS DEMONSTRATES:
    - Conditional routing in LangGraph
    - Loop control based on state
    - Max iteration safety limit
    """
    error = state["error"]
    iterations = state["iterations"]
    
    print(f"\n--- Decision Point ---")
    print(f"Error: {error}")
    print(f"Iterations: {iterations}/{MAX_ITERATIONS}")
    
    if error == "no":
        print("Decision: SUCCESS - Code passed all tests!")
        return "end"
    
    if iterations >= MAX_ITERATIONS:
        print("Decision: MAX ITERATIONS - Returning best effort")
        return "end"
    
    print("Decision: RETRY - Will try again with error feedback")
    return "generate"


# ==============================================================================
# BUILD THE GRAPH
# ==============================================================================
#
# Graph structure:
#
#   START → generate → check_code → decide_to_finish
#                ↑                         │
#                │         "generate"      │
#                └─────────────────────────┘
#                                          │
#                              "end"       ▼
#                                         END
#
# ==============================================================================

def build_code_gen_graph():
    """
    Build the self-correcting code generation graph.
    
    KEY CONCEPTS:
    
    1. generate: LLM produces code with structured output
    2. check_code: Actually runs the code with exec()
    3. decide_to_finish: Routes to END or back to generate
    
    The self-correction happens because:
    - Error messages are added to state["messages"]
    - generate node sees full message history
    - LLM learns from previous mistakes
    """
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("generate", generate)
    builder.add_node("check_code", check_code)
    
    # Add edges
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "check_code")
    
    # Conditional edge for the loop
    builder.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "generate": "generate"
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
    Run the self-correcting code generation demonstration.
    
    WHAT YOU'LL SEE:
    
    1. LLM generates code based on documentation
    2. Code is actually executed with exec()
    3. If it fails, error is fed back and LLM retries
    4. Loop continues until success or max iterations
    """
    global llm
    
    print("\n" + "="*70)
    print("LANGGRAPH CODE GENERATION WITH SELF-CORRECTION DEMO")
    print("="*70)
    print("""
    This demo shows a self-correcting code generation agent:
    
    1. GENERATE: LLM produces code (structured output)
    2. CHECK: Code is actually executed with exec()
    3. FEEDBACK: Errors are fed back to LLM
    4. RETRY: LLM tries again with knowledge of what failed
    
    The key insight: Real validation + error feedback = better code
    
    Graph flow:
    ┌─────────────────────────────────────────────────┐
    │  START → [generate] → [check_code] → {decide}  │
    │               ↑                          │     │
    │               └──────── retry ───────────┘     │
    │                                   │            │
    │                                success/max     │
    │                                   ↓            │
    │                                  END           │
    └─────────────────────────────────────────────────┘
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
    graph = build_code_gen_graph()
    
    # ===== TEST QUESTIONS =====
    test_questions = [
        "How do I convert a string to uppercase using DataProcessor?",
        "How do I create a pipeline that first makes text uppercase then reverses it?",
        "How do I use the transform function to lowercase a string?",
    ]
    
    for i, question in enumerate(test_questions):
        print("\n" + "="*70)
        print(f"TEST {i+1}: {question}")
        print("="*70)
        
        # Initial state
        initial_state = {
            "question": question,
            "messages": [("user", question)],
            "generation": None,
            "error": "",
            "iterations": 0
        }
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        # Display results
        print("\n" + "-"*70)
        print("FINAL RESULT")
        print("-"*70)
        
        solution = final_state["generation"]
        error = final_state["error"]
        iterations = final_state["iterations"]
        
        print(f"Status: {'SUCCESS' if error == 'no' else 'FAILED (best effort)'}")
        print(f"Iterations used: {iterations}")
        
        print(f"\n--- Solution ---")
        print(f"Description: {solution.prefix}")
        print(f"\nImports:\n{solution.imports}")
        print(f"\nCode:\n{solution.code}")
        print("-"*70)
        
        # Actually run the final solution to show it works
        if error == "no":
            print("\n--- Executing Final Solution ---")
            exec_globals = {}
            exec(MOCK_LIBRARY_CODE, exec_globals)
            try:
                exec(solution.imports + "\n" + solution.code, exec_globals)
                print("✓ Solution executed successfully!")
            except Exception as e:
                print(f"✗ Execution error: {e}")
    
    # ===== RECAP =====
    print("\n" + "="*70)
    print("PATTERN RECAP")
    print("="*70)
    print(f"""
    SELF-CORRECTING CODE GENERATION:
    
    1. STRUCTURED OUTPUT:
       └── Pydantic model: CodeSolution(prefix, imports, code)
       └── LLM forced to return valid structure
       └── Easy to extract and test components
    
    2. REAL VALIDATION:
       └── exec(imports) - tests import statements
       └── exec(imports + code) - tests full execution
       └── Actual Python errors, not LLM guessing
    
    3. ERROR FEEDBACK LOOP:
       └── Errors added to message history
       └── LLM sees: "Your solution failed: ImportError..."
       └── Next attempt informed by real failure
    
    4. CONTROLLED ITERATION:
       └── Max {MAX_ITERATIONS} attempts
       └── Exit on success OR max iterations
       └── Returns best effort if can't succeed
    
    WHY THIS WORKS:
    - LLM + real validation > LLM self-assessment
    - Concrete errors > vague "try again"
    - Multiple attempts > single shot
    """)
    print("="*70 + "\n")


# ==============================================================================
# BONUS: DEMONSTRATE ERROR CORRECTION
# ==============================================================================

def demo_error_correction():
    """
    Demonstrate the error correction loop with a deliberately tricky question.
    """
    global llm
    
    print("\n" + "="*70)
    print("ERROR CORRECTION DEMO")
    print("="*70)
    print("""
    This demo uses a question that's likely to cause initial errors,
    showing how the self-correction loop recovers.
    """)
    
    llm = create_llm()
    graph = build_code_gen_graph()
    
    # A question that might trip up the LLM initially
    tricky_question = "Create a Pipeline that strips whitespace, then uppercases, then reverses the text. Test it with '  hello world  '."
    
    print(f"Question: {tricky_question}")
    print("\nWatch the iterations...\n")
    
    initial_state = {
        "question": tricky_question,
        "messages": [("user", tricky_question)],
        "generation": None,
        "error": "",
        "iterations": 0
    }
    
    final_state = graph.invoke(initial_state)
    
    print("\n" + "-"*70)
    print(f"Completed in {final_state['iterations']} iteration(s)")
    print(f"Final status: {'SUCCESS' if final_state['error'] == 'no' else 'BEST EFFORT'}")
    print("-"*70)


if __name__ == "__main__":
    main()
    
    # Uncomment to see error correction in action:
    # demo_error_correction()