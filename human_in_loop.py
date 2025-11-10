from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import tool
from typing import Optional
import asyncio

# Define tools for human interaction
@tool
async def request_human_input(ctx: RunContext, question: str) -> str:
    """Request input from human operator."""
    print(f"\n🤖 AGENT ASKS: {question}")
    response = input("👤 HUMAN INPUT: ")
    return response

@tool  
async def request_human_approval(ctx: RunContext, action: str, rationale: str) -> bool:
    """Request approval from human operator before proceeding."""
    print(f"\n🤖 AGENT SEEKS APPROVAL: {action}")
    print(f"📝 REASON: {rationale}")
    response = input("👤 APPROVE? (y/n): ").lower().strip()
    return response in ['y', 'yes']

# Create the agent with human interaction capabilities
agent = Agent(
    model='openai:gpt-4',
    tools=[request_human_input, request_human_approval],
    system_prompt="""You are an assistant that knows when to ask for human guidance.
    Use the available tools to:
    1. Ask for human input when you need clarification
    2. Request approval before taking significant actions
    3. Explain what you're doing so humans can monitor your progress
    
    Always be transparent about your reasoning and next steps."""
)

async def main():
    """Run the agent with human interaction demo."""
    print("=== Pydantic AI Human-in-the-Loop Demo ===")
    print("Starting agent execution...")
    
    # Example task that requires human guidance
    task = """
    Help me plan a marketing campaign for a new product. 
    I need you to:
    1. Suggest target audience segments
    2. Propose marketing channels
    3. Estimate budget requirements
    
    Ask for human input when you need clarification or approval for significant decisions.
    """
    
    result = await agent.run(task)
    
    print(f"\n=== FINAL RESULT ===")
    print(result.data)

if __name__ == "__main__":
    asyncio.run(main())