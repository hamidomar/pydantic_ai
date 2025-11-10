"""
Human-in-the-loop integration scaffolder (single-file generator)

This script creates a directory `human_in_loop_examples/` with subdirectories for
framework-specific minimal examples that demonstrate:
  * Pausing agent execution to wait for a human
  * Incorporating human feedback to guide agent behavior
  * Providing simple interfaces for human monitoring and interaction

The `pydantic_ai` example is implemented with two interfaces:
  - CLI-based pause + feedback (synchronous)
  - FastAPI-based HTTP endpoint for async pause/response + monitoring

Two other framework placeholders are included to show how to add more.

Usage:
  python human_in_loop_integration.py --output-dir ./human_in_loop_examples

Note: this is a scaffolding generator. The produced examples assume you have
`pydantic_ai` installed for the pydantic_ai example. The FastAPI example requires
`fastapi` and `uvicorn` if you choose to run it.

"""
from pathlib import Path
import argparse
import textwrap

ROOT_TEMPLATE = """
# {title}

{desc}
"""

PYDANTIC_CLI = """"""# pydantic_ai CLI human-in-the-loop example

# This example demonstrates a minimal pattern for pausing agent execution,
# collecting human feedback, and continuing execution based on that feedback.

# Assumptions:
#   - `pydantic_ai` exposes an Agent-like interface. Replace the stubbed
#     `SimpleAgent` below with your real agent class or import from your
#     project's module.

import time
from typing import Any, Dict

# ----- STUB: replace with actual import from your pydantic_ai package -----
class SimpleAgent:
    def __init__(self, name: str = "pydantic-agent"):
        self.name = name
        self.state: Dict[str, Any] = {}

    def step(self, input_data: str) -> Dict[str, Any]:
        # pretend the agent reasons for a second and returns a result that
        # indicates it wants human approval for a branch
        time.sleep(1)
        # If agent sees the word 'risk' it asks for human approval
        if 'risk' in input_data.lower():
            return {"action": "ask_human", "reason": "high_risk_decision", "proposal": input_data}
        return {"action": "complete", "result": f"processed: {input_data}"}

# ----- CLI loop that demonstrates pause-and-feedback -----

def cli_run():
    agent = SimpleAgent()
    print("pydantic_ai CLI human-in-the-loop demo")
    print("Type 'quit' to exit. Type a sentence containing 'risk' to trigger human approval.")

    while True:
        user_input = input("[User] > ")
        if not user_input:
            continue
        if user_input.strip().lower() in ('quit','exit'):
            break

        out = agent.step(user_input)

        if out.get('action') == 'ask_human':
            print("\nAgent requests human review:\n")
            print("  Proposal:", out.get('proposal'))
            print("  Reason:", out.get('reason'))
            # Pause for human decision
            while True:
                decision = input("Approve proposal? (y/n) > ").strip().lower()
                if decision in ('y','n'):
                    break
                print("Please type 'y' or 'n'.")

            if decision == 'y':
                print("Human approved. Agent continues and applies the proposal.")
                # In a real agent you might call agent.apply_feedback(...)
                agent.state['last_decision'] = 'approved'
                print("Result: proposal applied\n")
            else:
                print("Human rejected. Agent will choose an alternate plan.\n")
                agent.state['last_decision'] = 'rejected'
                # agent could be instructed to replan here

        else:
            print("Agent result:", out.get('result'))

if __name__ == '__main__':
    cli_run()
""""""

PYDANTIC_FASTAPI = """"""# pydantic_ai FastAPI human-in-the-loop example

# Run with:
#   uvicorn agent_fastapi:app --reload --port 8001

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uuid
import threading
import time

app = FastAPI(title='pydantic_ai HIL (FastAPI)')

# Simple in-memory store for paused decisions
PAUSES: Dict[str, Dict] = {}

# ----- STUB agent (replace with real pydantic_ai agent) -----
class SimpleAgent:
    def __init__(self):
        self.id = 'pydantic-agent'

    def plan(self, prompt: str):
        # signal pause if 'risk' in prompt
        if 'risk' in prompt.lower():
            token = str(uuid.uuid4())
            return {"pause": True, "pause_token": token, "proposal": prompt, "reason": "requires_human"}
        return {"pause": False, "result": f"ok: {prompt}"}

agent = SimpleAgent()

class RunRequest(BaseModel):
    prompt: str

class PauseResponse(BaseModel):
    token: str
    proposal: str
    reason: str

class Decision(BaseModel):
    approved: bool
    notes: Optional[str] = None

@app.post('/run')
async def run(req: RunRequest):
    plan = agent.plan(req.prompt)
    if plan.get('pause'):
        token = plan['pause_token']
        PAUSES[token] = {"proposal": plan['proposal'], "reason": plan['reason'], "created_at": time.time(), "decision": None}
        return PauseResponse(token=token, proposal=plan['proposal'], reason=plan['reason'])
    return {"result": plan['result']}

@app.get('/pauses')
async def list_pauses():
    # return current pauses for monitoring
    return PAUSES

@app.post('/decide/{token}')
async def decide(token: str, d: Decision):
    entry = PAUSES.get(token)
    if not entry:
        raise HTTPException(status_code=404, detail='pause token not found')
    entry['decision'] = {'approved': d.approved, 'notes': d.notes, 'decided_at': time.time()}
    # In a real system you'd notify the agent runloop to continue. This demo simulates that.
    return {"status": "recorded", "token": token}

# Simple background cleaner (optional)
def cleaner():
    while True:
        now = time.time()
        stale = [k for k,v in PAUSES.items() if now - v['created_at'] > 60*60]
        for k in stale:
            PAUSES.pop(k, None)
        time.sleep(60)

threading.Thread(target=cleaner, daemon=True).start()
""""""

PLACEHOLDERS = """"""# placeholders for other frameworks

# Example placeholder for 'framework_x' - shows recommended integration points

README = """
# Framework X - Human-in-the-loop example (placeholder)

This file describes where to add code to support:
 - Pausing the agent for a human decision
 - Accepting human feedback and applying it to the agent
 - Exposing a monitoring endpoint/UI

Suggested approach:
 - Provide a `pause()` primitive on the agent run context that creates an external ID
 - Provide a `resume(pause_id, decision)` API that the UI/human can call
 - Emit events or logs to a /monitoring endpoint for visibility
"""
""""""

GENERATOR = f"""
# Auto-generated scaffolding for human-in-the-loop examples
# Generated by human_in_loop_integration.py

"""

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf8')


def generate(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # pydantic_ai CLI example
    pyd_cli_path = output_dir / 'pydantic_ai' / 'agent_cli.py'
    write_file(pyd_cli_path, PYDANTIC_CLI)

    # pydantic_ai FastAPI example
    pyd_fastapi_path = output_dir / 'pydantic_ai' / 'agent_fastapi.py'
    write_file(pyd_fastapi_path, PYDANTIC_FASTAPI)

    # README for pydantic_ai
    pyd_readme = textwrap.dedent('''
    # pydantic_ai human-in-the-loop examples

    This folder contains two minimal examples demonstrating human-in-the-loop
    patterns with `pydantic_ai`:

    - `agent_cli.py`: synchronous CLI pause + human feedback
    - `agent_fastapi.py`: HTTP-driven pause + /pauses monitoring endpoint

    Replace the stubbed SimpleAgent with your real agent class and wire the
    `pause` / `decision` flow to your agent runloop.
    ''')
    write_file(output_dir / 'pydantic_ai' / 'README.md', pyd_readme)

    # placeholders for two other frameworks
    write_file(output_dir / 'framework_x' / 'README.md', PLACEHOLDERS)
    write_file(output_dir / 'framework_y' / 'README.md', PLACEHOLDERS)

    # top-level README
    top_readme = textwrap.dedent('''
    # Human-in-the-loop integration examples

    Generated examples to explore how frameworks support:
      * Pausing agent execution to wait for a human
      * Incorporating human feedback to guide agent behavior
      * Exposing simple monitoring / interaction surfaces for humans

    Run the pydantic_ai CLI example:
      python pydantic_ai/agent_cli.py

    Run the FastAPI example (if you have fastapi + uvicorn):
      uvicorn pydantic_ai/agent_fastapi:app --reload --port 8001

    Replace the provided SimpleAgent stubs with your real agent implementations
    and connect `pause`/`resume` primitives to your agent runloop.
    ''')
    write_file(output_dir / 'README.md', top_readme)

    print(f"Wrote examples to: {output_dir.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', default='human_in_loop_examples')
    args = parser.parse_args()
    generate(Path(args.output_dir))
""
