"""
CLI-only Human-in-the-loop example for pydantic_ai

Save as: pydantic_ai/agent_cli_only.py

Usage:
  python pydantic_ai/agent_cli_only.py

Notes:
- Replace the `Agent` stub with your real pydantic_ai Agent import/constructor.
- Implement `google_auth_and_instantiate()` to perform Google provider auth and
  return a properly constructed Agent(model, **params).
"""
from typing import Any, Dict, Optional
import time
import uuid

# -------------------- Agent & Auth stubs --------------------
# Replace or implement these with your real code.

class Agent:
    """
    Minimal Agent interface expected by this example.

    Replace with: Agent(model, <other_params>)
    """
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.state: Dict[str, Any] = {}

    def run_step(self, input_text: str) -> Dict[str, Any]:
        """
        Run a single step of the agent. Return a dict describing the outcome.

        Expected return shapes used by this example:
          {"action": "ask_human", "proposal": ..., "reason": ...}
          {"action": "complete", "result": ...}
        """
        # Dummy behavior: short "thinking" delay, then ask human if 'risk' in input
        time.sleep(0.25)
        if 'risk' in input_text.lower():
            return {"action": "ask_human", "proposal": input_text, "reason": "high_risk_decision"}
        return {"action": "complete", "result": f"processed: {input_text}"}

    def apply_feedback(self, token: str, approved: bool, notes: Optional[str] = None) -> None:
        """
        Apply human feedback to the agent's internal state or runloop.
        Replace or extend this to hook into your agent's real feedback APIs.
        """
        self.state['last_feedback'] = {"token": token, "approved": approved, "notes": notes, "at": time.time()}


def google_auth_and_instantiate() -> Agent:
    """
    Placeholder for authentication + Agent instantiation using Google provider.

    Implement Google OAuth / service account / ADC auth here and construct the
    Agent instance with the provider-backed client/credentials.

    Example pseudocode (do not treat as runnable):
      credentials = google.auth.default()
      client = SomeProviderClient(credentials=credentials)
      return Agent(model='gpt-xyz', client=client, other_param=...)

    Currently returns a placeholder Agent. Replace before production use.
    """
    # TODO: implement real Google provider auth + create Agent(model, **params)
    return Agent(model='placeholder-google-model')

# -------------------- CLI human-in-the-loop loop --------------------

def cli_loop(agent: Agent):
    print("pydantic_ai — CLI human-in-the-loop demo (CLI-only)")
    print("Type 'quit' to exit. Type input containing 'risk' to trigger a human pause.\n")

    while True:
        try:
            user_input = input("[User] > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.strip().lower() in ('quit', 'exit'):
            print("Goodbye.")
            break

        outcome = agent.run_step(user_input)

        action = outcome.get('action')
        if action == 'ask_human':
            token = str(uuid.uuid4())
            proposal = outcome.get('proposal')
            reason = outcome.get('reason')

            print("\n--- AGENT PAUSE: HUMAN REVIEW REQUIRED ---")
            print(f"Token : {token}")
            print(f"Reason: {reason}")
            print("Proposal:")
            print(proposal)
            print("-----------------------------------------\n")

            # Pause: collect human decision
            while True:
                decision = input("Approve proposal? (y/n) > ").strip().lower()
                if decision in ('y', 'n'):
                    break
                print("Please type 'y' or 'n'.")

            notes = input("Optional notes (press Enter to skip) > ").strip() or None
            approved = (decision == 'y')

            # Apply human feedback to the agent (hook into real agent API here)
            agent.apply_feedback(token=token, approved=approved, notes=notes)

            if approved:
                print("\nHuman approved. Agent will apply the proposal and continue.\n")
                # If agent needs to perform the approved action immediately, you would
                # call into the agent's execution method here (not shown in stub).
            else:
                print("\nHuman rejected. Agent will replan or abort per its policy.\n")
                # Trigger agent replanning or safe-fail behavior as appropriate.

        elif action == 'complete':
            print("Agent result:", outcome.get('result'), "\n")
        else:
            print("Unknown agent action:", action, "\n")


if __name__ == '__main__':
    # Instantiate agent via the Google auth + instantiation stub (replace as needed)
    agent = google_auth_and_instantiate()
    cli_loop(agent)
