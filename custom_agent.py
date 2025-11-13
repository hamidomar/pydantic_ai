# custom_agent.py
"""
Minimal Vertex AI custom agent template
- Only implements query()
- Uses Application Default Credentials (ADC)
- Ready for deployment via Vertex Agent Engine
- Pydantic AI-compatible structure
"""

import os
import google.auth


class MyAgent:
    """Minimal Agent class with only query()"""

    def __init__(self, project_id: str | None = None, region: str | None = None):
        # Keep __init__ lightweight (pickle-safe)
        self.project_id = project_id or os.environ.get("PROJECT_ID")
        self.region = region or os.environ.get("REGION", "us-central1")
        self.credentials = None
        self._is_setup = False

    def set_up(self) -> None:
        """Obtain ADC credentials (done at container startup)."""
        creds, default_project = google.auth.default()
        self.credentials = creds
        if not self.project_id and default_project:
            self.project_id = default_project
        self._is_setup = True

    def query(self, inputs: dict) -> dict:
        """
        Minimal query implementation.
        Inputs and outputs are plain dicts, JSON-serializable.
        """
        if not self._is_setup:
            self.set_up()

        prompt = inputs.get("prompt", "").strip()
        max_tokens = int(inputs.get("max_tokens", 128))

        # Example logic — replace with your actual reasoning/model call
        response_text = f"[Echo: {prompt}] (max {max_tokens} tokens)"

        return {
            "response": response_text,
            "project_id": self.project_id,
            "region": self.region,
            "auth_acquired": bool(self.credentials),
        }
