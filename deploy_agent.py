# deploy_agent.py
"""
Minimal deployment script for Vertex AI Agent Engine.

This script:
 - Loads your local agent class (MyAgent)
 - Uses Application Default Credentials (ADC)
 - Creates a remote agent deployment on Vertex AI Agent Engine
"""

from google.cloud import agentengine_v1beta as agentengine
from custom_agent import MyAgent

def main():
    # Initialize the client (uses ADC automatically)
    client = agentengine.AgentEnginesClient()

    # Instantiate your local agent
    local_agent = MyAgent()

    # Define minimal deployment config
    config = {
        "display_name": "my-minimal-agent",
        "description": "Minimal ADC-based agent deployment example.",
        "requirements": "requirements.txt",  # local file path
        "env_vars": {
            "PROJECT_ID": "your-gcp-project-id",
            "REGION": "us-central1",
        },
    }

    # Create (deploy) the remote agent
    remote_agent = client.agent_engines.create(
        agent=local_agent,
        config=config,
    )

    print("✅ Deployment started.")
    print(f"Agent resource name: {remote_agent.name}")
    print(f"Display name: {remote_agent.display_name}")

if __name__ == "__main__":
    main()
