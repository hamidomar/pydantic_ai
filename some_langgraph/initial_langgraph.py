from langchain_google_vertexai import ChatVertexAI

# Initialize the model - it will automatically use your ADC credentials
llm = ChatVertexAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=6,
)

# Simple usage
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "What is the capital of France?"),
]

response = llm.invoke(messages)
print(response.content)