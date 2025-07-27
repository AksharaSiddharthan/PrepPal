from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create a client
client = OpenAI(api_key=api_key)

# Send chat completion request
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello, who are you?"}
    ]
)

# Print the reply
print(response.choices[0].message.content)
