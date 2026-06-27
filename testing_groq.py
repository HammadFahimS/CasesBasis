import sys
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python testing_groq.py \"your prompt here\"")
    sys.exit(1)

prompt = " ".join(sys.argv[1:])
client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)
