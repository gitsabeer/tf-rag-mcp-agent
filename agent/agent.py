import json
import requests
from openai import OpenAI
from rag.rag_tf import rag_query

client = OpenAI()

def tool_rag(query):
    return rag_query(query)

def tool_sentiment(text):
    r = requests.post("http://localhost:9000/sentiment", json={"text": text})
    return r.json()

def tool_add(a, b):
    r = requests.post("http://localhost:9000/add", json={"a": a, "b": b})
    return r.json()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search knowledge using TensorFlow RAG.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sentiment",
            "description": "Classify sentiment using TensorFlow model.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}
        }
    }
]

def call_tool(name, args):
    if name == "rag_search":
        return tool_rag(args["query"])
    if name == "sentiment":
        return tool_sentiment(args["text"])
    if name == "add":
        return tool_add(args["a"], args["b"])

def agent(goal):
    messages = [
        {"role": "system", "content": "You are an agent that uses tools intelligently."},
        {"role": "user", "content": goal}
    ]

    while True:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            result = call_tool(name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

if __name__ == "__main__":
    print(agent("Explain RAG using the knowledge base, then classify the sentiment of 'I love TensorFlow', then add 5 and 7."))
