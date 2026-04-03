
# Full file explanation block (line-by-line):
# 1. import json: handles JSON serialization/deserialization for tool argument passing and results.
# 2. import requests: used to send HTTP requests to local tool servers at localhost:9000.
# 3. from openai import OpenAI: OpenAI SDK client class used to call the chat model.
# 4. from rag.rag_tf import rag_query: RAG retrieval + generation helper imported from the rag module.

# 5. client = OpenAI(): instantiate OpenAI client once, reused for each agent call.

# 6. tool_rag(query): wrapper function that calls rag_query(query) and returns its result.
# 7. tool_sentiment(text): wrapper that posts {'text': text} to sentiment endpoint and returns JSON response.
# 8. tool_add(a, b): wrapper that posts {'a': a, 'b': b} to add endpoint and returns JSON response.
# 9. TOOLS: list of tool specs for OpenAI function-calling with names, descriptions, and parameter schemas.

# 10. call_tool(name, args): routes the requested function name to the corresponding wrapper and arguments.
# 11. agent(goal): constructs messages, calls OpenAI with tools enabled, handles tool_calls, and returns final model content when done.
# 12. if __name__ == "__main__": example usage running agent with a compound task.
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

    # The agent loop continues until the model returns a result without tool calls.
    # 1. Send all messages to OpenAI chat completion with tools enabled.
    # 2. If response has no tool_calls, return text output (final answer).
    # 3. If tool_calls are present, execute each tool and append its result as a tool message.
    # 4. Repeat: model can then reason further with tool outputs in context.
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
