from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.agent import agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat(payload: dict):
    user_input = payload["message"]
    result = agent(user_input)
    return {"response": result}
