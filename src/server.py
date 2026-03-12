from dotenv import load_dotenv
load_dotenv()
import os
from agent.graph import graph
from fastapi import FastAPI, WebSocket
from agent.state import AgentState

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    thread_id = "course-consultant-thread"
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    while True:
        data = await websocket.receive_text()
        input_state = AgentState(question=data)
        result = await graph.ainvoke(input_state, config=config)
        await websocket.send_text(f"\nAI助手: {result['generation']}\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
    print(f"Server is running on {HOST}:{PORT}")