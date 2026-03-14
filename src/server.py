from dotenv import load_dotenv
load_dotenv()
import os
from agent.advanced_rag import AdvancedRAG, AgentState
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI()
agent = AdvancedRAG()

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
    try:
        while True:
            data = await websocket.receive_text()
            print(f"[WS] 收到訊息: {data}")
            
            await websocket.send_text("\nAI助手: ")
            async for chunk in agent.stream(user_message=data, session_id=thread_id):
                await websocket.send_text(chunk)
            await websocket.send_text("\n")
    except WebSocketDisconnect:
        print("[WS] 使用者中斷連線")
    except Exception as e:
        print(f"[WS] 錯誤: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
    print(f"Server is running on {HOST}:{PORT}")