import asyncio
from agent.graph import graph
from agent.state import AgentState
from db.vector.db import milvus_store

async def main():
    await milvus_store.get_store()
    
    thread_id = "course-consultant-thread"
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        user_input = input("\n請輸入您的問題 (或輸入 'exit' 退出): ")
        if user_input.lower() == 'exit':
            break

        input_state = AgentState(question=user_input)
        result = await graph.ainvoke(input_state, config=config)
        print(f"\nAI助手: {result['generation']}\n")

if __name__ == "__main__":
    asyncio.run(main())