import asyncio
from agent.graph import graph
from agent.state import AgentState

async def main():

    thread_id = "course-consultant-thread"
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        user_input = input("請輸入您的問題 (或輸入 'exit' 退出): ")
        if user_input.lower() == 'exit':
            break

        input_state = AgentState(question=user_input)
        result = await graph.ainvoke(input_state, config=config)
        print("AI助手:", result["generation"])

if __name__ == "__main__":
    asyncio.run(main())