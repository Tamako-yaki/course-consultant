import asyncio
from RAG import AdvancedRAG

async def main():
    rag = AdvancedRAG()
    while True:
        user_message = input("\n請輸入您的問題 (或輸入 'exit' 退出): ")
        if user_message.lower() == "exit":
            print("退出程式。")
            break
        async for chunk in rag.stream(user_message=user_message):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())