from langchain_core.messages import AIMessageChunk

from agent import agent

SESSION_ID = "default"


def main() -> None:
    print("Chat with the agent. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("Agent: ", end="", flush=True)
        for message_chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": SESSION_ID}},
            stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                print(message_chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
