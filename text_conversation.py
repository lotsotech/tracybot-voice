"""
Text-only conversation with Clawdbot
For testing the integration without voice components
"""

import asyncio
from clawdbot_agent import ClawdbotAgent, ClawdbotAgentConfig


async def text_chat():
    """Simple text chat interface"""
    print("💬 Clawdbot Text Chat (type 'quit' or 'exit' to end)\n")

    agent = ClawdbotAgent(ClawdbotAgentConfig(agent_id="main"))

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            print("🤖 Clawdbot: ", end="", flush=True)
            async for gr in agent.generate_response(user_input, conversation_id="text-chat"):
                print(gr.message.text, end=" ", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(text_chat())
