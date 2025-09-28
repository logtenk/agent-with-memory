import requests
import sseclient
import json

API_URL = "http://127.0.0.1:8080/chat"

def chat_round(agent_id: str, user_text: str):
    """Send one user input, stream assistant response back."""
    r = requests.post(
        API_URL,
        json={
            "agent_id": agent_id,
            "user": user_text,
            "stream": True,
            "tool_calls_allowed": True
        },
        stream=True,
    )
    client = sseclient.SSEClient(r)

    print("Assistant:", end=" ", flush=True)
    response_text = ""
    used_tools = []

    for ev in client.events():
        if ev.event == "chunk":
            print(ev.data, end="", flush=True)
            response_text += ev.data
        elif ev.event == "done":
            meta = json.loads(ev.data)
            used_tools = meta.get("used_tools", [])
            break

    print("\n")
    if used_tools:
        print("[Tools used this turn]:")
        print(json.dumps(used_tools, indent=2, ensure_ascii=False))

    return response_text, used_tools

def interactive_chat():
    agent_id = "default"
    print("Interactive chat with agent:", agent_id)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        chat_round(agent_id, user_text)

if __name__ == "__main__":
    interactive_chat()
