"""Simple CLI for exercising the local chat API."""

from __future__ import annotations

import json
from textwrap import indent
from typing import Dict, List

import requests
import sseclient


API_ROOT = "http://127.0.0.1:50001"
CHAT_URL = f"{API_ROOT}/chat"
HISTORY_URL = f"{API_ROOT}/agents/{{agent_id}}/history"
STREAM = True


def fetch_history(agent_id: str) -> List[Dict]:
    """Retrieve chat history records for an agent."""

    resp = requests.get(HISTORY_URL.format(agent_id=agent_id), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("history", [])


def print_history(history: List[Dict]):
    """Display the chat history in a numbered list."""

    if not history:
        print("No history available.")
        return

    print("Current chat history:")
    for idx, message in enumerate(history):
        msg_id = message.get("message_id", "<unknown>")
        role = message.get("role", "?")
        content = message.get("content", "")
        print(f"[{idx}] ({role}) id={msg_id}")
        print(indent(content, "    "))


def update_history_message(agent_id: str, message_id: str, role: str | None = None, content: str | None = None) -> bool:
    """Send a PATCH request to update a specific history message."""

    payload: Dict[str, str] = {}
    if role:
        payload["role"] = role
    if content is not None:
        payload["content"] = content

    if not payload:
        print("Nothing to update.")
        return False

    resp = requests.put(
        f"{API_ROOT}/agents/{agent_id}/history/{message_id}",
        json=payload,
        timeout=10,
    )
    if resp.ok:
        print("Message updated.")
        return True

    print(f"Failed to update message: {resp.status_code} {resp.text}")
    return False


def delete_history_message(agent_id: str, message_id: str) -> bool:
    """Delete a history message via the API."""

    resp = requests.delete(
        f"{API_ROOT}/agents/{agent_id}/history/{message_id}",
        timeout=10,
    )
    if resp.ok:
        print("Message deleted.")
        return True

    print(f"Failed to delete message: {resp.status_code} {resp.text}")
    return False


def chat_round(agent_id: str, user_text: str):
    """Send one user input, stream assistant response back."""

    r = requests.post(
        CHAT_URL,
        json={
            "agent_id": agent_id,
            "user": user_text,
            "stream": True,
            "tool_calls_allowed": True,
        },
        stream=STREAM,
        timeout=10,
    )
    r.raise_for_status()
    client = sseclient.SSEClient(r)

    print("Assistant:", end=" ", flush=True)
    response_text = ""

    for ev in client.events():
        if ev.event == "chunk":
            print(ev.data, end="", flush=True)
            response_text += ev.data
        elif ev.event == "done":
            break

    print("\n")
    return response_text


def print_help():
    print(
        "Commands:\n"
        "  :history / :refresh    - Fetch and display the latest history.\n"
        "  :edit <n>              - Edit the nth message (prompts for new text/role).\n"
        "  :delete <n>            - Delete the nth message.\n"
        "  :help                  - Show this help message.\n"
        "  exit / quit            - Leave the chat.\n"
    )


def interactive_chat():
    agent_id = "default"
    print("Interactive chat with agent:", agent_id)
    print("Type ':help' for commands, 'exit' to quit.\n")

    history_cache: List[Dict] = []

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

        if user_text.startswith(":"):
            command, *rest = user_text[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""
            command = command.lower()

            if command in {"history", "refresh"}:
                history_cache = fetch_history(agent_id)
                print_history(history_cache)
            elif command == "help":
                print_help()
            elif command == "edit":
                if not history_cache:
                    history_cache = fetch_history(agent_id)
                if not arg:
                    print("Usage: :edit <number>")
                    continue
                try:
                    idx = int(arg) - 1
                    message = history_cache[idx]
                except (ValueError, IndexError):
                    print("Invalid message number.")
                    continue
                print("Current content:")
                print(indent(message.get("content", ""), "    "))
                new_role = input("New role (leave blank to keep current): ").strip() or None
                new_content = input("New content (leave blank to keep current): ")
                if not new_content:
                    new_content = None
                if update_history_message(agent_id, message["message_id"], new_role, new_content):
                    history_cache = fetch_history(agent_id)
            elif command == "delete":
                if not history_cache:
                    history_cache = fetch_history(agent_id)
                if not arg:
                    print("Usage: :delete <number>")
                    continue
                try:
                    idx = int(arg) - 1
                    message = history_cache[idx]
                except (ValueError, IndexError):
                    print("Invalid message number.")
                    continue
                confirm = input(f"Delete message {arg}? (y/N): ").strip().lower()
                if confirm == "y":
                    if delete_history_message(agent_id, message["message_id"]):
                        history_cache = fetch_history(agent_id)
                else:
                    print("Deletion cancelled.")
            else:
                print(f"Unknown command: {command}")
                print_help()
            continue

        chat_round(agent_id, user_text)


if __name__ == "__main__":
    interactive_chat()
