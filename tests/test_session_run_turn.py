import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agent_host.app.orchestrator import history
from agent_host.app.orchestrator import session

@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_run_turn_memory_insert_clears_history(tmp_path, monkeypatch):
    # point persistence to the tmp path
    monkeypatch.setattr(session, "CHROMA_PERSIST_ROOT", str(tmp_path))

    # provide deterministic tool prompt list
    monkeypatch.setattr(
        session,
        "list_tools_for_prompt",
        lambda: [
            {
                "name": "memory.insert",
                "description": "store memory",
                "input_schema": {"type": "object"},
            }
        ],
    )

    # stub nonstream_chat to return primary assistant reply then post-turn tool call
    responses = [
        {"text": "hello user"},
        {
            "text": 'TOOL_CALL: {"name":"memory.insert","payload":{"items":[{"text":"remember"}]}}'
        },
    ]

    async def fake_nonstream_chat(*args, **kwargs):
        if not responses:
            raise AssertionError("nonstream_chat called more times than expected")
        return responses.pop(0)

    monkeypatch.setattr(session, "nonstream_chat", fake_nonstream_chat)

    # capture memory.insert invocations
    recorded_payloads = []

    def fake_memory_insert(payload):
        recorded_payloads.append(payload)
        return {"ok": True}

    monkeypatch.setitem(session.TOOLS, "memory.insert", SimpleNamespace(handler=fake_memory_insert))

    # track assistant append operations while preserving original behavior
    original_append_turn = session.H.append_turn
    assistant_appends = []

    def tracking_append(root, agent_id, role, content, **kwargs):
        if role == "assistant":
            assistant_appends.append(content)
        return original_append_turn(root, agent_id, role, content, **kwargs)

    monkeypatch.setattr(session.H, "append_turn", tracking_append)

    monkeypatch.setattr(session, "build_system_prompt", lambda profile: "system prompt")

    profile = {
        "agent_id": "agent-1",
        "character": "Test Agent",
        "impression_of_user": "friendly",
        "current_mood": "curious",
        "capabilities": ["memory"],
        "memory_summary": "none",
    }

    events = []
    async for event in session.run_turn(profile, "hello", allow_tools=True, stream=False):
        events.append(event)

    # ensure both events (token + done) were produced
    assert [e["type"] for e in events] == ["token", "done"]

    done_event = events[-1]
    used_tools = done_event["data"]["used_tools"]

    assert recorded_payloads == [
        {"items": [{"text": "remember"}]}
    ], "memory.insert should receive payload from post-turn phase"
    assert any(tool["name"] == "memory.insert" for tool in used_tools)

    assert len(assistant_appends) == 1, "assistant message should be appended once"

    turns_after = history.load_all_turns(str(tmp_path), "agent-1")
    assert turns_after == [], "history should be cleared after memory insertion"
