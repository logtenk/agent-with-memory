import json
from pathlib import Path

import types
import pytest

from agent_host.app.orchestrator.tools import TOOLS, ToolSpec, register, list_tools_for_prompt
from agent_host.app import config as app_config
from agent_host.app.agents import profiles as agent_profiles


def test_registry_and_listing():
    # Registry exists and core tools are registered
    core = {"memory.insert", "memory.retrieve", "memory.update", "memory.delete",
            "agent.update_impression", "agent.update_mood", "agent.update_memory_summary"}
    names = set(TOOLS.keys())
    assert core.issubset(names)

    # list_tools_for_prompt returns schemas & descriptions
    tools = list_tools_for_prompt()
    by_name = {t["name"]: t for t in tools}
    assert "memory.insert" in by_name
    assert "input_schema" in by_name["memory.insert"]
    assert isinstance(by_name["memory.insert"]["input_schema"], dict)


def test_memory_insert_and_retrieve_forward_where(tmp_path, monkeypatch):
    """
    - monkeypatch CHROMA_PERSIST_ROOT to tmp
    - stub chroma_store to capture inputs
    - call tool handlers and assert behavior
    """
    # 1) redirect CHROMA_PERSIST_ROOT to a temp dir
    monkeypatch.setattr(app_config, "CHROMA_PERSIST_ROOT", tmp_path.as_posix(), raising=True)

    # 2) stub chroma functions
    calls = {"upsert": None, "query": None}

    def fake_upsert(agent_id, persist_root, items):
        calls["upsert"] = {"agent_id": agent_id, "persist_root": persist_root, "items": items}
        # fabricate IDs
        return [f"id_{i}" for i in range(len(items))]

    def fake_query(agent_id, persist_root, query, k, where=None):
        calls["query"] = {"agent_id": agent_id, "persist_root": persist_root,
                          "query": query, "k": k, "where": where}
        return [{"memory_id": "id_0", "text": "stub text", "metadata": {"tag":"food"}, "distance": 0.1}]

    import agent_host.app.memory.chroma_store as chroma_store_mod
    monkeypatch.setattr(chroma_store_mod, "upsert_memories", fake_upsert, raising=True)
    monkeypatch.setattr(chroma_store_mod, "query_memories", fake_query, raising=True)

    # 3) call memory.insert
    insert_payload = {
        "agent_id": "default",
        "items": [{"text": "User likes sushi.", "type": "preference", "date": 20250910, "tag": "food"}]
    }
    res_insert = TOOLS["memory.insert"].handler(insert_payload)
    assert res_insert["ok"] is True
    assert res_insert["ids"] == ["id_0"]
    assert calls["upsert"]["items"][0]["text"] == "User likes sushi."

    # 4) call memory.retrieve with a where filter
    where = {"tag": "food", "type": "preference"}  # friendly multi-field form
    res_retrieve = TOOLS["memory.retrieve"].handler({
        "agent_id": "default",
        "query": "sushi",
        "k": 5,
        "where": where
    })
    assert res_retrieve["ok"] is True
    assert isinstance(res_retrieve["results"], list)
    # Ensure where was forwarded (normalization to $and happens in chroma_store, not here)
    assert calls["query"]["where"] == where


def test_agent_update_mood_writes_profile(tmp_path, monkeypatch):
    import json
    from agent_host.app import config as app_config
    from agent_host.app.orchestrator.tools import TOOLS
    import agent_host.app.orchestrator.tools as tools_mod

    root = tmp_path / "agents"
    agent_dir = root / "default"
    agent_dir.mkdir(parents=True)
    profile_path = agent_dir / "profile.json"
    profile_path.write_text(json.dumps({
        "agent_id": "default",
        "character": "Helper",
        "impression_of_user": "Curious",
        "current_mood": "Neutral",
        "capabilities": ["rag_memory"],
        "memory_path": (agent_dir / "memory").as_posix(),
        "memory_summary": "empty",
        "tool_instructions": "be helpful"
    }), encoding="utf-8")

    # Patch BOTH: the config module (for any code that reads it dynamically)
    # and the already-imported constant in tools.py (import-time snapshot)
    monkeypatch.setattr(app_config, "CHROMA_PERSIST_ROOT", root.as_posix(), raising=True)
    monkeypatch.setattr(tools_mod, "CHROMA_PERSIST_ROOT", root.as_posix(), raising=True)

    res = TOOLS["agent.update_mood"].handler({"agent_id": "default", "current_mood": "Focused"})
    assert res["ok"] is True

    updated = json.loads(profile_path.read_text(encoding="utf-8"))
    assert updated["current_mood"] == "Focused"
