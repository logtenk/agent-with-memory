import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agent_host.app import config as config_module
import agent_host.app.main as main_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    persist_root = tmp_path / "agents"
    monkeypatch.setattr(config_module, "CHROMA_PERSIST_ROOT", str(persist_root))
    monkeypatch.setattr(main_module, "CHROMA_PERSIST_ROOT", str(persist_root))
    return TestClient(main_module.app)


def _sample_profile(agent_id: str = "test-agent") -> dict:
    return {
        "agent_id": agent_id,
        "character": "Helpful assistant",
        "impression_of_user": "Curious developer",
        "current_mood": "calm",
        "capabilities": ["chat", "code"],
        "memory_path": "memory/test-agent",
        "memory_summary": "Knows about codebases",
        "tool_instructions": "Use tools wisely",
    }


def test_agent_crud_flow(client, tmp_path):
    profile = _sample_profile()

    create_resp = client.post("/agents", json=profile)
    assert create_resp.status_code == 201
    payload = create_resp.json()
    assert payload["ok"] is True
    assert payload["profile"] == profile

    profile_path = tmp_path / "agents" / profile["agent_id"] / "profile.json"
    assert profile_path.exists()
    assert json.loads(profile_path.read_text()) == profile

    list_resp = client.get("/agents")
    assert list_resp.status_code == 200
    assert list_resp.json() == {"agents": [profile]}

    get_resp = client.get(f"/agents/{profile['agent_id']}")
    assert get_resp.status_code == 200
    assert get_resp.json() == profile

    patch = {
        "current_mood": "excited",
        "memory_summary": "Updated knowledge",
    }
    patch_resp = client.patch(f"/agents/{profile['agent_id']}", json=patch)
    assert patch_resp.status_code == 200
    updated = patch_resp.json()["profile"]
    assert updated["current_mood"] == "excited"
    assert updated["memory_summary"] == "Updated knowledge"

    persisted = json.loads(profile_path.read_text())
    assert persisted == updated

    delete_resp = client.delete(f"/agents/{profile['agent_id']}")
    assert delete_resp.status_code == 204
    assert not profile_path.exists()

    list_resp = client.get("/agents")
    assert list_resp.status_code == 200
    assert list_resp.json() == {"agents": []}


def test_create_duplicate_agent_conflict(client):
    profile = _sample_profile()

    first = client.post("/agents", json=profile)
    assert first.status_code == 201

    second = client.post("/agents", json=profile)
    assert second.status_code == 409


def test_update_missing_agent(client):
    patch_resp = client.patch("/agents/missing-agent", json={"current_mood": "tired"})
    assert patch_resp.status_code == 404


def test_delete_missing_agent(client):
    delete_resp = client.delete("/agents/missing-agent")
    assert delete_resp.status_code == 404
