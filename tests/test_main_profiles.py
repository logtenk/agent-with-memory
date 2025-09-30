import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agent_host.app.models import AgentProfile
from agent_host.app import config as config_module
import agent_host.app.main as main_module


def test_profile_roundtrip(tmp_path, monkeypatch):
    persist_root = tmp_path / "agents"
    monkeypatch.setattr(config_module, "CHROMA_PERSIST_ROOT", str(persist_root))
    monkeypatch.setattr(main_module, "CHROMA_PERSIST_ROOT", str(persist_root))

    agent_id = "test-agent"
    profile_dir = persist_root / agent_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    original_profile = AgentProfile(
        agent_id=agent_id,
        character="Helpful assistant",
        impression_of_user="Curious developer",
        current_mood="calm",
        capabilities=["chat", "code"],
        memory_path="memory/test-agent",
        memory_summary="Knows about codebases",
        tool_instructions="Use tools wisely",
    )

    profile_path = profile_dir / "profile.json"
    profile_path.write_text(json.dumps(original_profile.model_dump()))

    client = TestClient(main_module.app)

    get_response = client.get(f"/agents/{agent_id}")
    assert get_response.status_code == 200
    assert get_response.json() == original_profile.model_dump()

    updated_profile = original_profile.model_copy(
        update={
            "current_mood": "excited",
            "memory_summary": "Updated knowledge",
            "tool_instructions": "Use tools carefully",
        }
    )

    post_response = client.post(f"/agents/{agent_id}", json=updated_profile.model_dump())
    assert post_response.status_code == 200
    assert post_response.json() == {"ok": True}

    persisted = json.loads(profile_path.read_text())
    assert persisted == updated_profile.model_dump()

    refreshed_response = client.get(f"/agents/{agent_id}")
    assert refreshed_response.status_code == 200
    assert refreshed_response.json() == updated_profile.model_dump()
