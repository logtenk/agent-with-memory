import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agent_host.app.orchestrator import history as H
from agent_host.app import main as main_app


def _prompt_view(msgs):
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def test_history_roundtrip(tmp_path):
    root = tmp_path.as_posix()
    agent = "a1"

    # Initially empty
    msgs = H.load_history(root, agent, max_pairs=20)
    assert msgs == []

    # Append a user and assistant turn
    user = H.append_turn(root, agent, "user", "hi")
    assist = H.append_turn(root, agent, "assistant", "hello")

    msgs = H.load_history(root, agent, max_pairs=20)
    assert len(msgs) == 2
    assert _prompt_view(msgs)[0] == {"role": "user", "content": "hi"}
    assert _prompt_view(msgs)[1] == {"role": "assistant", "content": "hello"}
    assert user["message_id"] != assist["message_id"]
    assert user["created_at"] <= user["updated_at"]

    # External edit (simulate manual file change)
    H.write_all(root, agent, [{"role": "user", "content": "edited"}])
    msgs2 = H.load_history(root, agent, max_pairs=20)
    assert len(msgs2) == 1
    assert _prompt_view(msgs2)[0] == {"role": "user", "content": "edited"}
    assert "message_id" in msgs2[0]

    # Clear
    H.clear_history(root, agent)
    msgs3 = H.load_history(root, agent, max_pairs=20)
    assert msgs3 == []


def test_update_and_delete_turn(tmp_path):
    root = tmp_path.as_posix()
    agent = "agent"

    first = H.append_turn(root, agent, "user", "hi")
    second = H.append_turn(root, agent, "assistant", "hello")

    updated = H.update_turn(root, agent, first["message_id"], {"content": "updated"})
    assert updated is True
    after_update = H.load_all_turns(root, agent)
    updated_first = next(m for m in after_update if m["message_id"] == first["message_id"])
    assert updated_first["content"] == "updated"
    assert updated_first["updated_at"] >= first["updated_at"]

    deleted = H.delete_turn(root, agent, second["message_id"])
    assert deleted is True
    remaining = H.load_all_turns(root, agent)
    assert len(remaining) == 1
    assert remaining[0]["message_id"] == first["message_id"]

    missing = H.delete_turn(root, agent, "does-not-exist")
    assert missing is False


def test_history_endpoints(tmp_path, monkeypatch):
    root = tmp_path.as_posix()
    agent = "agent-api"

    monkeypatch.setattr(main_app, "CHROMA_PERSIST_ROOT", root, raising=False)
    client = TestClient(main_app.app)

    first = H.append_turn(root, agent, "user", "hi")
    second = H.append_turn(root, agent, "assistant", "hello")

    resp = client.get(f"/agents/{agent}/history")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert len(payload["history"]) == 2
    assert payload["history"][0]["message_id"] == first["message_id"]

    resp = client.put(
        f"/agents/{agent}/history/{first['message_id']}",
        json={"content": "updated"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = client.get(f"/agents/{agent}/history")
    payload = resp.json()
    assert payload["history"][0]["content"] == "updated"

    resp = client.delete(f"/agents/{agent}/history/{second['message_id']}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = client.get(f"/agents/{agent}/history")
    payload = resp.json()
    assert len(payload["history"]) == 1
    assert payload["history"][0]["message_id"] == first["message_id"]
