from agent_host.app.orchestrator import history as H

def test_history_roundtrip(tmp_path):
    root = tmp_path.as_posix()
    agent = "a1"

    # Initially empty
    msgs = H.load_history(root, agent, max_pairs=20)
    assert msgs == []

    # Append a user and assistant turn
    H.append_turn(root, agent, "user", "hi")
    H.append_turn(root, agent, "assistant", "hello")

    msgs = H.load_history(root, agent, max_pairs=20)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "hi"
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "hello"

    # External edit (simulate manual file change)
    H.write_all(root, agent, [{"role": "user", "content": "edited"}])
    msgs2 = H.load_history(root, agent, max_pairs=20)
    assert msgs2 == [{"role": "user", "content": "edited"}]

    # Clear
    H.clear_history(root, agent)
    msgs3 = H.load_history(root, agent, max_pairs=20)
    assert msgs3 == []
