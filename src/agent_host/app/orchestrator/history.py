import os, json, io
from typing import List, Dict

def _hist_path(root: str, agent_id: str) -> str:
    base = os.path.join(root, agent_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "chat_history.jsonl")

def load_history(root: str, agent_id: str, max_pairs: int = 20) -> List[Dict[str, str]]:
    path = _hist_path(root, agent_id)
    msgs: List[Dict[str, str]] = []
    if not os.path.exists(path):
        return msgs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "role" in obj and "content" in obj:
                    msgs.append({"role": obj["role"], "content": obj["content"]})
            except Exception:
                continue
    # keep last 2*max_pairs messages
    if len(msgs) > 2*max_pairs:
        msgs = msgs[-2*max_pairs:]
    return msgs

def append_turn(root: str, agent_id: str, role: str, content: str):
    path = _hist_path(root, agent_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

def write_all(root: str, agent_id: str, msgs: List[Dict[str, str]]):
    path = _hist_path(root, agent_id)
    with open(path, "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps({"role": m["role"], "content": m["content"]}, ensure_ascii=False) + "\n")

def clear_history(root: str, agent_id: str):
    path = _hist_path(root, agent_id)
    open(path, "w").close()
