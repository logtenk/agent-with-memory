import os
import json
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

def _hist_path(root: str, agent_id: str) -> str:
    base = os.path.join(root, agent_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "chat_history.jsonl")

def _now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "role" not in raw or "content" not in raw:
        raise ValueError("history records require role and content")
    record: Dict[str, Any] = {
        "message_id": raw.get("message_id") or uuid4().hex,
        "role": raw["role"],
        "content": raw["content"],
        "created_at": raw.get("created_at") or _now_ts(),
    }
    record["updated_at"] = raw.get("updated_at") or record["created_at"]
    for key, value in raw.items():
        if key not in record:
            record[key] = value
    return record

def _read_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                record = _normalize_record(obj)
            except Exception:
                continue
            records.append(record)
    return records

def load_all_turns(root: str, agent_id: str) -> List[Dict[str, Any]]:
    path = _hist_path(root, agent_id)
    return _read_records(path)

def load_history(root: str, agent_id: str, max_pairs: int = 20) -> List[Dict[str, Any]]:
    records = load_all_turns(root, agent_id)
    if len(records) > 2 * max_pairs:
        records = records[-2 * max_pairs:]
    return records

def append_turn(root: str, agent_id: str, role: str, content: str, *, message_id: str | None = None) -> Dict[str, Any]:
    path = _hist_path(root, agent_id)
    ts = _now_ts()
    record = {
        "message_id": message_id or uuid4().hex,
        "role": role,
        "content": content,
        "created_at": ts,
        "updated_at": ts,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

def write_all(root: str, agent_id: str, msgs: List[Dict[str, Any]]):
    path = _hist_path(root, agent_id)
    with open(path, "w", encoding="utf-8") as f:
        for m in msgs:
            record = _normalize_record(m)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def update_turn(root: str, agent_id: str, message_id: str, patch: Dict[str, Any]) -> bool:
    path = _hist_path(root, agent_id)
    records = _read_records(path)
    updated = False
    for rec in records:
        if rec["message_id"] == message_id:
            for key, value in patch.items():
                if key == "message_id":
                    continue
                rec[key] = value
            rec["updated_at"] = _now_ts()
            updated = True
            break
    if updated:
        write_all(root, agent_id, records)
    return updated

def delete_turn(root: str, agent_id: str, message_id: str) -> bool:
    path = _hist_path(root, agent_id)
    records = _read_records(path)
    new_records = [rec for rec in records if rec["message_id"] != message_id]
    if len(new_records) == len(records):
        return False
    write_all(root, agent_id, new_records)
    return True

def clear_history(root: str, agent_id: str):
    path = _hist_path(root, agent_id)
    open(path, "w", encoding="utf-8").close()
