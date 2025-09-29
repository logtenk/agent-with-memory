import time
from agent_host.app.memory import chroma_store

def _today_ymd():
    # YYYYMMDD as int
    return int(time.strftime("%Y%m%d"))

def _now_hms():
    # HHMMSS as int
    return int(time.strftime("%H%M%S"))

def test_upsert_and_basic_retrieve(tmp_path):
    root = tmp_path.as_posix()
    agent = "tester1"

    items = [
        {
            "text": "User prefers aisle seats on flights.",
            "type": "preference",
            "date": 20250910,
            "time": 93015,
            "tag": "travel",
            "salience": 0.8,
            # should be dropped (not in allowed list or non-primitive)
            "unknown": "drop-me",
            "tags": ["no", "lists"],
            "extra": {"no": "dicts"},
        },
        {
            "text": "Project X kickoff on Sept 15.",
            "type": "event",
            "date": 20250915,
            "time": 101500,
            "tag": "work",
            "salience": 0.6,
        },
        {
            "text": "User likes sushi.",
            "type": "preference",
            "date": 20240820,
            "time": 223045,
            "tag": "food",
            "salience": 0.9,
        },
    ]
    ids = chroma_store.upsert_memories(agent, root, items)
    assert len(ids) == 3
    assert len(set(ids)) == 3

    # basic semantic retrieve
    res = chroma_store.query_memories(agent, root, "aisle seats", k=5)
    assert len(res) >= 1
    assert any("aisle" in r["text"].lower() for r in res)

    # ensure metadata flattened & sanitized (no lists/dicts, unknown keys dropped)
    a = next(r for r in res if "aisle" in r["text"].lower())
    meta = a["metadata"]
    assert meta["type"] == "preference"
    assert isinstance(meta.get("date"), int)
    assert isinstance(meta.get("time"), int)
    assert meta.get("tag") == "travel"
    assert "unknown" not in meta
    assert "tags" not in meta
    assert "extra" not in meta

def test_where_filter_on_date_and_tag(tmp_path):
    root = tmp_path.as_posix()
    agent = "tester2"

    chroma_store.upsert_memories(agent, root, [
        {"text": "Old note", "type": "fact", "date": 20240101, "time": 90000, "tag": "misc"},
        {"text": "New note about ramen", "type": "preference", "date": 20250920, "time": 110000, "tag": "food"},
        {"text": "Another travel note", "type": "event", "date": 20250925, "time": 153000, "tag": "travel"},
    ])

    # filter newer than Sept 1, 2025
    newer = chroma_store.query_memories(agent, root, "note", k=10, where={"date": {"$gt": 20250901}})
    assert len(newer) >= 2
    assert all(r["metadata"]["date"] > 20250901 for r in newer)

    # filter by tag and type
    food_prefs = chroma_store.query_memories(agent, root, "ramen", k=5, where={"tag": "food", "type": "preference"})
    assert len(food_prefs) >= 1
    assert all(r["metadata"].get("tag") == "food" and r["metadata"].get("type") == "preference" for r in food_prefs)

def test_update_merge_and_retrieve(tmp_path):
    root = tmp_path.as_posix()
    agent = "tester3"

    ids = chroma_store.upsert_memories(agent, root, [
        {"text": "User likes sushi.", "type": "preference", "date": 20240820, "time": 223045, "tag": "food"},
    ])
    mid = ids[0]

    # Update: change text, change tag, add time
    ok = chroma_store.update_memory(agent, root, mid, {
        "text": "User loves ramen.",
        "tag": "ramen",
        "time": 81505,  # 08:15:05
        # Attempt to sneak a list in (should be dropped)
        "tags": ["nope"]
    })
    assert ok

    # Retrieval by content
    res = chroma_store.query_memories(agent, root, "ramen", k=5)
    assert len(res) >= 1
    got = next(r for r in res if r["memory_id"] == mid)
    assert "ramen" in got["text"].lower()
    assert got["metadata"].get("tag") == "ramen"
    assert got["metadata"].get("time") == 81505
    assert "tags" not in got["metadata"]

def test_delete_and_absence(tmp_path):
    root = tmp_path.as_posix()
    agent = "tester4"

    ids = chroma_store.upsert_memories(agent, root, [
        {"text": "Temporary project note", "type": "event", "date": 20250927, "time": 94500, "tag": "work"},
        {"text": "Keep me", "type": "fact", "date": 20250928, "time": 101500, "tag": "misc"},
    ])
    to_delete = ids[0]
    chroma_store.delete_memory(agent, root, to_delete)

    # query should never return the deleted id
    res = chroma_store.query_memories(agent, root, "project", k=10)
    assert all(r["memory_id"] != to_delete for r in res)

def test_today_helpers_and_filters(tmp_path):
    # Sanity check we can store today's date/time and filter
    root = tmp_path.as_posix()
    agent = "tester5"
    today = _today_ymd()
    now = _now_hms()

    chroma_store.upsert_memories(agent, root, [
        {"text": "Today event", "type": "event", "date": today, "time": now, "tag": "today"},
        {"text": "Yesterday event", "type": "event", "date": max(19000101, today - 1), "time": 100000, "tag": "yday"},
    ])

    # filter date >= today
    res = chroma_store.query_memories(agent, root, "event", k=10, where={"date": {"$gte": today}})
    assert len(res) >= 1
    assert all(r["metadata"]["date"] >= today for r in res)
