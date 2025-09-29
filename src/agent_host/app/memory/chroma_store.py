# src/agent_host/app/memory/chroma_store.py
import os, uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

ALLOWED_META_KEYS = {"type", "date", "time", "tag", "memory_id", "salience", "created_at", "last_seen_at"}

def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Accept friendly forms like:
      {"tag":"food","type":"preference"}  -> {"$and":[{"tag":"food"},{"type":"preference"}]}
      {"date":{"$gt":20250901}}          -> unchanged (single key)
      {"$or":[{"tag":"travel"},{"tag":"food"}]} -> unchanged (already logical)
    """
    if not where:
        return None
    if not isinstance(where, dict):
        return where  # let Chroma error loudly if it's totally wrong

    # Already a logical operator at top-level? (e.g. "$and", "$or")
    if any(k in where for k in ("$and", "$or")):
        return where

    # Single field: pass-through
    if len(where) == 1:
        return where

    # Multiple fields: wrap into $and
    return {"$and": [{k: v} for k, v in where.items()]}

def get_collection_for_agent(agent_id: str, persist_root: str) -> chromadb.api.models.Collection.Collection:
    path = os.path.join(persist_root, agent_id, "memory")
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(
        name="memories",
        metadata={"hnsw:space":"cosine"},
    )

def _flat_meta_only(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only allowed keys; coerce to primitives Chroma accepts
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if k not in ALLOWED_META_KEYS:
            continue
        # Only allow primitives or None
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            # Drop non-primitive values silently (no lists/dicts)
            continue
    return clean

def upsert_memories(agent_id: str, persist_root: str, items: List[Dict[str, Any]]):
    col = get_collection_for_agent(agent_id, persist_root)
    ids, docs, metas = [], [], []
    for it in items:
        _id = it.get("memory_id") or str(uuid.uuid4())
        it["memory_id"] = _id
        # enforce flat metadata schema
        meta = {k: v for k, v in it.items() if k != "text"}
        meta = _flat_meta_only(meta)
        # recommended fields (optional)
        # - type: str (e.g., "preference" | "fact" | "event" | "task")
        # - date: int (YYYYMMDD)
        # - time: int (HHMMSS)
        # - tag: str (single tag)
        ids.append(_id)
        docs.append(it["text"])
        metas.append(meta)
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return [it["memory_id"] for it in items]

def query_memories(agent_id: str, persist_root: str, query: str, k: int = 6,
                   where: Optional[Dict[str, Any]] = None):
    col = get_collection_for_agent(agent_id, persist_root)
    normalized = _normalize_where(where)
    # Pass through Chroma metadata filter
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents","metadatas","distances"],
        where=normalized or None
    )
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, mid in enumerate(ids):
        out.append({
            "memory_id": mid,
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })
    return out

def delete_memory(agent_id: str, persist_root: str, memory_id: str):
    col = get_collection_for_agent(agent_id, persist_root)
    col.delete(ids=[memory_id])

def update_memory(agent_id: str, persist_root: str, memory_id: str, patch: Dict[str, Any]):
    col = get_collection_for_agent(agent_id, persist_root)
    recs = col.get(ids=[memory_id], include=["documents","metadatas"])
    if not recs["ids"]:
        return False
    doc = recs["documents"][0]
    meta = recs["metadatas"][0] or {}
    if "text" in patch:
        doc = patch["text"]
    # merge and flatten only allowed metadata
    merged = {**meta, **{k: v for k, v in patch.items() if k != "text"}}
    meta = _flat_meta_only(merged)
    col.upsert(ids=[memory_id], documents=[doc], metadatas=[meta])
    return True
