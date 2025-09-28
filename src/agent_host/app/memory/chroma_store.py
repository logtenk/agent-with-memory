import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os, uuid

def get_collection_for_agent(agent_id: str, persist_root: str) -> chromadb.api.models.Collection.Collection:
    path = os.path.join(persist_root, agent_id, "memory")
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(
        name="memories",
        metadata={"hnsw:space":"cosine"} # default
    )

def upsert_memories(agent_id: str, persist_root: str, items: List[Dict[str, Any]]):
    col = get_collection_for_agent(agent_id, persist_root)
    ids, docs, metas = [], [], []
    for it in items:
        _id = it.get("memory_id") or str(uuid.uuid4())
        it["memory_id"] = _id
        ids.append(_id)
        docs.append(it["text"])
        meta = {k:v for k,v in it.items() if k not in ("text",)}
        metas.append(meta)
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return [it["memory_id"] for it in items]

def query_memories(agent_id: str, persist_root: str, query: str, k: int = 6):
    col = get_collection_for_agent(agent_id, persist_root)
    res = col.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])
    out = []
    for i, mid in enumerate(res.get("ids", [[]])[0]):
        out.append({
            "memory_id": mid,
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return out

def delete_memory(agent_id: str, persist_root: str, memory_id: str):
    col = get_collection_for_agent(agent_id, persist_root)
    col.delete(ids=[memory_id])

def update_memory(agent_id: str, persist_root: str, memory_id: str, patch: Dict[str, Any]):
    col = get_collection_for_agent(agent_id, persist_root)
    # Chroma has no partial updates; re-upsert with same id
    recs = col.get(ids=[memory_id], include=["documents","metadatas"])
    if not recs["ids"]:
        return False
    doc = recs["documents"][0]
    meta = recs["metadatas"][0]
    if "text" in patch:
        doc = patch["text"]
    meta.update({k:v for k,v in patch.items() if k!="text"})
    col.upsert(ids=[memory_id], documents=[doc], metadatas=[meta])
    return True
