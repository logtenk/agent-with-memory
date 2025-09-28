from typing import Dict, Any, Callable, Optional, List
import httpx, json

# ===== Unified Tool Registry =====

class ToolSpec:
    def __init__(self, name: str, description: str, schema: Dict[str, Any],
                 handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler

TOOLS: Dict[str, ToolSpec] = {}

def register(tool: ToolSpec):
    TOOLS[tool.name] = tool
    return tool

def list_tools_for_prompt() -> List[Dict[str, Any]]:
    """Return minimal metadata to show the LLM what exists and how to call it."""
    out = []
    for t in TOOLS.values():
        out.append({
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema
        })
    return out

# ===== Internal: Memory (Chroma) =====

from ..config import CHROMA_PERSIST_ROOT
from ..memory import chroma_store

def _mem_insert(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    items = payload["items"]
    ids = chroma_store.upsert_memories(agent_id, CHROMA_PERSIST_ROOT, items)
    return {"ok": True, "ids": ids}

def _mem_retrieve(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    query = payload["query"]
    k = int(payload.get("k", 6))
    res = chroma_store.query_memories(agent_id, CHROMA_PERSIST_ROOT, query, k)
    return {"ok": True, "results": res}

def _mem_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    mid = payload["memory_id"]
    patch = payload.get("patch", {})
    ok = chroma_store.update_memory(agent_id, CHROMA_PERSIST_ROOT, mid, patch)
    return {"ok": ok}

def _mem_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    mid = payload["memory_id"]
    chroma_store.delete_memory(agent_id, CHROMA_PERSIST_ROOT, mid)
    return {"ok": True}

register(ToolSpec(
    name="memory.insert",
    description="Insert durable memory items (facts, preferences, events) into the agent's memory store.",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "items":{"type":"array","items":{"type":"object"}}
        },
        "required":["items"]
    },
    handler=_mem_insert
))
register(ToolSpec(
    name="memory.retrieve",
    description="Retrieve top-K relevant memories for a query.",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "query":{"type":"string"},
            "k":{"type":"integer","default":6}
        },
        "required":["query"]
    },
    handler=_mem_retrieve
))
register(ToolSpec(
    name="memory.update",
    description="Update a memory item by ID (partial upsert).",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "memory_id":{"type":"string"},
            "patch":{"type":"object"}
        },
        "required":["memory_id","patch"]
    },
    handler=_mem_update
))
register(ToolSpec(
    name="memory.delete",
    description="Delete a memory item by ID.",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "memory_id":{"type":"string"}
        },
        "required":["memory_id"]
    },
    handler=_mem_delete
))

# ===== Internal: Agent profile updates =====

from ..agents import profiles as agent_profiles

def _agent_update_impression(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    prof = agent_profiles.read_profile(CHROMA_PERSIST_ROOT, agent_id)
    if "impression_of_user" in payload:
        prof.impression_of_user = payload["impression_of_user"]
    agent_profiles.write_profile(CHROMA_PERSIST_ROOT, prof)
    return {"ok": True}

def _agent_update_mood(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    mood = payload["current_mood"]
    prof = agent_profiles.read_profile(CHROMA_PERSIST_ROOT, agent_id)
    prof.current_mood = mood
    agent_profiles.write_profile(CHROMA_PERSIST_ROOT, prof)
    return {"ok": True}

def _agent_update_mem_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = payload.get("agent_id", "default")
    summary = payload["memory_summary"]
    prof = agent_profiles.read_profile(CHROMA_PERSIST_ROOT, agent_id)
    prof.memory_summary = summary
    agent_profiles.write_profile(CHROMA_PERSIST_ROOT, prof)
    return {"ok": True}

register(ToolSpec(
    name="agent.update_impression",
    description="Update the agent's impression of the user.",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "impression_of_user":{"type":"string"}
        },
        "required":["impression_of_user"]
    },
    handler=_agent_update_impression
))
register(ToolSpec(
    name="agent.update_mood",
    description="Update the agent's current mood.",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "current_mood":{"type":"string"}
        },
        "required":["current_mood"]
    },
    handler=_agent_update_mood
))
register(ToolSpec(
    name="agent.update_memory_summary",
    description="Replace the agent's memory summary text.",
    schema={
        "type":"object",
        "properties":{
            "agent_id":{"type":"string","default":"default"},
            "memory_summary":{"type":"string"}
        },
        "required":["memory_summary"]
    },
    handler=_agent_update_mem_summary
))

# ===== External: DuckDuckGo MCP adapters =====
# Assumes your mcp/duckduckgo container exposes an HTTP API on localhost:7801
# (adjust host/port/routes to your MCP server)

DDG_BASE = "http://mcp_duckduckgo:7801"


def _ddg_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = payload["query"]
    k = int(payload.get("k", 5))
    with httpx.Client(timeout=20) as c:
        r = c.get(f"{DDG_BASE}/search", params={"q": q, "k": k})
        r.raise_for_status()
        return r.json()

def _ddg_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = payload["url"]
    with httpx.Client(timeout=20) as c:
        r = c.get(f"{DDG_BASE}/fetch", params={"url": url})
        r.raise_for_status()
        return r.json()

register(ToolSpec(
    name="duckduckgo.search",
    description="Web search via the DuckDuckGo MCP server. Returns a ranked list of results.",
    schema={
        "type":"object",
        "properties":{
            "query":{"type":"string"},
            "k":{"type":"integer","default":5}
        },
        "required":["query"]
    },
    handler=_ddg_search
))

register(ToolSpec(
    name="duckduckgo.fetch_content",
    description="Fetch and return cleaned main content from a given URL via MCP DuckDuckGo.",
    schema={
        "type":"object",
        "properties":{"url":{"type":"string"}},
        "required":["url"]
    },
    handler=_ddg_fetch
))
