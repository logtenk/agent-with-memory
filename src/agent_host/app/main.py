from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator
import json

from agent_host.app.config import HOST, PORT, CHROMA_PERSIST_ROOT
from agent_host.app.agents import profiles
from agent_host.app.orchestrator.session import run_turn
from agent_host.app.models import ChatRequest, MemoryItem, RetrieveQuery, AgentProfile
from agent_host.app.memory import chroma_store
from agent_host.app.orchestrator.tools import list_tools_for_prompt

app = FastAPI(title="Local LLM Host")

@app.on_event("startup")
async def startup():
    print("Starting up...")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/tools")
async def get_tools():
    # Helpful for inspecting what the LLM sees
    return {"tools": list_tools_for_prompt()}

@app.post("/chat")
async def chat(req: ChatRequest):
    """SSE endpoint: streams assistant tokens and finishes with a 'done' event."""
    prof = profiles.read_profile(CHROMA_PERSIST_ROOT, req.agent_id)

    async def event_gen() -> AsyncGenerator[dict, None]:
        async for ev in run_turn(
            prof.model_dump(),
            req.user,
            allow_tools=req.tool_calls_allowed,
            stream=req.stream
        ):
            if ev["type"] == "token":
                # one token (or line chunk) at a time
                yield {"event": "chunk", "data": ev["data"]}
            else:
                # final metadata (used_tools, etc.)
                yield {"event": "done", "data": json.dumps(ev["data"])}
    return EventSourceResponse(event_gen())

# ------- Optional: REST wrappers around memory tools --------
# These are convenience endpoints for non-LLM callers (scripts, admin panels).
# The LLM itself calls memory via the TOOL_CALL convention, not these routes.

@app.post("/tools/memory/insert")
async def t_mem_insert(agent_id: str, items: list[MemoryItem]):
    ids = chroma_store.upsert_memories(agent_id, CHROMA_PERSIST_ROOT, [i.model_dump() for i in items])
    return {"ok": True, "ids": ids}

@app.post("/tools/memory/retrieve")
async def t_mem_retrieve(q: RetrieveQuery):
    res = chroma_store.query_memories(q.agent_id, CHROMA_PERSIST_ROOT, q.query, q.k)
    return {"ok": True, "results": res}

@app.post("/tools/memory/update")
async def t_mem_update(agent_id: str, memory_id: str, patch: dict):
    ok = chroma_store.update_memory(agent_id, CHROMA_PERSIST_ROOT, memory_id, patch)
    return {"ok": ok}

@app.post("/tools/memory/delete")
async def t_mem_delete(agent_id: str, memory_id: str):
    chroma_store.delete_memory(agent_id, CHROMA_PERSIST_ROOT, memory_id)
    return {"ok": True}

# ------- Profiles (handy for editing agent settings from scripts) --------

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    return profiles.read_profile(CHROMA_PERSIST_ROOT, agent_id).model_dump()

@app.post("/agents/{agent_id}")
async def set_agent(agent_id: str, profile: AgentProfile):
    profiles.write_profile(CHROMA_PERSIST_ROOT, profile)
    return {"ok": True}
