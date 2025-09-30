from fastapi import FastAPI, HTTPException, Response, status
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator
import json

from agent_host.app.config import HOST, PORT, CHROMA_PERSIST_ROOT
from agent_host.app.agents import profiles
from agent_host.app.orchestrator.session import run_turn
from agent_host.app.models import (
    AgentProfile,
    AgentProfileCreate,
    AgentProfilePatch,
    ChatMessage,
    ChatMessagePatch,
    ChatRequest,
    MemoryItem,
    RetrieveQuery,
)
from agent_host.app.memory import chroma_store
from agent_host.app.orchestrator.tools import list_tools_for_prompt
from agent_host.app.orchestrator import history as history_store

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
    try:
        prof = profiles.read_profile(CHROMA_PERSIST_ROOT, req.agent_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

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

@app.get("/agents/{agent_id}/history")
async def get_history(agent_id: str):
    records = history_store.load_all_turns(CHROMA_PERSIST_ROOT, agent_id)
    return {
        "ok": True,
        "history": [ChatMessage(**rec).model_dump() for rec in records],
    }

@app.put("/agents/{agent_id}/history/{message_id}")
async def update_history(agent_id: str, message_id: str, patch: ChatMessagePatch):
    data = patch.model_dump(exclude_unset=True, exclude_none=True)
    ok = history_store.update_turn(CHROMA_PERSIST_ROOT, agent_id, message_id, data)
    return {"ok": ok}

@app.delete("/agents/{agent_id}/history/{message_id}")
async def delete_history(agent_id: str, message_id: str):
    ok = history_store.delete_turn(CHROMA_PERSIST_ROOT, agent_id, message_id)
    return {"ok": ok}

# ------- Profiles (handy for editing agent settings from scripts) --------

@app.get("/agents")
async def list_agents():
    agents = [p.model_dump() for p in profiles.list_profiles(CHROMA_PERSIST_ROOT)]
    return {"agents": agents}


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    try:
        return profiles.read_profile(CHROMA_PERSIST_ROOT, agent_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(profile: AgentProfileCreate):
    try:
        created = profiles.create_profile(
            CHROMA_PERSIST_ROOT, AgentProfile(**profile.model_dump())
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {"ok": True, "profile": created.model_dump()}


@app.patch("/agents/{agent_id}")
async def patch_agent(agent_id: str, patch: AgentProfilePatch):
    patch_data = patch.model_dump(exclude_unset=True, exclude_none=True)
    try:
        updated = profiles.update_profile(CHROMA_PERSIST_ROOT, agent_id, patch_data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"ok": True, "profile": updated.model_dump()}


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    try:
        profiles.delete_profile(CHROMA_PERSIST_ROOT, agent_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
