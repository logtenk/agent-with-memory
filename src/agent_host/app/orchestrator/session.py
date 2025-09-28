from typing import List, Dict, Any, AsyncGenerator
from ..clients.llamacpp import stream_chat, nonstream_chat
from .tools import TOOLS, list_tools_for_prompt
from ..config import CHROMA_PERSIST_ROOT
from . import history as H

MAX_TURNS = 20  # pairs

SYS_HEADER = """You are {character}.
Impression of user: {impression}
Current mood: {mood}
Capabilities: {capabilities}
Memory summary: {memory_summary}

TOOL CALLING CONVENTION:
- To call ANY tool, emit ONE line exactly:
  TOOL_CALL: {{"name":"<toolname>","payload":{{ ... JSON matching schema ... }}}}
- After tool results arrive (as a tool message), continue your answer.
- If no tool is helpful, continue normally.

AVAILABLE TOOLS:
"""

def build_system_prompt(profile: Dict[str, Any]) -> str:
    head = SYS_HEADER.format(
        character=profile["character"],
        impression=profile["impression_of_user"],
        mood=profile["current_mood"],
        capabilities=", ".join(profile["capabilities"]),
        memory_summary=profile["memory_summary"]
    )
    lines = [head]
    for t in list_tools_for_prompt():
        lines.append(f"- {t['name']}\n  When: {t['description']}\n  Input JSON schema: {t['input_schema']}")
    return "\n".join(lines)

async def run_turn(profile: Dict[str, Any], user_text: str, allow_tools=True, stream=True) -> AsyncGenerator[Dict[str, Any], None]:
    agent_id = profile.get("agent_id", "default")
    system_prompt = build_system_prompt(profile)

    # Reload history from disk so external edits are respected
    hist = H.load_history(CHROMA_PERSIST_ROOT, agent_id, max_pairs=MAX_TURNS)

    # Append CURRENT user turn first (correct order)
    H.append_turn(CHROMA_PERSIST_ROOT, agent_id, "user", user_text)

    # Build messages for the model: system + (disk history + this user)
    messages: List[Dict[str, str]] = [{"role":"system","content":system_prompt}]
    messages.extend(hist)
    messages.append({"role":"user","content":user_text})

    used_tools: List[Dict[str, Any]] = []
    did_memory_insert = False
    assist_buffer = ""  # we’ll append assistant once at the end

    # === Primary turn ===
    if stream:
        async for tok in stream_chat(messages, cache_prompt=True):  # see §3
            t = tok if isinstance(tok, str) else tok
            # Try to split by lines to detect TOOL_CALL; keep residual
            assist_buffer += t
            while "\n" in assist_buffer:
                line, assist_buffer = assist_buffer.split("\n", 1)
                ls = line.strip()
                if ls.startswith("TOOL_CALL:"):
                    # emit nothing to user (or echo the line if you prefer)
                    # Execute tool
                    try:
                        import json
                        spec = json.loads(ls.split("TOOL_CALL:",1)[1].strip())
                        name = spec["name"]; payload = spec.get("payload", {})
                        if allow_tools and name in TOOLS:
                            result = TOOLS[name].handler(payload)
                            used_tools.append({"name":name,"payload":payload,"result":result})
                            if name == "memory.insert":
                                did_memory_insert = True
                            # Continue after tool
                            messages.append({"role":"tool","content":f"{name} -> {result}"})
                            cont = await nonstream_chat(messages, max_tokens=512, temperature=0.7, cache_prompt=True)
                            # stream continuation
                            yield {"type":"token","data":cont["text"]}
                            assist_buffer += cont["text"]
                        else:
                            # Not allowed/not found -> just show literal line (optional)
                            yield {"type":"token","data": line + "\n"}
                    except Exception:
                        yield {"type":"token","data": line + "\n"}
                else:
                    # normal text line
                    yield {"type":"token","data": line + "\n"}
        # flush remainder
        if assist_buffer:
            yield {"type":"token","data": assist_buffer}
    else:
        out = await nonstream_chat(messages, cache_prompt=True)
        assist_buffer = out["text"]
        yield {"type":"token","data": assist_buffer}

    # Append the assistant message ONCE (complete text)
    if assist_buffer:
        H.append_turn(CHROMA_PERSIST_ROOT, agent_id, "assistant", assist_buffer)

    # === Post-turn maintenance ===
    post_q = (
        "Decide if you should update:\n"
        "- agent.update_impression (impression_of_user)\n"
        "- agent.update_mood (current_mood)\n"
        "- agent.update_memory_summary (memory_summary)\n"
        "- memory.insert (new durable facts)\n"
        "Reply with zero or more TOOL_CALL lines."
    )
    post_msgs = [{"role":"system","content":system_prompt}]
    # Reload again (user+assistant now on disk) to keep absolute source of truth
    post_msgs.extend(H.load_history(CHROMA_PERSIST_ROOT, agent_id, max_pairs=MAX_TURNS))
    post_msgs.append({"role":"user","content":post_q})

    cont = await nonstream_chat(post_msgs, max_tokens=256, temperature=0.2, cache_prompt=True)
    for line in cont["text"].splitlines():
        if line.strip().startswith("TOOL_CALL:"):
            import json
            try:
                spec = json.loads(line.strip().split("TOOL_CALL:",1)[1].strip())
                name = spec["name"]; payload = spec.get("payload", {})
                if name in TOOLS:
                    result = TOOLS[name].handler(payload)
                    used_tools.append({"name":name,"payload":payload,"result":result})
                    if name == "memory.insert":
                        did_memory_insert = True
            except Exception:
                pass

    # === History policy: clear after memory.insert ===
    if did_memory_insert:
        H.clear_history(CHROMA_PERSIST_ROOT, agent_id)
    else:
        # Truncate to last MAX_TURNS pairs
        msgs = H.load_history(CHROMA_PERSIST_ROOT, agent_id, max_pairs=MAX_TURNS)
        H.write_all(CHROMA_PERSIST_ROOT, agent_id, msgs)

    yield {"type":"done", "data":{"used_tools":used_tools}}
