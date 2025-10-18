from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator
from ..clients.llamacpp import stream_chat, nonstream_chat
from .tools import TOOLS, list_tools_for_prompt
from ..config import CHROMA_PERSIST_ROOT
from . import history as H

MAX_TURNS = 20  # pairs

SYS_HEADER = """{character}.
System-managed notes: {notes}
Current date: {current_date}
Current time: {current_time}\n"""
TOOL_HEADER = """TOOL CALLING CONVENTION:
- To call ANY tool, emit ONE line exactly:
  TOOL_CALL: {{"name":"<toolname>","payload":{{ ... JSON matching schema ... }}}}
- After tool results arrive (as a tool message), continue your answer.
- If no tool is helpful, continue normally.

AVAILABLE TOOLS:
"""

def build_system_prompt(profile: Dict[str, Any]) -> str:
    now = datetime.now()
    head = SYS_HEADER.format(
        character=profile["character"],
        notes=profile["notes"],
        current_date=now.strftime("%Y-%m-%d"),
        current_time=now.strftime("%H:%M:%S"),
    ) + TOOL_HEADER
    lines = [head]
    for t in list_tools_for_prompt():
        lines.append(f"- {t['name']}\n  When: {t['description']}\n  Input JSON schema: {t['input_schema']}")
    return "\n".join(lines)

# Given a system output, handle all tool calls found within it. Returns list of tool names and tool messages.
def run_tool(assistant_output: str) -> List[tuple[str, str]]:
    import json

    outputs = []

    try:
        matches = assistant_output.split("TOOL_CALL:")[1:]
        for match in matches:
            print("current match: ", match, "\n\n")
            end = match.rfind('}')
            match = match[:end+1] if end != -1 else None
            # Attempt to parse the JSON payload
            spec = json.loads(match.strip())
            name = spec["name"]
            payload = spec.get("payload", {})
            if name in TOOLS:
                result = TOOLS[name].handler(payload)
                outputs.append((name, str(result)))
    except Exception as e:
        print("Error processing tool call:", e)

    return outputs

async def run_turn(profile: Dict[str, Any], user_text: str, allow_tools=True, stream=True) -> AsyncGenerator[Dict[str, Any], None]:
    agent_id = profile.get("agent_id", "default")
    system_prompt = build_system_prompt(profile)

    # Reload history from disk so external edits are respected
    hist_records = H.load_history(CHROMA_PERSIST_ROOT, agent_id, max_pairs=MAX_TURNS)

    # Build messages for the model: system + (disk history + this user)
    messages: List[Dict[str, str]] = [{"role":"system","content":system_prompt}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in hist_records)
    if user_text != "<none>":
        messages.append({"role":"user","content":user_text})
        H.append_turn(CHROMA_PERSIST_ROOT, agent_id, "user", user_text)

    # === Assistant response phase ===
    assist_buffer = ""
    if stream:
        async for tok in stream_chat(messages, cache_prompt=True):  # see §3
            # Try to split by lines to detect TOOL_CALL; keep residual
            assist_buffer += tok
            yield {"type":"token","data":tok}
    else:
        out = await nonstream_chat(messages, cache_prompt=True)
        assist_buffer = out["text"]
        yield {"type":"token","data":assist_buffer}

    H.append_turn(CHROMA_PERSIST_ROOT, agent_id, "assistant", assist_buffer)
    messages.append({"role":"assistant","content":assist_buffer})

    # === Tool handling phase ===
    if allow_tools:
        tool_outputs = run_tool(assist_buffer)
        if tool_outputs:
            for name, result in tool_outputs:
                H.append_turn(CHROMA_PERSIST_ROOT, agent_id, "tool", f"{name} -> {result}")
                messages.append({"role":"tool", "content":f"{name} -> {result}"})
                yield {"type":"tool","data": {"Tool name": name, "Tool result": result}}
            # After tool calls, we could have the assistant continue
            followup_out = await nonstream_chat(
                messages,
                cache_prompt=True
            )
            followup_text = followup_out["text"]
            H.append_turn(CHROMA_PERSIST_ROOT, agent_id, "assistant", followup_text)
            messages.append({"role":"assistant","content":followup_text})
            yield {"type":"token","data":followup_text}

    # === Post-turn maintenance ===
    # post_q = (
    #     "Decide if you should update:\n"
    #     "- agent.update_notes"
    #     "Reply with zero or more TOOL_CALL lines."
    # )
    # post_msgs = [{"role":"system","content":system_prompt}]
    # # Reload again (user+assistant now on disk) to keep absolute source of truth
    # post_hist = H.load_history(CHROMA_PERSIST_ROOT, agent_id, max_pairs=MAX_TURNS)
    # post_msgs.extend({"role": m["role"], "content": m["content"]} for m in post_hist)
    # post_msgs.append({"role":"user","content":post_q})

    # cont = await nonstream_chat(post_msgs, max_tokens=256, temperature=0.2, cache_prompt=True)
    # for line in cont["text"].splitlines():
    #     if line.strip().startswith("TOOL_CALL:"):
    #         import json
    #         try:
    #             spec = json.loads(line.strip().split("TOOL_CALL:",1)[1].strip())
    #             name = spec["name"]; payload = spec.get("payload", {})
    #             if name in TOOLS:
    #                 result = TOOLS[name].handler(payload)
    #         except Exception:
    #             pass

    yield {"type":"done","data":{}}

    
