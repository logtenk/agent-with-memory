import json


ls = """TOOL_CALL: {"name":"duckduckgo.search","payload":{"query":"movies","k":5}}"""
spec = json.loads(ls.split("TOOL_CALL:",1)[1].strip())
name = spec["name"]; payload = spec.get("payload", {})
print(name, payload)