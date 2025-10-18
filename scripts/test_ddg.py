import httpx, json
from typing import Dict, Any, Callable, Optional, List
DDG_BASE = "http://127.0.0.1:50002"

def _ddg_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = payload["query"]
    k = int(payload.get("k", 5))
    with httpx.Client(timeout=20) as c:
        r = c.get(f"{DDG_BASE}/search", params={"q": q, "k": k})
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    query = "yang hansen"
    payload = {"query": query, "k": 5}
    result = _ddg_search(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))