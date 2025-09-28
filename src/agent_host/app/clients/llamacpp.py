# app/clients/llamacpp.py

import httpx
from typing import AsyncGenerator, Dict, Any, Optional
from ..config import LLAMACPP_BASE_URL

async def stream_chat(
    messages,
    temperature=0.7,
    max_tokens=1024,
    cache_prompt: bool = True,
    cache_key: Optional[str] = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    url = f"{LLAMACPP_BASE_URL}/v1/chat/completions"
    payload = {
        "model": "local-llama",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "cache_prompt": cache_prompt,
    }
    if cache_key is not None:
        # if your server build supports a cache/session key, pass it through
        payload["id"] = cache_key  # harmless if ignored
    payload.update(kwargs)
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = httpx.Response(200, content=data).json()
                    except Exception:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    tok = delta.get("content")
                    if tok:
                        yield tok

async def nonstream_chat(
    messages,
    temperature=0.7,
    max_tokens=1024,
    cache_prompt: bool = True,
    cache_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    url = f"{LLAMACPP_BASE_URL}/v1/chat/completions"
    payload = {
        "model": "local-llama",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": cache_prompt,
    }
    if cache_key is not None:
        payload["id"] = cache_key
    payload.update(kwargs)
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "raw": data
        }
