import os, sys, json, requests
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass  # Py<3.7 or environments that don't support reconfigure

# If your shell isn't UTF-8, this helps on Windows; harmless elsewhere
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
LLAMA_BASE = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8001")  # change if needed

def stream_chat(messages, model="local-llama", temperature=0.7, max_tokens=512):
    url = f"{LLAMA_BASE}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "cache_prompt": True
    }
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            tok = delta.get("content")
            if tok:
                yield tok

def nonstream_chat(messages, model="local-llama", temperature=0.7, max_tokens=512):
    url = f"{LLAMA_BASE}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    r.encoding = "utf-8"
    data = r.json()
    return data["choices"][0]["message"]["content"]

def interactive():
    print(f"[raw llama.cpp] base={LLAMA_BASE}")
    print("Type 'exit' to quit. Add '!' prefix for non-stream.\n")
    messages = []
    # simple system prompt so we know it's working
    messages.append({"role":"system","content":"You are a helpful, concise assistant."})
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user:
            continue
        if user.lower() in {"exit","quit"}:
            break
        messages.append({"role":"user","content":user})
        if user.startswith("!"):
            # non-stream test
            messages[-1]["content"] = user[1:]
            reply = nonstream_chat(messages)
            print("Assistant:", reply, "\n")
            messages.append({"role":"assistant","content":reply})
        else:
            print("Assistant:", end=" ", flush=True)
            buf = []
            for tok in stream_chat(messages):
                sys.stdout.write(tok)
                sys.stdout.flush()
                buf.append(tok)
            print("\n")
            reply = "".join(buf)
            messages.append({"role":"assistant","content":reply})

if __name__ == "__main__":
    interactive()
