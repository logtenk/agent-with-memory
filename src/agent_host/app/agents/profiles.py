import os, json
from typing import Optional
from ..models import AgentProfile

def _profile_path(root: str, agent_id: str) -> str:
    return os.path.join(root, agent_id, "profile.json")

def read_profile(root: str, agent_id: str) -> AgentProfile:
    path = _profile_path(root, agent_id)
    with open(path, "r", encoding="utf-8") as f:
        return AgentProfile(**json.load(f))

def write_profile(root: str, profile: AgentProfile):
    base = os.path.dirname(_profile_path(root, profile.agent_id))
    os.makedirs(base, exist_ok=True)
    with open(_profile_path(root, profile.agent_id), "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, ensure_ascii=False, indent=2)
