import os
import json
from typing import List

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


def list_profiles(root: str) -> List[AgentProfile]:
    if not os.path.isdir(root):
        return []

    profiles_data: List[AgentProfile] = []
    for agent_id in sorted(os.listdir(root)):
        path = _profile_path(root, agent_id)
        if os.path.isfile(path):
            profiles_data.append(read_profile(root, agent_id))
    return profiles_data


def create_profile(root: str, profile: AgentProfile) -> AgentProfile:
    path = _profile_path(root, profile.agent_id)
    if os.path.exists(path):
        raise FileExistsError(f"Profile already exists for agent '{profile.agent_id}'")
    write_profile(root, profile)
    return profile


def update_profile(root: str, agent_id: str, patch: dict) -> AgentProfile:
    current = read_profile(root, agent_id)
    if "agent_id" in patch and patch["agent_id"] != agent_id:
        raise ValueError("agent_id cannot be modified")

    data = current.model_dump()
    data.update(patch)
    updated = AgentProfile(**data)
    write_profile(root, updated)
    return updated


def delete_profile(root: str, agent_id: str) -> None:
    path = _profile_path(root, agent_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile for agent '{agent_id}' does not exist")

    os.remove(path)

    agent_dir = os.path.dirname(path)
    try:
        os.rmdir(agent_dir)
    except OSError:
        # Directory not empty or removal failed; ignore silently.
        pass
