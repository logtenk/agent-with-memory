from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    agent_id: str = "default"
    user: str
    stream: bool = True
    tools: Optional[List[str]] = None  # which tools allowed this turn
    tool_calls_allowed: bool = True

class ChatChunk(BaseModel):
    token: str

class ChatResponse(BaseModel):
    text: str
    used_tools: List[Dict[str, Any]] = Field(default_factory=list)

class ChatMessage(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
    updated_at: datetime

class ChatMessagePatch(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class MemoryItem(BaseModel):
    memory_id: Optional[str] = None
    type: str
    text: str
    tags: List[str] = Field(default_factory=list)
    salience: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    last_seen_at: Optional[str] = None

class RetrieveQuery(BaseModel):
    agent_id: str = "default"
    query: str
    k: int = 6
    where: Optional[Dict[str, Any]] = None

class AgentProfile(BaseModel):
    agent_id: str
    character: str
    impression_of_user: str
    current_mood: str
    capabilities: List[str]
    memory_path: str
    memory_summary: str
    tool_instructions: str
