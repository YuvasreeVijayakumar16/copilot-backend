from pydantic import BaseModel
from typing import List, Optional


from uuid import uuid4
from pydantic import Field

class AgentConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    role: str
    purpose: str
    instructions: List[str]
    capabilities: List[str]
    welcome_message: str
    tone: str
    knowledge_base: List[str]
    sample_prompts: List[str]
    schedule_enabled: bool = False
    frequency: Optional[str] = None
    time: Optional[str] = None
    output_method: Optional[str] = None
    published: bool = False
    is_active: bool = True 

