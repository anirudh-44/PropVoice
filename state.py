from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # The history of the conversation
    messages: List[BaseMessage]
    
    # Track if the input is safe/on-topic
    is_safe: bool
    
    # The routed destination (e.g., "rag_agent", "scheduling_agent")
    next_node: str