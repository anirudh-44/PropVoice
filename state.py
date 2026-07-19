from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    
    messages: List[BaseMessage] # The history of the conversation 
    is_safe: bool # Track if the input is safe/on-topic
    next_node: str # The routed destination (e.g., "rag_agent", "scheduling_agent")  
    booking_details: Optional[Dict[str, Any]] # New field to store structured booking data for tool calling and HITL