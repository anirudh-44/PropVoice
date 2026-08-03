from langgraph.graph import StateGraph, END
from state import AgentState
from agents import router_node, rag_agent_node, scheduling_agent_node, finalize_booking_node
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Initialize the graph with our state schema
workflow = StateGraph(AgentState)

#1. Add our nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("scheduling_agent", scheduling_agent_node)
workflow.add_node("finalize_booking", finalize_booking_node)

# Set the entry point
workflow.set_entry_point("router")

# Define the conditional routing logic
def route_after_analysis(state: AgentState):
    if not state["is_safe"]:
        print("--- GUARDRAIL TRIGGERED: Ending Conversation ---")
        return END
    
    if state["next_node"] == "rag_agent":
        print("--- ROUTING TO: Property Search ---")
        return "rag_agent" # return the node (not the function)
        
    if state["next_node"] == "scheduling_agent":
        print("--- ROUTING TO: Booking ---")
        return "scheduling_agent" # return the node (not the function)

# Add conditional edges out of the router
workflow.add_conditional_edges("router", route_after_analysis)

# 2. The RAG Agent is terminal for now; once it answers, the conversation turn ends
workflow.add_edge("rag_agent", END)
# 3. Once the scheduling agent drafts a booking, route it directly to the finalizer node
workflow.add_edge("scheduling_agent", "finalize_booking")
workflow.add_edge("finalize_booking", END)

# 3. Create a persistent SQLite checkpointer connected to a shared file
# We use check_same_thread=False because FastAPI (uvicorn) and Streamlit 
# will access this file from different processes/threads.
conn = sqlite3.connect("shared_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile the graph into an executable application
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["finalize_booking"] # <-- Pauses execution right here!
)