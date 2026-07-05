from langgraph.graph import StateGraph, END
from state import AgentState
from agents import router_node

# Initialize the graph with our state schema
workflow = StateGraph(AgentState)

# Add our node to the graph
workflow.add_node("router", router_node)

# Set the entry point
workflow.set_entry_point("router")

# Define the conditional routing logic
def route_after_analysis(state: AgentState):
    if not state["is_safe"]:
        print("--- GUARDRAIL TRIGGERED: Ending Conversation ---")
        return END
    
    if state["next_node"] == "rag_agent":
        print("--- ROUTING TO: Property Search ---")
        return END # We will build the RAG agent tomorrow and change this
        
    if state["next_node"] == "scheduling_agent":
        print("--- ROUTING TO: Booking ---")
        return END # We will build the Scheduling agent on Day 6

# Add conditional edges out of the router
workflow.add_conditional_edges(
    "router",
    route_after_analysis
)

# Compile the graph into an executable application
app = workflow.compile()