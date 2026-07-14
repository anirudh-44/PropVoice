from langgraph.graph import StateGraph, END
from state import AgentState
from agents import router_node, rag_agent_node

# Initialize the graph with our state schema
workflow = StateGraph(AgentState)

# Add our nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("rag_agent", rag_agent_node)

# Set the entry point
workflow.set_entry_point("router")

# Define the conditional routing logic
def route_after_analysis(state: AgentState):
    if not state["is_safe"]:
        print("--- GUARDRAIL TRIGGERED: Ending Conversation ---")
        return END
    
    if state["next_node"] == "rag_agent":
        print("--- ROUTING TO: Property Search ---")
        return "rag_agent"
        
    if state["next_node"] == "scheduling_agent":
        print("--- ROUTING TO: Booking ---")
        return END # We will build the Scheduling agent on Day 6

# Add conditional edges out of the router
workflow.add_conditional_edges("router", route_after_analysis)

# 2. The RAG Agent is terminal for now; once it answers, the conversation turn ends
workflow.add_edge("rag_agent", END)

# Compile the graph into an executable application
app = workflow.compile()