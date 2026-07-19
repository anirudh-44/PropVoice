from langchain_core.messages import HumanMessage
from graph import app

def test_workflow(user_query: str):
    print(f"\n==========================================")
    print(f"[USER]: {user_query}")
    print(f"==========================================")
    
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "is_safe": True,
        "next_node": "",
        "booking_details": None
    }
    
    final_state = app.invoke(initial_state)
    
    print("\n--- Final Concierge Response ---")
    print(final_state["messages"][-1].content)
    
    if final_state.get("booking_details"):
        print("\n--- State Booking Payload (Ready for HITL) ---")
        print(final_state["booking_details"])

if __name__ == "__main__":
    # Test 1: Check availability tool
    test_workflow("What times are available to tour The Vertex?")
    
    # Test 2: Draft booking tool
    test_workflow("I want to schedule a tour for Oakwood Townhomes tomorrow at 3:00 PM.")