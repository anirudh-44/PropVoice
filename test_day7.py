from langchain_core.messages import HumanMessage
from graph import app

def run_hitl_test():
    user_query = "I want to schedule a tour for The Vertex tomorrow at 2:00 PM."
    print(f"[USER]: {user_query}\n")
    
    # Every thread needs a unique thread_id when using a checkpointer
    config = {"configurable": {"thread_id": "tour_thread_1"}}
    
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "is_safe": True,
        "next_node": "",
        "booking_details": None
    }
    
    print("--- [PHASE 1]: Running graph until the HITL breakpoint ---")
    # Run the graph; it will pause before entering 'finalize_booking'
    for event in app.stream(initial_state, config):
        print(f"Node executed: {list(event.keys())}")
        
    # Inspect the current state while paused
    current_state = app.get_state(config)
    print("\n--- [PAUSED AT BREAKPOINT] ---")
    print(f"Next node to execute: {current_state.next}")
    print(f"Drafted Payload awaiting approval: {current_state.values.get('booking_details')}")
    
    # Simulate user approval/confirmation
    user_approved = input("\nDo you want to approve and finalize this booking? (y/n): ")
    
    if user_approved.lower() == 'y':
        print("\n--- [PHASE 2]: Resuming graph execution ---")
        # Passing None as input with the same config resumes the graph from the checkpoint
        for event in app.stream(None, config):
            print(f"Node resumed: {list(event.keys())}")
            
        final_state = app.get_state(config)
        print("\n--- Final Confirmed State ---")
        print(final_state.values["messages"][-1])
        print(final_state.values["booking_details"])
    else:
        print("\n--- Booking cancelled by human supervisor. ---")

if __name__ == "__main__":
    run_hitl_test()