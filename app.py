import streamlit as st
from langchain_core.messages import HumanMessage
from graph import app as agent_graph

st.set_page_config(page_title="PropVoice Concierge", layout="wide")

# Initialize session state for the LangGraph thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session_1"
if "messages" not in st.session_state:
    st.session_state.messages = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("PropVoice Agentic Concierge")

# Sidebar for Admin / HITL Approvals
with st.sidebar:
    st.header("Admin Dashboard")
    st.subheader("Pending Approvals")
    
    # Check current graph state
    current_state = agent_graph.get_state(config)
    if current_state.next and "finalize_booking" in current_state.next:
        pending_payload = current_state.values.get("booking_details")
        st.warning(f"Pending Tour: {pending_payload['property_name']}")
        st.write(f"Date: {pending_payload['date']} | Time: {pending_payload['time']}")
        
        if st.button("Approve Tour", type="primary"):
            with st.spinner("Writing to database..."):
                # Resume the graph from the breakpoint
                for _ in agent_graph.stream(None, config):
                    pass
                final_state = agent_graph.get_state(config)
                st.success("Tour Confirmed & Saved!")
                st.session_state.messages.append({"role": "assistant", "content": final_state.values["messages"][-1]})
                st.rerun()
    else:
        st.info("No pending tours require approval.")

# Main Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about apartments or schedule a tour..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    with st.spinner("Agent is thinking..."):
        # Run graph until it hits an end or a breakpoint
        for event in agent_graph.stream(initial_state, config):
            pass
            
        new_state = agent_graph.get_state(config)
        
        # If the graph paused for approval, notify the user
        if new_state.next and "finalize_booking" in new_state.next:
            response = "I have drafted your tour. Please wait while a supervisor approves it."
        else:
            # Otherwise, print the final AI response
            response = new_state.values["messages"][-1]
            if hasattr(response, 'content'):
                response = response.content
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        st.rerun()