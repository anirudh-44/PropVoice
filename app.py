import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from graph import app as agent_graph
import os

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
    
    # Add a manual refresh button so the admin can pull new calls without typing
    st.button("🔄 Refresh Pending Tours")
    st.subheader("Pending Approvals")
    
    # 1. Gather all known thread IDs (Streamlit's own + Twilio's)
    threads_to_check = [st.session_state.thread_id]
    if os.path.exists("active_threads.txt"):
        with open("active_threads.txt", "r") as f:
            for line in f.read().splitlines():
                if line and line not in threads_to_check:
                    threads_to_check.append(line)
                    
    pending_count = 0
    
    # 2. Loop through every thread to check for pending approvals
    for tid in threads_to_check:
        cfg = {"configurable": {"thread_id": tid}}
        current_state = agent_graph.get_state(cfg)
        
        # If this specific thread is paused for database approval
        if current_state.next and "finalize_booking" in current_state.next:
            pending_count += 1
            #pending_payload = current_state.values.get("booking_details", {})
            # FIX 1: Add 'or {}' to guarantee it is a dictionary, even if the state holds 'None'
            pending_payload = current_state.values.get("booking_details") or {}
            
            # Use a container to visually separate multiple pending requests
            with st.container(border=True):
                st.warning(f"🏡 {pending_payload.get('property_name', 'Unknown')}")
                st.write(f"**Date:** {pending_payload.get('date')} | **Time:** {pending_payload.get('time')}")
                st.caption(f"Call/Thread ID: {tid}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Unique key prevents Duplicate Widget Error
                    if st.button("Approve", key=f"approve_{tid}", type="primary", use_container_width=True):
                        with st.spinner("Writing to database..."):
                            for _ in agent_graph.stream(None, cfg):
                                pass
                            final_state = agent_graph.get_state(cfg)
                            st.success("Tour Confirmed!")
                            
                            # Only update the local chat UI if it's the Streamlit text thread
                            if tid == st.session_state.thread_id:
                                st.session_state.messages.append({"role": "assistant", "content": final_state.values["messages"][-1]})
                            st.rerun()
                            
                with col2:
                    if st.button("Reject", key=f"reject_{tid}", type="secondary", use_container_width=True):
                        rejection_text = "Your request was declined by an administrator. Please select another time or property."
                        agent_graph.update_state(
                            cfg,
                            {"messages": [AIMessage(content=rejection_text)], "booking_details": None},
                            as_node="finalize_booking"
                        )
                        st.error("Declined.")
                        if tid == st.session_state.thread_id:
                            st.session_state.messages.append({"role": "assistant", "content": rejection_text})
                        st.rerun()
                        
    if pending_count == 0:
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