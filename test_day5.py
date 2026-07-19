from langchain_core.messages import HumanMessage
from graph import app

def run_rag_test():
    user_input = "Do you have any studio lofts with exposed brick walls?"
    print(f"[USER]: {user_input}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "is_safe": True,
        "next_node": ""
    }
    
    # Execute the compiled graph workflow
    final_state = app.invoke(initial_state)
    
    print("\n--- Final Concierge Response ---")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    run_rag_test()