import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from state import AgentState
from langchain_core.messages import AIMessage
from db_utils import search_properties_vector
from tools import check_availability, draft_booking

load_dotenv()

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.1,
    max_new_tokens=400,
    task='conversational'
)

model = ChatHuggingFace(llm=llm)

# Define the exact JSON structure we want the LLM to return
class RouterOutput(BaseModel):
    is_safe: bool = Field(description="True if the user is asking about real estate, apartments, or booking tours. False if toxic, political, or totally off-topic.")
    next_node: str = Field(description="If safe, output either 'rag_agent' for general questions, or 'scheduling_agent' to book a tour. If not safe, output 'end'.")

router_parser = JsonOutputParser(pydantic_object=RouterOutput)

router_prompt = PromptTemplate(
    template="""You are the routing manager for a luxury apartment concierge. 
    Analyze the user's input and determine if it is safe/relevant, and where it should be routed.
    
    User Input: {input}
    
    {format_instructions}
    """,
    input_variables=["input"],
    partial_variables={"format_instructions": router_parser.get_format_instructions()},
)

# Chain the prompt, model, and parser together
router_chain = router_prompt | model | router_parser

def router_node(state: AgentState):
    print("--- ROUTER AGENT: Analyzing Input ---")
    
    # Get the latest message from the user
    latest_message = state["messages"][-1].content
    
    # Invoke the chain to get our structured JSON
    result = router_chain.invoke({"input": latest_message})
    
    print(f"Router Decision: {result}")
    
    # Update the state with the agent's decisions
    return {
        "is_safe": result["is_safe"],
        "next_node": result["next_node"]
    }

#---------------------------RAG AGENT---------------------------------------------------------------------
# The prompt template that provides the retrieved context to the LLM
rag_prompt = PromptTemplate(
    template="""You are a helpful, enthusiastic leasing concierge for a luxury property tech company.
    Answer the user's question using ONLY the verified property context provided below. 
    If the context doesn't contain the answer, politely tell the user you couldn't find a matching property.
    
    Verified Property Context:
    {context}
    
    User Question: {question}
    
    Answer:""",
    input_variables=["context", "question"]
)

rag_chain = rag_prompt | model

def rag_agent_node(state: AgentState):
    print("--- RAG AGENT: Fetching matching properties ---")
    
    # Get the user's last message text
    latest_message = state["messages"][-1].content
    
    # 1. Run our PGVector similarity search tool
    retrieved_context = search_properties_vector(latest_message, limit=2)
    
    # 2. Let the LLM synthesize the answer using the database results
    response = rag_chain.invoke({
        "context": retrieved_context,
        "question": latest_message
    })
    
    # 3. Append the agent's response to the message history state
    return {
        "messages": [response]
    }

#------------------------------------Scheduling Agent------------------------------------------

# Define the structured schema for scheduling extraction
class SchedulingIntent(BaseModel):
    action: str = Field(description="Either 'check_availability' if the user wants to see times, or 'book_tour' if they provided a specific time/date.")
    property_name: str = Field(description="The name of the property (e.g., 'The Vertex', 'Echo Lofts'). Use 'Unknown' if not specified.")
    date: str = Field(default="Tomorrow", description="The date requested for the tour.")
    time: str = Field(default="2:00 PM", description="The time requested for the tour.")

scheduling_parser = JsonOutputParser(pydantic_object=SchedulingIntent)

scheduling_prompt = PromptTemplate(
    template="""You are a leasing coordinator assistant for luxury apartments.
    Extract the scheduling details from the user's request.
    
    User Input: {input}
    
    {format_instructions}
    """,
    input_variables=["input"],
    partial_variables={"format_instructions": scheduling_parser.get_format_instructions()},
)

scheduling_chain = scheduling_prompt | llm | scheduling_parser

def scheduling_agent_node(state: AgentState):
    print("--- SCHEDULING AGENT: Processing Appointment Request ---")
    latest_message = state["messages"][-1].content
    
    # 1. Extract structured parameters using the LLM
    intent = scheduling_chain.invoke({"input": latest_message})
    print(f"Extracted Scheduling Intent: {intent}")
    
    # 2. Execute the appropriate tool based on extracted action
    if intent["action"] == "check_availability" or intent["property_name"] == "Unknown":
        tool_result = check_availability(intent["property_name"])
        response_text = f"{tool_result} Would you like me to draft a booking for one of these times?"
        booking_payload = None
    else:
        # Execute the draft_booking tool
        booking_payload = draft_booking(
            property_name=intent["property_name"],
            date=intent.get("date", "Tomorrow"),
            time=intent.get("time", "2:00 PM")
        )
        response_text = f"I have drafted your tour for {intent['property_name']} on {intent.get('date', 'Tomorrow')} at {intent.get('time', '2:00 PM')}. Please confirm if you would like me to finalize this appointment."
        
    return {
        "messages": [response_text],
        "booking_details": booking_payload
    }