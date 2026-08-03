import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from langchain_core.messages import HumanMessage
from graph import app as agent_graph
from typing import Optional

app = FastAPI(title="PropVoice API")

# A dedicated thread for voice calls
VOICE_CONFIG = {"configurable": {"thread_id": "twilio_voice_thread"}}

@app.get("/")
async def root():
    return {"message": "PropVoice Backend is Running", "status": "Healthy"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Returns an empty 204 No Content response to satisfy the browser
    return Response(status_code=204)

@app.post("/twillio/voice")
async def twilio_voice_webhook(request: Request):
    """Handles incoming calls and Speech-to-Text from Twilio"""
    form_data = await request.form()

    # 1. Capture the unique CallSid and update the existing VOICE_CONFIG
    call_sid = form_data.get("CallSid", "twilio_voice_thread")

    # Save the CallSid to tracker file so Streamlit can find it
    with open("active_threads.txt", "a+") as f:
        f.seek(0)
        if call_sid not in f.read():
            f.write(call_sid + "\n")
            
    VOICE_CONFIG["configurable"]["thread_id"] = call_sid 
    
    # 'SpeechResult' contains the transcribed text from the caller
    user_speech = form_data.get("SpeechResult")
    response = VoiceResponse()

    ACTION_URL = "https://denial-clergyman-goatskin.ngrok-free.dev/twillio/voice"
    
    if user_speech:
        print(f"--- INCOMING VOICE QUERY [{call_sid}]: {user_speech} ---")
        
        # Pass the transcribed text into our LangGraph
        initial_state = {"messages": [HumanMessage(content=user_speech)]}
        for _ in agent_graph.stream(initial_state, VOICE_CONFIG):
            pass
            
        new_state = agent_graph.get_state(VOICE_CONFIG)
        
        if new_state.next and "finalize_booking" in new_state.next:
            ai_text = "I have drafted your appointment. A leasing agent will manually review and confirm it shortly. Goodbye!"
            response.say(ai_text, voice="Polly.Matthew-Neural")
        else:
            # Extract AI response
            ai_text = new_state.values["messages"][-1]
            if hasattr(ai_text, 'content'):
                ai_text = ai_text.content

            # Create a Gather block for ongoing conversation using the absolute URL
            gather = Gather(input="speech", action=ACTION_URL, timeout=3)
            gather.say(ai_text, voice="Polly.Matthew-Neural")
            response.append(gather)
    else:
        # Initial greeting when they first call
        gather = Gather(input="speech", action=ACTION_URL, timeout=3)
        gather.say("Welcome to the Prop Voice Concierge. How can I help you find your next apartment today?", voice="Polly.Matthew-Neural")
        response.append(gather)

    # Keep listening for the next voice input
    #response.gather(input="speech", action="/twillio/voice", timeout=3)
    
    return Response(content=str(response), media_type="text/xml")
    
    ''' DEBUG CODE
    if user_speech:
        print(f"--- INCOMING VOICE QUERY: {user_speech} ---")

        
        # DUMMY REPLY: Bypassing the LLM to test Twilio webhook latency
        ai_text = "I received your message. This is a dummy response to test our connection."
        
        
        
        # Pass the transcribed text into our LangGraph
        initial_state = {"messages": [HumanMessage(content=user_speech)]}
        for _ in agent_graph.stream(initial_state, VOICE_CONFIG):
            pass
            
        new_state = agent_graph.get_state(VOICE_CONFIG)
        
        if new_state.next and "finalize_booking" in new_state.next:
            ai_text = "I have drafted your appointment. A leasing agent will manually review and confirm it shortly. Goodbye!"
            response.say(ai_text, voice="Polly.Matthew-Neural") # Just say, no gather needed if ending
        
        # Create a Gather block for ongoing conversation
        gather = Gather(input="speech", action=ACTION_URL, timeout=3)
        gather.say(ai_text, voice="Polly.Matthew-Neural")
        response.append(gather)

        
        else:
            ai_text = new_state.values["messages"][-1]
            if hasattr(ai_text, 'content'):
                ai_text = ai_text.content
            
            # Create a Gather block for ongoing conversation
            gather = Gather(input="speech", action="/twillio/voice", timeout=3)
            gather.say(ai_text, voice="Polly.Matthew-Neural")
            response.append(gather)
        

    else:
        # Initial greeting when they first call
        gather = Gather(input="speech", action=ACTION_URL, timeout=3)
        gather.say("Welcome to the Prop Voice Concierge. How can I help you find your next apartment today?", voice="Polly.Matthew-Neural")
        response.append(gather)

    # OUR INTERNAL DEBUGGER: Print the exact XML being sent back
    xml_string = str(response)
    print("\n--- GENERATED XML SENT TO TWILIO ---")
    print(xml_string)
    print("------------------------------------\n")

    return Response(content=str(response), media_type="text/xml")
    '''