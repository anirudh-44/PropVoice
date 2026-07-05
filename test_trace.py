import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# Load environment variables (this activates the LangSmith tracing)
load_dotenv()

def run_traced_query():
    print("Sending traced query to Hugging Face...")
    
    # Initialize the LLM
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash",
        temperature=0.1,
        max_new_tokens=250,
    )

    model = ChatHuggingFace(llm=llm)
    
    # Create a simple prompt template
    prompt = PromptTemplate.from_template(
        "You are an expert real estate assistant. Explain why {amenity} is highly valued by renters in exactly two sentences."
    )
    
    # Chain the prompt and the LLM
    chain = prompt | model
    
    # Execute the chain
    response = chain.invoke({"amenity": "in-unit laundry"})
    
    print("\n--- AI Response ---")
    print(response)
    print("\nTrace dispatched! Check your LangSmith dashboard.")

if __name__ == "__main__":
    run_traced_query()