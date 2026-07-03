import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

def test_connection():
    print("Connecting to Hugging Face...")
    
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash",
        temperature=0.1,
        max_new_tokens=250
    )

    model = ChatHuggingFace(llm=llm)
    
    response = model.invoke("What are the top 3 amenities tenants look for in a luxury apartment?")
    print("\n--- API Response ---")
    print(response)

if __name__ == "__main__":
    test_connection()