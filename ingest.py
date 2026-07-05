import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from database import SessionLocal, PropertyListing

load_dotenv()

def load_data():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Initialize the free Hugging Face embedding model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

    # Load JSON file
    with open("properties.json", "r") as f:
        properties_data = json.load(f)

    session = SessionLocal()

    print("Generating embeddings and saving to database...")
    for prop in properties_data:
        # Check if property already exists to avoid duplicates during testing
        existing = session.query(PropertyListing).filter(PropertyListing.id == prop["id"]).first()
        if existing:
            print(f"Skipping {prop['name']}, already in database.")
            continue
            
        # Generate the vector embedding for the description
        vector = embeddings.embed_query(prop["description"])
        
        # Create the database record
        new_property = PropertyListing(
            id=prop["id"],
            name=prop["name"],
            neighborhood=prop["neighborhood"],
            price=prop["price"],
            bedrooms=prop["bedrooms"],
            pet_friendly=prop["pet_friendly"],
            description=prop["description"],
            embedding=vector
        )
        
        session.add(new_property)
        print(f"Successfully ingested: {prop['name']}")

    session.commit()
    session.close()
    print("All data successfully stored in PGVector!")

if __name__ == "__main__":
    load_data()