import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from database import SessionLocal, PropertyListing

load_dotenv()

# Initialize the exact same embedding model used for ingestion
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token
)

def search_properties_vector(query_text: str, limit: int = 2):
    """
    Embeds the user query and performs an L2 distance vector search in PGVector.
    """
    # 1. Convert the user's natural language question into a vector
    query_vector = embeddings.embed_query(query_text)
    
    session = SessionLocal()
    try:
        # 2. Use the pgvector l2_distance operator to find the closest matches
        results = session.query(PropertyListing).order_by(
            PropertyListing.embedding.l2_distance(query_vector)
        ).limit(limit).all()
        
        # 3. Format the results into clean text context for the LLM
        context_blocks = []
        for prop in results:
            context_blocks.append(
                f"Property: {prop.name} in {prop.neighborhood}\n"
                f"Price: ${prop.price}/month | Bedrooms: {prop.bedrooms}\n"
                f"Pet Friendly: {prop.pet_friendly}\n"
                f"Description: {prop.description}\n"
                f"---"
            )
        return "\n".join(context_blocks)
    finally:
        session.close()