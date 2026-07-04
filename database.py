import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PropertyListing(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    neighborhood = Column(String)
    price = Column(Float)
    bedrooms = Column(Integer)
    pet_friendly = Column(Boolean)
    description = Column(Text)
    
    # We use a dimension of 384, which matches the BAAI/bge-small-en-v1.5 embedding model
    embedding = Column(Vector(384))

# Create the table and enable the vector extension
def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    print("Database tables and pgvector extension initialized.")

if __name__ == "__main__":
    from sqlalchemy import text
    init_db()