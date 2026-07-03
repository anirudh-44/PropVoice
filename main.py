from fastapi import FastAPI

app = FastAPI(title="PropVoice API")

@app.get("/")
async def root():
    return {"message": "PropVoice Backend is Running", "status": "Healthy"}