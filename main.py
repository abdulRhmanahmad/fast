from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI          
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "FastAPI Chat API is running! Use POST /chat to send messages."}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "FastAPI Chat API"}


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": request.message}]
        )
        reply = completion.choices[0].message.content
        return MessageResponse(response=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
