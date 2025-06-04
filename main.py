from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# 🔸 التعليمة التي تُقيّد النموذج
system_prompt = (
    "أنت مساعد حجز تاكسي محترف. "
    "لا تتحدث إلا عن خدمة التاكسي والرحلات. "
    "إذا طُلب منك أي موضوع خارج هذا النطاق، اعتذر بأدب وكرر أنك مختص بالحجز فقط."
)

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str

@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": req.message}
            ]
        )
        reply = completion.choices[0].message.content
        return MessageResponse(response=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
