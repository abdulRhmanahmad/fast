import os, json, asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()   # قفل لحماية الكتابة المتزامنة

# ─── system prompt باللهجة الشامية ───
system_prompt = """
انت هلأ «يا هو» مساعد حجز تاكسي شغال ضمن سوريا وبتحكي باللهجة الشامية البسيطة.
مهمّتك تنظيم مشوار التاكسي وما بتحكي إلا بهالمجال.

• سلّم واسأل الوجهة
• اسأل نوع السيارة (عادية / VIP)
• اسأل عن الأغاني
• اطلب أي تفاصيل إضافية
• لخّص بهذا الشكل:

• الوجهة: …
• نوع السيارة: …
• وقت الانطلاق: …
• الأغاني: …
• ملاحظات: …

وختم بـ «تم! رح أحجزلك وأخبرك لما تكون السيارة جاهزة ✅»
إذا سُئلت عن شيء خارج التاكسي ➜ «آسف يا غالي…»
"""

# ─── نماذج ───
class MessageRequest(BaseModel):
    user_id: str
    message: str

class MessageResponse(BaseModel):
    response: str

# ─── مساعدا ت ───
def parse_summary(text: str) -> dict:
    def grab(key):
        if (idx := text.find(key)) == -1:
            return ""
        start = idx + len(key)
        end = text.find("\n", start)
        return text[start:end].strip(" :") if end != -1 else text[start:].strip(" :")
    return {
        "destination": grab("الوجهة"),
        "car_type":   grab("نوع السيارة"),
        "ride_time":  grab("وقت الانطلاق"),
        "music":      grab("الأغاني"),
        "notes":      grab("ملاحظات")
    }

async def save_booking(data: dict) -> str | None:
    async with file_lock:
        booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        record = {"booking_id": booking_id,
                  "timestamp": datetime.now().isoformat(),
                  **data}
        try:
            if os.path.exists(BOOKINGS_FILE):
                with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
                    arr = json.load(f)
            else:
                arr = []
            arr.append(record)
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(arr, f, ensure_ascii=False, indent=2)
            return booking_id
        except Exception as e:
            print("❌ JSON save error:", e)
            return None

# ─── /chat ───
@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": req.message}
            ]
        )
        reply = completion.choices[0].message.content

        # حفظ الحجز إذا اكتمل
        if "رح أحجزلك" in reply and "✅" in reply:
            details = parse_summary(reply) | {"user_id": req.user_id}
            if (bid := await save_booking(details)):
                reply += f"\nرقم حجزك: {bid}"

        return MessageResponse(response=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── /bookings ───
@app.get("/bookings", response_model=list[dict])
async def get_bookings():
    if not os.path.exists(BOOKINGS_FILE):
        return []
    with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
