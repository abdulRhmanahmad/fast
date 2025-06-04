import os, json, asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ملف الحجوزات المؤقَّت
BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()          # قفل لحماية الكتابة

# ───────── system prompt باللهجة الشامية ─────────
system_prompt = """
انت هلأ «يا هو» مساعد حجز تاكسي باللهجة الشامية. تنحصر مهمّتك بتنظيم المشوار:

• سلّم واسأل الوجهة
• اسأل نوع السيارة (عادية / VIP)
• اسأل عن الأغاني
• اطلب أي تفاصيل إضافية (وقت الانطلاق…)
• لخّص هكذا، ثم اختم: «تم! رح أحجزلك وأخبرك لما تكون السيارة جاهزة ✅»

إذا سُئلت عن أي شيء خارج التاكسي ➜ اعتذر بجملة قصيرة وارجع للموضوع.
لا تذكر تفاصيل تقنية ولا أنك "شغال ضمن سوريا".
"""

# ───────── نماذج Pydantic ─────────
class ChatMessage(BaseModel):
    role: str     # "user" أو "assistant"
    content: str

class MessageRequest(BaseModel):
    user_id: str
    messages: list[ChatMessage]     # سجلُّ المحادثة كاملًا

class MessageResponse(BaseModel):
    response: str

# ───────── دوال مساعدة ─────────
def parse_summary(text: str) -> dict:
    # يسحب القيم بعد "• ..."
    def grab(key):
        idx = text.find(key)
        if idx == -1:
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
            arr = []
            if os.path.exists(BOOKINGS_FILE):
                with open(BOOKINGS_FILE, encoding="utf-8") as f:
                    arr = json.load(f)
            arr.append(record)
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(arr, f, ensure_ascii=False, indent=2)
            return booking_id
        except Exception as exc:
            print("❌ JSON save error:", exc)
            return None

# ───────── endpoint /chat ─────────
@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    try:
        # system_prompt + تاريخ المحادثة
        messages = ([
            {"role": "system", "content": system_prompt}
        ] + [m.model_dump() for m in req.messages])

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.4,
            messages=messages
        )
        reply = completion.choices[0].message.content

        # حفظ الحجز إذا ظهرت الجملة النهائية
        if "رح أحجزلك" in reply and "✅" in reply:
            details = parse_summary(reply) | {"user_id": req.user_id}
            if (bid := await save_booking(details)):
                reply += f"\nرقم حجزك: {bid}"

        return MessageResponse(response=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ───────── endpoint /bookings ─────────
@app.get("/bookings", response_model=list[dict])
async def get_bookings():
    if not os.path.exists(BOOKINGS_FILE):
        return []
    with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
