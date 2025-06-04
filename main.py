import os, json, asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()

system_prompt_base = """
أنت "يا هو" - مساعد حجز تاكسي باللهجة الشامية السورية، دورك تسهّل الحجز خطوة خطوة.
- اسأل سؤال واحد فقط في كل رسالة.
- إذا المستخدم جاوب على سؤال، انتقل فوراً للسؤال اللي بعده ولا تعيد نفس السؤال.
- لو أعطاك المستخدم معلومة غير متوقعة أو ناقصة، وضح له المطلوب بدون تكرار السؤال السابق حرفياً.
- إذا اكتملت كل المعلومات، لخّص الحجز بالتفصيل ثم قل له: "تم الحجز".
- لا تجاوب إلا عن الحجوزات، لو سأل عن شي ثاني اعتذر باختصار.

### أمثلة:
المستخدم: "بدي روح ع التل"
الرد: "حلو، التل. ومن وين رح نجيك؟"
المستخدم: "من الشام الجديدة"
الرد: "تمام، من الشام الجديدة للتل. بتفضل سيارة عادية ولا VIP؟"
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class MessageRequest(BaseModel):
    user_id: str
    messages: list[ChatMessage]

class MessageResponse(BaseModel):
    response: str

def extract_last_qa(messages: list[dict]) -> tuple[str, str]:
    """
    دور على آخر سؤال من المساعد وجواب المستخدم عليه
    """
    for i in reversed(range(len(messages))):
        if messages[i]["role"] == "assistant" and i+1 < len(messages):
            user_msg = messages[i+1]
            if user_msg["role"] == "user":
                return messages[i]["content"], user_msg["content"]
    return "", ""

def parse_summary(text: str) -> dict:
    def grab(key):
        patterns = [f"• {key}:", f"•{key}:", f"{key}:"]
        for pattern in patterns:
            idx = text.find(pattern)
            if idx != -1:
                start = idx + len(pattern)
                end = text.find("\n", start)
                result = text[start:end].strip(" :[]") if end != -1 else text[start:].strip(" :[]")
                return result if result and result != "-" else ""
        return ""
    return {
        "destination": grab("الوجهة") or grab("الوجهه"),
        "pickup_location": grab("من وين") or grab("نقطة الانطلاق") or grab("الانطلاق"),
        "car_type": grab("نوع السيارة") or grab("السيارة"),
        "ride_time": grab("وقت الانطلاق") or grab("الوقت"),
        "music": grab("الأغاني") or grab("الموسيقا"),
        "notes": grab("ملاحظات") or grab("طلبات خاصة")
    }

async def save_booking(data: dict) -> str | None:
    async with file_lock:
        booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        record = {
            "booking_id": booking_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            **data
        }
        try:
            bookings = []
            if os.path.exists(BOOKINGS_FILE):
                with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        bookings = json.loads(content)
            bookings.append(record)
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(bookings, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ الحجز: {booking_id}")
            return booking_id
        except Exception as exc:
            print(f"❌ خطأ في حفظ الحجز: {exc}")
            return None

@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    try:
        messages = [m.model_dump() for m in req.messages]
        last_q, last_a = extract_last_qa(messages)
        # لو فيه سؤال وجواب، أضفهم للبرومبت!
        system_prompt = system_prompt_base
        if last_q and last_a:
            system_prompt += f"\n\nآخر سؤال سألته للمستخدم كان:\n{last_q}\nوجاوب المستخدم:\n{last_a}\nانتقل فوراً للسؤال اللي بعده."
        system_prompt += f"\n\nمعرف المستخدم: {req.user_id}\nالوقت الحالي: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        all_msgs = [{"role": "system", "content": system_prompt}] + messages

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=300,
            messages=all_msgs
        )
        reply = completion.choices[0].message.content

        # شرط الحجز النهائي
        booking_indicators = ["رح أحجزلك", "تم!", "✅", "تم الحجز"]
        if any(ind in reply for ind in booking_indicators):
            details = parse_summary(reply)
            if details.get("destination"):
                details["user_id"] = req.user_id
                if (booking_id := await save_booking(details)):
                    reply += f"\n\n📱 رقم حجزك: {booking_id}"

        return MessageResponse(response=reply)
    except Exception as e:
        print(f"❌ خطأ في معالجة الرسالة: {e}")
        raise HTTPException(
            status_code=500, 
            detail="عذراً، صار خطأ تقني. جرب مرة تانية."
        )

@app.get("/bookings", response_model=list[dict])
async def get_bookings():
    try:
        if not os.path.exists(BOOKINGS_FILE):
            return []
        with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except Exception as e:
        print(f"❌ خطأ في قراءة الحجوزات: {e}")
        raise HTTPException(status_code=500, detail="خطأ في جلب الحجوزات")

@app.get("/booking/{booking_id}")
async def get_booking_status(booking_id: str):
    try:
        bookings = await get_bookings()
        booking = next((b for b in bookings if b["booking_id"] == booking_id), None)
        if not booking:
            raise HTTPException(status_code=404, detail="الحجز غير موجود")
        return booking
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
