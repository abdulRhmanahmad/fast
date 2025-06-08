import os, json, asyncio, re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()

# === كشف اللغة ===
def detect_language(text: str) -> str:
    text_clean = text.strip().lower()
    arabic_words = len(re.findall(r'[\u0600-\u06FF]+', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    arabic_keywords = ['بدي', 'أروح', 'وين', 'من', 'إلى', 'الساعة', 'تاكسي', 'سيارة', 'عادي', 'مطار', 'جامعة', 'بيت', 'شغل']
    english_keywords = ['want', 'go', 'from', 'to', 'at', 'taxi', 'car', 'airport', 'university', 'home', 'work', 'take', 'me']
    for word in arabic_keywords:
        if word in text_clean:
            arabic_words += 2
    for word in english_keywords:
        if word in text_clean:
            english_words += 2
    if arabic_words > english_words:
        return 'arabic'
    elif english_words > arabic_words:
        return 'english'
    else:
        # افتراضي: عربي
        return 'arabic'

def get_response_templates(language: str) -> dict:
    if language == 'english':
        return {
            'greeting': "Hello! Where would you like to go today? 🚖",
        }
    else:
        return {
            'greeting': "أهلين! وين حابب تروح اليوم؟ 🚖",
        }
system_prompt_base = """
أنت مساعد صوتي اسمي "يا هو" داخل تطبيق تاكسي، تتكلم باللهجة السورية إذا كان المستخدم بالعربي، وتتكلم بالإنجليزي إذا كان المستخدم بالإنجليزي.

## القواعد الأساسية:
- في كل سؤال أو جواب، استخدم دائمًا نفس لغة آخر رسالة أرسلها المستخدم.
- إذا المستخدم كتب بالعربي (مثلاً: "من حلبون"، "الآن"، إلخ)، جاوب بالعربي.
- إذا كتب بالإنجليزي (مثلاً: "now", "from airport"...)، جاوب بالإنجليزي.
- لا تدمج بين العربي والإنجليزي في نفس الرد أبدًا.
- إذا غير المستخدم اللغة أثناء المحادثة، انتقل للرد باللغة الجديدة فورًا في نفس الرسالة.
- اسأل سؤال واحد في كل مرة وانتظر الجواب.
- لا تذكر أنك ذكاء صناعي أو تابع لـ OpenAI.

## ترحيب أولي:
- بالعربي: "أهلين! أنا يا هو، مساعدك للمشاوير. وين حابب تروح اليوم؟"
- بالإنجليزي: "Hello! I’m Yaho, your ride assistant. Where would you like to go today?"

## نصوص الأسئلة والردود:
**بالعربي:**
- إذا ذكر وجهة فقط: "بتحب نجي ناخدك من موقعك الحالي ([اسم الموقع الحالي]) أو في مكان تاني؟"
- إذا نقص معلومات: "إيمتى حابب تطلع؟"
- نوع السيارة: "شو نوع السيارة يلي بتفضلها، عادي ولا VIP؟"
- الموسيقى: "بتحب تسمع شي بالمشوار؟ قرآن، موسيقى، ولا بدون؟"
- إذا نعم: "في قارئ أو نوع تلاوة بتحبها؟"
- ملخص الرحلة: "رحلتك من [الانطلاق] إلى [الوجهة] الساعة [الوقت] بسيارة [نوع السيارة]{، مع تلاوة قرآنية}."
- التأكيد: "ثبتلك الحجز بهالمعلومات؟"
- إذا وافق المستخدم: "✔️ تم! رح أحجزلك الرحلة..."

**بالإنجليزي:**
- If destination only: "Would you like us to pick you up from your current location ([current location]) or somewhere else?"
- If info missing: "What time would you like to leave?"
- Car type: "What type of car do you prefer, regular or VIP?"
- Music: "Would you like to listen to something during the ride? Quran, music, or nothing?"
- If yes: "Any preferred reciter or type of Quran recitation?"
- Summary: "Your trip from [pickup] to [destination] at [time] with a [car type] car{, with Quran recitation}."
- Confirmation: "Shall I confirm the booking with this info?"
- If confirmed: "✔️ Done! Your ride is confirmed..."

## أمثلة:
- المستخدم كتب: "Where is the driver?" → جاوب: "The driver will reach you in 5-10 minutes."
- المستخدم كتب: "وين السائق؟" → جاوب: "السائق رح يوصلك خلال 5-10 دقايق."

تذكّر: **كل خطوة، جاوب بلغة المستخدم دائماً.**
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
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        lang = detect_language(last_user_msg)
        greeting = get_response_templates(lang)['greeting']

        system_prompt = system_prompt_base
        if len(messages) <= 1:  # أول رسالة
            system_prompt += f"\n\nابدأ المحادثة بـ: {greeting}"

        last_q, last_a = extract_last_qa(messages)
        if last_q and last_a:
            system_prompt += f"\n\nآخر سؤال سألته للمستخدم كان:\n{last_q}\nوجاوب المستخدم:\n{last_a}\nانتقل فوراً للسؤال اللي بعده."

        # ⭐️ أهم إضافة: أخبر المساعد بلغة المستخدم
        system_prompt += f"\n\nاللغة الحالية للمستخدم: {'عربي' if lang == 'arabic' else 'English'}.\nجاوب بهذه اللغة فقط في جميع الردود حتى يغير المستخدم اللغة."

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
