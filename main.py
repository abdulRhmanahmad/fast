import os, json, asyncio, re, requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()

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
        return 'arabic'

def get_response_templates(language: str) -> dict:
    if language == 'english':
        return {'greeting': "Hello! Where would you like to go today? 🚖"}
    else:
        return {'greeting': "مرحباً! أين تود الذهاب اليوم؟ 🚖"}

# ========== تحقق من الموقع ==========
def check_place_exists(place: str) -> bool:
    api_key = GOOGLE_MAPS_API_KEY
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place}&key={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        return bool(data.get("results"))
    except Exception as e:
        print("Geocoding error:", e)
        return False

system_prompt_base = """
أنت مساعد صوتي اسمي "يا هو" داخل تطبيق تاكسي، تتكلم بالفصحى إذا كان المستخدم بالعربية، وتتكلم بالإنجليزية إذا كان المستخدم بالإنجليزية.

## القواعد الأساسية:
- في كل سؤال أو جواب، استخدم دائمًا نفس لغة آخر رسالة أرسلها المستخدم.
- لا تدمج بين العربي والإنجليزي في نفس الرد.
- اسأل سؤال واحد في كل مرة.

## ترحيب أولي:
- بالعربية: "مرحباً! أنا يا هو، مساعدك للمشاوير. أين تود الذهاب اليوم؟"
- بالإنجليزية: "Hello! I’m Yaho, your ride assistant. Where would you like to go today?"

## الأسئلة:
- إذا ذكر وجهة فقط: "هل ترغب أن نأخذك من موقعك الحالي ([اسم الموقع الحالي]) أم من مكان آخر؟"
- إذا نقص معلومات: "متى تود الانطلاق؟"
- نوع السيارة: "ما نوع السيارة التي تفضلها؟ عادية أم VIP؟"
- الصوت أثناء الرحلة: "هل تود الاستماع إلى شيء أثناء الرحلة؟ قرآن، موسيقى، أم لا شيء؟"
- إن قال قرآن: "هل تفضل قارئًا معينًا أو نوع تلاوة؟"
- ملخص الرحلة: "رحلتك من [الانطلاق] إلى [الوجهة] في الساعة [الوقت] بسيارة [نوع السيارة]{، مع تلاوة قرآنية}."
- التأكيد: "هل أؤكد الحجز بهذه المعلومات؟"
- إذا وافق المستخدم: "✔️ تم تأكيد حجزك..."

تذكر: أجب دائمًا بلغة المستخدم.
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
        "destination": grab("الوجهة") or grab("destination"),
        "pickup_location": grab("من وين") or grab("الانطلاق"),
        "car_type": grab("نوع السيارة") or grab("السيارة"),
        "ride_time": grab("الوقت") or grab("وقت الانطلاق"),
        "music": grab("الأغاني") or grab("الموسيقا"),
        "notes": grab("ملاحظات") or grab("طلبات خاصة")
    }

def get_google_maps_link(destination: str) -> str:
    dest_encoded = destination.replace(" ", "+")
    return f"https://www.google.com/maps/search/?api=1&query={dest_encoded}"

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

        # 🚨 تحقق إذا المستخدم ذكر مكان (مكان استلام أو وجهة) قبل متابعة GPT
        # ابحث عن اسم مكان واضح في آخر رسالة
        place_keywords = ["من", "الى", "إلى", "to", "from", "destination", "pickup", "الوجهة", "مكان"]
        place_detected = any(k in last_user_msg.lower() for k in place_keywords)
        place_name = last_user_msg.strip()
        # تحقق إذا فيه جملة مثل: "من [مكان]"
        # إذا بدك تدقق أكثر استخدم regex أو معالجة متقدمة حسب مشروعك

        if place_detected and len(place_name) > 2:
            if not check_place_exists(place_name):
                msg = (
                    "الموقع المدخل غير موجود على الخريطة. جرب تكتب اسم المكان بشكل أوضح."
                    if lang == "arabic"
                    else "The entered location could not be found on the map. Please try a clearer name."
                )
                return MessageResponse(response=msg)

        system_prompt = system_prompt_base
        if len(messages) <= 1:
            system_prompt += f"\n\nابدأ المحادثة بـ: {greeting}"

        last_q, last_a = extract_last_qa(messages)
        if last_q and last_a:
            system_prompt += f"\n\nآخر سؤال سألته للمستخدم كان:\n{last_q}\nوجاوب المستخدم:\n{last_a}\nانتقل للسؤال التالي."

        system_prompt += f"\n\nلغة المستخدم: {'عربية' if lang == 'arabic' else 'English'}"
        system_prompt += f"\nمعرف المستخدم: {req.user_id}"
        system_prompt += f"\nالوقت: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        all_msgs = [{"role": "system", "content": system_prompt}] + messages

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=300,
            messages=all_msgs
        )
        reply = completion.choices[0].message.content

        booking_indicators = ["تم تأكيد حجزك", "✔️", "تم!", "تم الحجز"]
        if any(ind in reply for ind in booking_indicators):
            details = parse_summary(reply)
            if details.get("destination"):
                details["user_id"] = req.user_id
                if (booking_id := await save_booking(details)):
                    reply += f"\n\n📱 رقم حجزك: {booking_id}"

        # 🔗 أضف رابط Google Maps إذا تم تحديد الوجهة
        dest_match = re.search(r"(?:الوجهة|destination):\s*(.+)", reply, re.IGNORECASE)
        if dest_match:
            location = dest_match.group(1).strip()
            maps_link = get_google_maps_link(location)
            reply += f"\n\n🗺️ يمكنك رؤية الموقع على الخريطة:\n{maps_link}"

        return MessageResponse(response=reply)
    except Exception as e:
        print(f"❌ خطأ في معالجة الرسالة: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء المعالجة")

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
