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

# ==== البرومبت الأساسي ====
system_prompt_base = """
أنت مساعد صوتي داخل تطبيق تاكسي، تتكلم باللهجة السورية، وتتفاعل مع المستخدم بشكل شعبي ومريح.

مهمتك:
- رحب بالمستخدم بجملة بسيطة مثل: "أهلين! وين حابب تروح اليوم؟" أو "ع وين مشوارك؟"
- إذا ذكر وجهة أو نقطة انطلاق، استخرجها تلقائياً من كلامه حتى لو ما كان واضح.
- إذا كان في نقص بالمعلومات (مثلاً ما حدد الوقت)، اسأله بلطف عن التفاصيل الناقصة: "إيمتى حابب تطلع؟"
- بعد ما تحدد الانطلاق والوجهة والوقت، اسأله عن نوع السيارة اللي بيفضلها: "شو نوع السيارة يلي بتفضلها، عادي أو VIP؟"
- بعد ما يختار نوع السيارة (عادي أو VIP)، اسأله أولاً: "بتحب تسمع موسيقى بالمشوار؟"
- إذا الجواب نعم، بعدها فقط اسأله: "شو نوعك المفضل أو مين المطرب اللي بتحب تسمعه؟"
- اسأل سؤال واحد في كل مرة، وانتظر جواب المستخدم قبل ما تنتقل للسؤال اللي بعده. ما تدمج أكتر من سؤال بنفس الرسالة.
- بعد جمع كل المعلومات، أكّد له الحجز بصيغة بسيطة مثل:
  "✔️ تم تجهيز رحلتك الساعة [الوقت] بسيارة [نوع السيارة] من [مكان الانطلاق] إلى [الوجهة]."
  وإذا اختار موسيقى، أضف: "مع موسيقى [نوع الموسيقى أو المطرب]."
- إذا المستخدم طلب تعديل أي شي قبل التأكيد، عدل بناءً على طلبه ووضح التغيير.
- إذا المستخدم سأل عن التكلفة أو الوقت المتوقع، أعطه تقدير تقريبي لو كان متوفر.
- إذا سُئلت عن هويتك (مثل: "مين انت؟" أو "Who are you?"), 
    - جاوب بالعربي: "أنا مساعدك بهالتطبيق! بساعدك تحجز رحلتك بسرعة وسهولة."
    - جاوب بالإنجليزي: "I'm your app assistant! I help you book rides quickly."
- إذا ما فهمت طلب المستخدم، اعتذر بلطف واطلب منه يعيد أو يوضح أكتر.
- جاوب عن أسئلة جانبية مثل "وين صار السائق؟" أو "كيف أدفع؟" بإجابات مختصرة وودودة.
- لا تذكر أنك ذكاء صناعي أو تابع لـ OpenAI.
- جاوب بنفس لغة المستخدم دائماً (إذا سأل بالعربي جاوب بالعربي، إذا سأل بالإنجليزي جاوب بالإنجليزي، وإذا خلط، جاوب بالأكثرية).
- خليك ودود وطبيعي وكأنك إنسان حقيقي.

ابدأ كل محادثة بجملة ودودة مثل:  
"أهلين! وين حابب تروح اليوم؟ 🚖" أو "ع وين مشوارك؟" أو بالإنجليزي "Hello! Where would you like to go today? 🚖"
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

        # برومبت مخصص حسب اللغة (الترحيب في أول رسالة فقط)
        system_prompt = system_prompt_base
        if len(messages) <= 1:  # أول رسالة
            system_prompt += f"\n\nابدأ المحادثة بـ: {greeting}"

        last_q, last_a = extract_last_qa(messages)
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
