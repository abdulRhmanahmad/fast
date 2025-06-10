import os, uuid, requests, math, json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

# برومبت ثابت يضبط شخصية وسيناريو المساعد يا هو
ASSISTANT_PROMPT = """
أنت مساعد صوتي ذكي اسمك "يا هو" داخل تطبيق تاكسي متطور. مهمتك مساعدة المستخدمين في حجز المشاوير بطريقة سهلة وودودة.

# القواعد الأساسية:
- استخدم نفس لغة المستخدم في كل رد (عربي أو إنجليزي)
- لا تخلط بين اللغات في نفس الرد
- اسأل سؤالاً واحداً واضحاً في كل مرة
- كن ودوداً ومفيداً
- تذكر المعلومات السابقة في المحادثة

# خطوات الحجز:
1. الترحيب والسؤال عن الوجهة.
2. تحديد نقطة الانطلاق (الموقع الحالي للمستخدم [اكتب اسم الشارع والمدينة فقط، بدون أرقام أو دولة] أو مكان آخر).
3. السؤال عن وقت الانطلاق.
4. نوع السيارة (عادية أم VIP).
5. تفضيلات الصوت (قرآن، موسيقى، صمت).
6. ملخص الحجز والتأكيد.

# أمثلة على الأسئلة:
- "من أين تود الانطلاق؟ من موقعك الحالي ([اكتب اسم الشارع والمدينة فقط]) أم من مكان آخر؟"
- "متى تود الانطلاق؟ الآن أم في وقت محدد؟"
- "أي نوع سيارة تفضل؟ عادية أم VIP؟"
- "هل تود الاستماع لشيء أثناء الرحلة؟"

# ردود الترحيب:
- عربي: "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖"
- إنجليزي: "Hello! I'm Yaho, your smart ride assistant. Where would you like to go today? 🚖"

# التأكيد النهائي:
عند اكتمال جميع المعلومات، اعرض ملخصاً كاملاً واطلب التأكيد هكذا:
"ملخص رحلتك:
• الوجهة: [الوجهة]
• الانطلاق: [نقطة الانطلاق]
• الوقت: [وقت الانطلاق]
• نوع السيارة: [عادية/VIP]
• الصوت: [قرآن/موسيقى/صمت]

هل تؤكد الحجز؟"

# بعد التأكيد:
"✔️ تم تأكيد حجزك بنجاح! سيصلك السائق في الوقت المحدد."

تذكر: كن طبيعياً وودوداً، واستخدم الرموز التعبيرية بشكل مناسب.
"""

def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def geocode(address: str) -> Optional[Dict[str, float]]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
    return None

def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        return data["results"][0].get("formatted_address", "")
    return None

def format_address(address: str) -> str:
    # مثال: "8844 شارع بدر، 2888، الثقبة، الخبر 34623، السعودية"
    # نريد: "شارع بدر، الخبر"
    parts = [p.strip() for p in address.split(",")]
    street = ""
    city = ""
    cities = ["الخبر", "الدمام", "الرياض", "جدة", "الظهران", "القطيف", "الجبيل", "الأحساء", "بقيق"]
    # ابحث عن الشارع
    for p in parts:
        if "شارع" in p or "طريق" in p:
            street = p
            break
    # ابحث عن المدينة
    for p in parts:
        for city_name in cities:
            if city_name in p:
                city = city_name
                break
    result = ""
    if street:
        result += street
    if city:
        result += ("، " if street else "") + city
    return result if result else parts[0]  # fallback

def get_location_text(lat, lng):
    address = reverse_geocode(lat, lng)
    if not address:
        return "موقعك غير معروف"
    return format_address(address)

class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

@app.post("/chatbot", response_model=BotResponse)
def chatbot(req: UserRequest):
    # بدء جلسة جديدة إذا لزم
    if not req.sessionId or req.sessionId not in sessions:
        if req.lat is None or req.lng is None:
            return BotResponse(sessionId="", botMessage="يرجى إرسال موقعك الحالي أولاً.",)
        sess_id = str(uuid.uuid4())
        loc_txt = get_location_text(req.lat, req.lng)
        sessions[sess_id] = {
            "lat": req.lat,
            "lng": req.lng,
            "history": [
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "assistant", "content": "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖"}
            ],
            "loc_txt": loc_txt
        }
        return BotResponse(sessionId=sess_id, botMessage="مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖")

    sess = sessions[req.sessionId]
    history = sess["history"]

    # إضافة رسالة المستخدم
    user_msg = req.userInput or ""
    history.append({"role": "user", "content": user_msg})

    # مرر الموقع الحالي في كل برومبت جديد كمعلومة
    location_text = sess.get("loc_txt")
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history + [{"role": "system", "content": f"اسم الموقع الحالي للمستخدم هو: {location_text}"}],
            temperature=0.3,
            max_tokens=350,
            timeout=20
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        reply = "⚠️ صار خطأ في الاتصال مع الذكاء الاصطناعي. جرب بعد شوي."

    # حفظ رد المساعد
    history.append({"role": "assistant", "content": reply})

    done = any(x in reply for x in ["تم تأكيد حجزك", "✔️", "تم الحجز", "✅"])
    return BotResponse(sessionId=req.sessionId, botMessage=reply, done=done)
