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
    parts = [p.strip() for p in address.split(",")]
    street = ""
    city = ""
    cities = ["الخبر", "الدمام", "الرياض", "جدة", "الظهران", "القطيف", "الجبيل", "الأحساء", "بقيق"]
    for p in parts:
        if "شارع" in p or "طريق" in p:
            street = p
            break
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
    return result if result else parts[0]

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
                {"role": "system",
                 "content": (
                    "أنت يا هو، مساعد تاكسي ذكي للمستخدم العربي. سلوكك ودي وعملي. "
                    f"ابدأ المحادثة بتحية، ثم أخبره بموقعه الحالي بشكل مختصر (شارع ومدينة فقط، لا تذكر أرقام أو دولة)، ثم اسأل عن الوجهة المطلوبة. "
                    "لا تكرر نفس السؤال لو لم يجب المستخدم. كلما حصلت على معلومة جديدة (وجهة، وقت، نوع سيارة، صوت... إلخ)، اطلب المعلومة الناقصة فقط حتى يكتمل الحجز. "
                    "إذا أجاب المستخدم على كل الأسئلة، أعطه ملخص مختصر وسأله هل يريد تأكيد الحجز. "
                    "لا تكرر أبداً سؤال تم إجابته. "
                    "لو الوجهة أو الانطلاق فيها غموض (مثل كلمة الجامعة أو المستشفى)، اطلب منه تحديدها."
                 )
                },
                {"role": "assistant", "content": f"مرحباً! أنت حالياً عند: {loc_txt}\nإلى أين الوجهة؟"}
            ]
        }
        return BotResponse(sessionId=sess_id, botMessage=f"مرحباً! أنت حالياً عند: {loc_txt}\nإلى أين الوجهة؟")

    sess = sessions[req.sessionId]
    history = sess["history"]

    # أضف رسالة المستخدم للتاريخ
    user_msg = req.userInput or ""
    history.append({"role": "user", "content": user_msg})

    # احصل على الرد من GPT (يرد حسب كل الحوار والمعلومات المتوفرة)
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history,
            temperature=0.3,
            max_tokens=350,
            timeout=20
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        reply = "⚠️ صار خطأ في الاتصال مع الذكاء الاصطناعي. جرب بعد شوي."

    # أضف رد المساعد للتاريخ
    history.append({"role": "assistant", "content": reply})

    # فحص انتهاء الحوار تلقائي: إذا الجواب فيه عبارة "تم تأكيد الحجز" أو "✅" اعتبره انتهى
    done = any(x in reply for x in ["تم تأكيد الحجز", "تم حجز", "✅", "✔️", "تمت العملية", "تم الحجز"])
    if done:
        # إذا تبي تنهي الجلسة، احذفها:
        # sessions.pop(req.sessionId, None)
        pass

    return BotResponse(sessionId=req.sessionId, botMessage=reply, done=done)
