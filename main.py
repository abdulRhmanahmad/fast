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
    # مثال: "8844 شارع بدر، 2888، الثقبة، الخبر 34623، السعودية"
    # نريد: "شارع بدر، الخبر"
    parts = [p.strip() for p in address.split(",")]
    street = ""
    city = ""
    cities = ["الخبر", "الدمام", "الرياض", "جدة", "الظهران", "القطيف", "الجبيل", "الأحساء", "بقيق"]
    # ابحث عن شارع
    for p in parts:
        if "شارع" in p or "طريق" in p:
            street = p
            break
    # ابحث عن مدينة
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

def extract_entities_gpt(text: str) -> Dict[str, str]:
    prompt = f"""
    استخرج من الرسالة التالية معلومات الحجز (بصيغة JSON فقط): 
    {{"pickup_location": "", "destination": "", "ride_time": "", "car_type": "", "audio": "", "reciter": ""}}
    إذا لم يوجد حقل اتركه فارغاً.
    نص المستخدم: {text}
    الجواب فقط JSON:
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            timeout=10
        )
        entities = completion.choices[0].message.content
        return json.loads(entities)
    except Exception:
        return {}

def is_ambiguous_place(place: str) -> Optional[str]:
    ambiguous = {
        "الجامعة": "يوجد عدة جامعات. حدد اسم الجامعة (مثلاً: جامعة الملك سعود، الجامعة الافتراضية...).",
        "المطار": "أي مطار تقصد؟ (مثلاً: مطار الرياض، مطار جدة...).",
        "المول": "حدد اسم المول.",
        "المستشفى": "حدد اسم المستشفى.",
        "البيت": "حدد العنوان بدقة."
    }
    return ambiguous.get(place.strip(), None)

class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

def build_summary(entities: Dict[str, Any], trip: dict | None) -> str:
    parts = []
    if entities.get("pickup_location"):
        parts.append(f"من: {entities['pickup_location']}")
    if entities.get("destination"):
        parts.append(f"إلى: {entities['destination']}")
    if entities.get("ride_time"):
        parts.append(f"وقت الانطلاق: {entities['ride_time']}")
    if entities.get("car_type"):
        parts.append(f"نوع السيارة: {entities['car_type']}")
    if entities.get("audio"):
        part = "تشغيل"
        if entities['audio'] == "القرآن":
            part += " القرآن الكريم"
            if entities.get("reciter"):
                part += f" بصوت {entities['reciter']}"
        else:
            part += f" {entities['audio']}"
        parts.append(part)
    summary = "، ".join(parts)
    if trip:
        summary += f"\nالمسافة: {trip['distance']}, الوقت المتوقع: {trip['duration']}"
    summary += "\nهل تريد تأكيد الحجز بهذه التفاصيل؟"
    return summary

@app.post("/chatbot", response_model=BotResponse)
def chatbot(req: UserRequest):
    # بدء جلسة جديدة إذا لزم
    if not req.sessionId or req.sessionId not in sessions:
        if req.lat is None or req.lng is None:
            return BotResponse(sessionId="", botMessage="يرجى إرسال موقعك الحالي أولاً.",)
        sess_id = str(uuid.uuid4())
        loc_txt = get_location_text(req.lat, req.lng)
        msg = f"مرحباً! أنت حالياً عند: {loc_txt}\nإلى أين الوجهة؟"
        sessions[sess_id] = {"step": "in_progress", "lat": req.lat, "lng": req.lng, "entities": {}, "last_question": "destination"}
        return BotResponse(sessionId=sess_id, botMessage=msg)

    sess = sessions[req.sessionId]
    user_msg = req.userInput or ""
    entities = extract_entities_gpt(user_msg)
    sess["entities"].update({k: v for k, v in entities.items() if v})
    e = sess["entities"]

    # سؤال الوجهة فقط مرة واحدة
    if not e.get("destination"):
        if sess.get("last_question") != "destination":
            sess["last_question"] = "destination"
            return BotResponse(sessionId=req.sessionId, botMessage="إلى أين الوجهة؟", done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="بانتظار تحديد الوجهة...", done=False)
    # باقي الأسئلة بنفس المنطق (تقدر تطبق على الوقت ونوع السيارة...)
    if not e.get("pickup_location"):
        if sess.get("last_question") != "pickup_location":
            sess["last_question"] = "pickup_location"
            msg = "من أين نبدأ الرحلة؟ إذا أردت موقعك الحالي اكتب: موقعي الحالي"
            return BotResponse(sessionId=req.sessionId, botMessage=msg, done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="بانتظار تحديد مكان الانطلاق...", done=False)
    if not e.get("ride_time"):
        if sess.get("last_question") != "ride_time":
            sess["last_question"] = "ride_time"
            return BotResponse(sessionId=req.sessionId, botMessage="متى تريد الانطلاق؟", done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="بانتظار تحديد وقت الانطلاق...", done=False)
    if not e.get("car_type"):
        if sess.get("last_question") != "car_type":
            sess["last_question"] = "car_type"
            return BotResponse(sessionId=req.sessionId, botMessage="ما نوع السيارة؟ عادية أم VIP؟", done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="بانتظار اختيار نوع السيارة...", done=False)
    if not e.get("audio"):
        if sess.get("last_question") != "audio":
            sess["last_question"] = "audio"
            return BotResponse(sessionId=req.sessionId, botMessage="هل تريد تشغيل شيء أثناء الرحلة؟ مثل القرآن أو موسيقى أو لا شيء.", done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="بانتظار اختيار الصوت...", done=False)
    if e.get("audio", "").strip().lower() in {"القرآن", "قرآن", "quran"} and not e.get("reciter"):
        if sess.get("last_question") != "reciter":
            sess["last_question"] = "reciter"
            return BotResponse(sessionId=req.sessionId, botMessage="هل لديك قارئ معين تفضل الاستماع إليه؟", done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="بانتظار تحديد القارئ...", done=False)

    # إذا اكتملت الكيانات، أعطيه ملخص واحسب الرحلة (اختياري: أضف حساب المسافة)
    if not sess.get("summary"):
        sess["summary"] = build_summary(e, None)  # ممكن تضيف معلومات الرحلة لو حاب
        return BotResponse(sessionId=req.sessionId, botMessage=sess["summary"], done=False)

    # بانتظار التأكيد
    txt = user_msg.strip().lower()
    if txt in {"نعم", "أجل", "أكيد", "موافق", "yes", "ok"}:
        sess["confirmed"] = True
        return BotResponse(sessionId=req.sessionId, botMessage="تم تأكيد الحجز! 🚖✔️", done=True)
    if txt in {"لا", "إلغاء", "cancel"}:
        sess["confirmed"] = False
        return BotResponse(sessionId=req.sessionId, botMessage="تم إلغاء الحجز بناء على طلبك.", done=True)
    return BotResponse(sessionId=req.sessionId, botMessage="هل تريد تأكيد الحجز؟", done=False)
