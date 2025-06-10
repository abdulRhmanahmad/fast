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

def places_search(query: str, user_lat: float, user_lng: float, max_results=3) -> Optional[List[Dict[str, Any]]]:
    url = (
        "https://maps.googleapis.com/maps/api/place/textsearch/json"
        f"?query={query}&location={user_lat},{user_lng}&radius=30000"
        f"&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    )
    data = requests.get(url).json()
    results = []
    if data["status"] == "OK" and data["results"]:
        for res in data["results"]:
            loc = res["geometry"]["location"]
            dist = haversine(user_lat, user_lng, loc["lat"], loc["lng"])
            results.append({
                "name": res.get("name"),
                "address": res.get("formatted_address"),
                "lat": loc["lat"],
                "lng": loc["lng"],
                "distance_km": round(dist, 2)
            })
        results.sort(key=lambda x: x["distance_km"])
        return results[:max_results]
    return None

def get_trip_info(start_lat, start_lng, end_lat, end_lng):
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={start_lat},{start_lng}&destination={end_lat},{end_lng}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    )
    data = requests.get(url).json()
    if data["status"] == "OK" and data["routes"]:
        leg = data["routes"][0]["legs"][0]
        return {
            "distance": leg["distance"]["text"],
            "duration": leg["duration"]["text"],
            "end_address": leg["end_address"]
        }
    return None

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

def new_session(lat: float | None, lng: float | None) -> tuple[str, str]:
    start_name = None
    if lat and lng:
        start_name = reverse_geocode(lat, lng)
    sess_id = str(uuid.uuid4())
    sessions[sess_id] = {
        "step": "in_progress",
        "lat": lat,
        "lng": lng,
        "entities": {},
        "summary": None,
        "confirmed": False
    }
    msg = f"مرحباً! أين تريد الذهاب اليوم؟"
    if start_name:
        msg = f"مرحباً! أنت حالياً عند: {start_name}\nأين ترغب بالذهاب؟"
    return sess_id, msg

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
            return BotResponse(
                sessionId="",
                botMessage="يرجى إرسال موقعك الحالي أولاً (إحداثيات).",
            )
        sess_id, msg = new_session(req.lat, req.lng)
        return BotResponse(sessionId=sess_id, botMessage=msg)
    sess = sessions[req.sessionId]
    user_msg = req.userInput or ""
    # تحليل النص واستخراج الكيانات دفعة وحدة
    entities = extract_entities_gpt(user_msg)
    sess["entities"].update({k: v for k, v in entities.items() if v})
    e = sess["entities"]

    # لو فيه غموض
    for field in ["destination", "pickup_location"]:
        if field in e and is_ambiguous_place(e[field]):
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=is_ambiguous_place(e[field]),
                done=False
            )
    # ترتيب الأسئلة حسب الناقص
    if not e.get("destination"):
        return BotResponse(sessionId=req.sessionId, botMessage="إلى أين الوجهة؟", done=False)
    if not e.get("pickup_location"):
        msg = "من أين نبدأ الرحلة؟ إذا أردت موقعك الحالي اكتب: موقعي الحالي"
        return BotResponse(sessionId=req.sessionId, botMessage=msg, done=False)
    if not e.get("ride_time"):
        return BotResponse(sessionId=req.sessionId, botMessage="متى تريد الانطلاق؟", done=False)
    if not e.get("car_type"):
        return BotResponse(sessionId=req.sessionId, botMessage="ما نوع السيارة؟ عادية أم VIP؟", done=False)
    # اختيار صوت/تلاوة
    if not e.get("audio"):
        return BotResponse(sessionId=req.sessionId, botMessage="هل تريد تشغيل شيء أثناء الرحلة؟ مثل القرآن أو موسيقى أو لا شيء.", done=False)
    if e.get("audio", "").strip().lower() in {"القرآن", "قرآن", "quran"} and not e.get("reciter"):
        return BotResponse(sessionId=req.sessionId, botMessage="هل لديك قارئ معين تفضل الاستماع إليه؟", done=False)

    # إذا اكتملت الكيانات، أعطيه ملخص واحسب الرحلة
    if not sess.get("summary"):
        # حساب الإحداثيات إذا لزم
        pickup_coords = geocode(e["pickup_location"]) if e["pickup_location"] != "موقعي الحالي" else {"lat": sess["lat"], "lng": sess["lng"]}
        dest_coords = geocode(e["destination"])
        trip = None
        if pickup_coords and dest_coords:
            trip = get_trip_info(
                pickup_coords["lat"], pickup_coords["lng"], dest_coords["lat"], dest_coords["lng"]
            )
        sess["summary"] = build_summary(e, trip)
        return BotResponse(sessionId=req.sessionId, botMessage=sess["summary"], done=False)
    # بانتظار التأكيد
    txt = user_msg.strip().lower()
    if txt in {"نعم", "أجل", "أكيد", "موافق", "yes", "ok"}:
        sess["confirmed"] = True
        return BotResponse(sessionId=req.sessionId, botMessage="تم تأكيد الحجز! 🚖✔️", done=True)
    if txt in {"لا", "إلغاء", "cancel"}:
        sess["confirmed"] = False
        return BotResponse(sessionId=req.sessionId, botMessage="تم إلغاء الحجز بناء على طلبك.", done=True)
    # إذا لا يزال في مرحلة التأكيد
    return BotResponse(sessionId=req.sessionId, botMessage="هل تريد تأكيد الحجز؟", done=False)
