import os, uuid, requests, math, random
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

# ---- Helpers ----
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def geocode(address: str) -> Optional[Dict[str, float]]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&region=SY&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
    return None

def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&region=SY&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        return data["results"][0]["formatted_address"]
    return None

def format_address(address: str) -> str:
    parts = address.split("،")
    street = ""
    city = ""
    cities = ["دمشق", "حلب", "اللاذقية", "حمص", "حماة", "طرطوس", "دير الزور", "السويداء", "درعا", "الرقة"]
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

# ---- Places Autocomplete ----
def places_autocomplete(query: str, user_lat: float, user_lng: float, max_results=5) -> list:
    url = (
        "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        f"?input={query}"
        f"&key={GOOGLE_MAPS_API_KEY}"
        f"&language=ar"
        f"&components=country:sy"
        f"&location={user_lat},{user_lng}"
        f"&radius=5000"
    )
    data = requests.get(url).json()
    results = []
    if data.get("status") == "OK" and data.get("predictions"):
        for e in data["predictions"][:max_results]:
            results.append({
                "description": e.get("description"),
                "place_id": e.get("place_id"),
            })
        return results
    return []

# ---- Place Details ----
def get_place_details(place_id: str) -> dict:
    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}"
        f"&key={GOOGLE_MAPS_API_KEY}"
        f"&language=ar"
    )
    data = requests.get(url).json()
    if data.get("status") == "OK" and data.get("result"):
        result = data["result"]
        loc = result["geometry"]["location"]
        return {
            "address": result.get("formatted_address"),
            "lat": loc["lat"],
            "lng": loc["lng"],
        }
    return {}
# ---- دالة الحجز الوهمية ----
def create_mock_booking(pickup, destination, time, car_type, audio_pref, user_id=None):
    booking_id = random.randint(10000, 99999)
    print({
        "pickup": pickup,
        "destination": destination,
        "time": time,
        "car_type": car_type,
        "audio_pref": audio_pref,
        "user_id": user_id,
        "booking_id": booking_id,
    })
    return booking_id

# ---- الموديلات ----
class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

# ---- السيناريو ----
ASSISTANT_PROMPT = """
أنت مساعد صوتي ذكي اسمك "يا هو" داخل تطبيق تاكسي متطور. مهمتك مساعدة المستخدمين في حجز المشاوير بطريقة سهلة وودودة.
- استخدم نفس لغة المستخدم في كل رد (عربي أو إنجليزي)
- اسأل سؤالاً واحداً واضحاً في كل مرة
- كن ودوداً ومفيداً
- تذكر المعلومات السابقة في المحادثة
خطوات الحجز:
1. الوجهة
2. نقطة الانطلاق (الموقع الحالي أو مكان آخر)
3. الوقت
4. نوع السيارة (عادية أو VIP)
5. تفضيلات الصوت (قرآن، موسيقى، صمت)
6. ملخص الطلب والتأكيد
"""

@app.post("/chatbot", response_model=BotResponse)
def chatbot(req: UserRequest):
    if not req.sessionId or req.sessionId not in sessions:
        if req.lat is None or req.lng is None:
            return BotResponse(sessionId="", botMessage="يرجى إرسال موقعك الحالي أولاً.")
        sess_id = str(uuid.uuid4())
        loc_txt = get_location_text(req.lat, req.lng)
        sessions[sess_id] = {
            "lat": req.lat,
            "lng": req.lng,
            "step": "ask_destination",
            "history": [
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "assistant", "content": "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖"}
            ],
            "loc_txt": loc_txt,
            "possible_places": None,
            "chosen_place": None,
            "possible_pickup_places": None,
            "pickup": None,
            "time": None,
            "car": None,
            "audio": None
        }
        return BotResponse(sessionId=sess_id, botMessage="مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖")

    sess = sessions[req.sessionId]
    user_msg = (req.userInput or "").strip()
    step = sess.get("step", "ask_destination")

    # -------- الخطوة 1: البحث عن الوجهة --------
    if step == "ask_destination":
        places = places_autocomplete(user_msg, sess["lat"], sess["lng"])
        if not places:
            return BotResponse(sessionId=req.sessionId, botMessage="لم أجد أماكن مطابقة، حاول كتابة اسم أوضح.", done=False)
        if len(places) > 1:
            sess["step"] = "choose_destination"
            sess["possible_places"] = places
            options = "\n".join([f"{i+1}. {p['description']}" for i, p in enumerate(places)])
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"وجدت أكثر من مكان:\n{options}\nيرجى اختيار رقم أو كتابة اسم المكان الصحيح.",
                done=False
            )
        else:
            place_info = get_place_details(places[0]['place_id'])
            sess["chosen_place"] = place_info
            sess["step"] = "ask_pickup"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"✔️ تم اختيار الوجهة: {place_info['address']}.\nمن أين تود الانطلاق؟ من موقعك الحالي ({sess['loc_txt']}) أم من مكان آخر؟",
                done=False
            )

    # -------- الخطوة 2: اختيار الوجهة من القائمة (اسم أو رقم) --------
    if step == "choose_destination":
        places = sess.get("possible_places", [])
        idx = -1
        user_reply = user_msg.strip().lower()
        found = False
        # إذا رقم
        try:
            idx = int(user_reply) - 1
            if 0 <= idx < len(places):
                place_id = places[idx]['place_id']
                place_info = get_place_details(place_id)
                sess["chosen_place"] = place_info
                sess["step"] = "ask_pickup"
                found = True
        except:
            pass
        # إذا نص (يطابق بالوصف)
        if not found:
            for i, p in enumerate(places):
                if user_reply in (p['description'] or '').lower():
                    place_id = p['place_id']
                    place_info = get_place_details(place_id)
                    sess["chosen_place"] = place_info
                    sess["step"] = "ask_pickup"
                    found = True
                    break
        if found:
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"✔️ تم اختيار الوجهة: {sess['chosen_place']['address']}.\nمن أين تود الانطلاق؟ من موقعك الحالي ({sess['loc_txt']}) أم من مكان آخر؟",
                done=False
            )
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="يرجى اختيار رقم أو كتابة اسم المكان كما في القائمة.", done=False)

    # -------- الخطوة 3: تحديد نقطة الانطلاق --------
    if step == "ask_pickup":
        user_reply = user_msg.strip().lower()
        if user_reply in ["موقعي", "موقعي الحالي", "الموقع الحالي"]:
            sess["pickup"] = sess["loc_txt"]
            sess["step"] = "ask_time"
            return BotResponse(sessionId=req.sessionId, botMessage="متى تود الانطلاق؟ الآن أم في وقت محدد؟", done=False)
        else:
            places = places_autocomplete(user_msg, sess["lat"], sess["lng"])
            if not places:
                return BotResponse(sessionId=req.sessionId, botMessage="لم أجد أماكن مطابقة، حاول كتابة اسم أوضح لنقطة الانطلاق.", done=False)
            if len(places) > 1:
                sess["step"] = "choose_pickup"
                sess["possible_pickup_places"] = places
                options = "\n".join([f"{i+1}. {p['description']}" for i, p in enumerate(places)])
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage=f"وجدت أكثر من مكان كنقطة انطلاق:\n{options}\nيرجى اختيار رقم أو كتابة    
