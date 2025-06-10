import os, uuid, requests, math
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

# حساب المسافة بين نقطتين (بالكيلو)
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# 1. Geocoding: اسم مكان إلى إحداثيات
def geocode(address: str) -> Optional[Dict[str, float]]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
    return None

# 2. Reverse Geocoding: إحداثيات إلى اسم مكان
def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        return data["results"][0].get("formatted_address", "")
    return None

# 3. البحث عن أماكن (Places API)، يرجع قائمة النتائج مع المسافة والعنوان
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
        # رتبهم حسب الأقرب
        results.sort(key=lambda x: x["distance_km"])
        return results[:max_results]
    return None

# 4. الاقتراحات حسب إحداثيات (أماكن قريبة)
def reverse_places(lat: float, lng: float, max_results=3) -> Optional[List[Dict[str, Any]]]:
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lng}&radius=500&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    )
    data = requests.get(url).json()
    results = []
    if data["status"] == "OK" and data["results"]:
        for res in data["results"]:
            loc = res["geometry"]["location"]
            results.append({
                "name": res.get("name"),
                "address": res.get("vicinity"),
                "lat": loc["lat"],
                "lng": loc["lng"],
            })
        return results[:max_results]
    return None

# 5. حساب زمن الرحلة والمسافة عبر Directions API
def get_trip_info(start_lat, start_lng, end_lat, end_lng):
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={start_lat},{start_lng}&destination={end_lat},{end_lng}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    )
    data = requests.get(url).json()
    if data["status"] == "OK" and data["routes"]:
        leg = data["routes"][0]["legs"][0]
        return {
            "distance": leg["distance"]["text"],      # مثال: "12 كم"
            "duration": leg["duration"]["text"],      # مثال: "16 دقيقة"
            "end_address": leg["end_address"]
        }
    return None

def extract_destination(text: str) -> str:
    prompt = f'استخرج اسم الوجهة من الرسالة التالية بدون أي كلمات إضافية:\n"{text}"'
    rsp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "أجب بالاسم فقط."},
            {"role": "user", "content": prompt},
        ],
    )
    return rsp.choices[0].message.content.strip()

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
    if not lat or not lng:
        return "", "لا أستطيع تحديد موقعك. الرجاء إرسال الإحداثيات أولاً."
    places = reverse_places(lat, lng)
    if not places:
        return "", "لم أستطع تحديد أقرب مكان لك. الرجاء المحاولة مرة أخرى."
    start_name = f"{places[0]['name']}، {places[0]['address']}"
    sess_id = str(uuid.uuid4())
    sessions[sess_id] = {
        "step": "ask_destination",
        "lat": lat,
        "lng": lng,
        "start_name": start_name,
        "dest_name": None,
        "time": None,
        "car": None,
        "audio": None,
        "reciter": None,
        "dest_coords": None,
        "trip_info": None
    }
    return sess_id, f"مرحباً! أنت الآن عند {start_name}. إلى أين تريد الذهاب اليوم؟"

def proceed(session: Dict[str, Any], user_input: str) -> str:
    step = session["step"]

    if step == "ask_destination":
        dest = extract_destination(user_input)
        places = places_search(dest, session["lat"], session["lng"])
        if not places:
            return "تعذر تحديد موقع الوجهة. يرجى كتابة اسم أوضح أو أقرب حي/شارع."
        if len(places) > 1:
            session["possible_dest"] = places
            session["step"] = "choose_dest"
            return ("وجدت أكثر من مكان بهذا الاسم. يرجى اختيار أحدها:\n" +
                "\n".join(f"{i+1}. {p['name']}، {p['address']} (يبعد {p['distance_km']} كم)" for i,p in enumerate(places)))
        chosen = places[0]
        session["dest_name"] = f"{chosen['name']}، {chosen['address']}"
        session["dest_coords"] = (chosen['lat'], chosen['lng'])
        session["step"] = "ask_start"
        return (f"هل تريد أن نأخذك من موقعك الحالي ({session['start_name']}) أم تفضل الانطلاق من مكان آخر؟")

    if step == "choose_dest":
        try:
            idx = int(user_input.strip()) - 1
            chosen = session["possible_dest"][idx]
            session["dest_name"] = f"{chosen['name']}، {chosen['address']}"
            session["dest_coords"] = (chosen['lat'], chosen['lng'])
            session["step"] = "ask_start"
            return (f"تم اختيار: {session['dest_name']}.\n"
                    "هل تريد أن نأخذك من موقعك الحالي أم تفضل الانطلاق من مكان آخر؟")
        except:
            return "الرجاء اختيار رقم صحيح من القائمة."

    if step == "ask_start":
        txt = user_input.strip().lower()
        if txt in {"موقعي", "موقعي الحالي", "الموقع الحالي"}:
            pass  # استخدم start_name المحسوب تلقائياً
        else:
            places = places_search(user_input, session["lat"], session["lng"])
            if not places:
                return "تعذر تحديد موقع الانطلاق. يرجى كتابة اسم أوضح أو أقرب حي/شارع."
            chosen = places[0]
            session["start_name"] = f"{chosen['name']}، {chosen['address']}"
            session["lat"], session["lng"] = chosen['lat'], chosen['lng']
        session["step"] = "ask_time"
        return "متى تريد الانطلاق؟"

    if step == "ask_time":
        session["time"] = user_input
        session["step"] = "ask_car"
        return "ما نوع السيارة التي تفضلها؟ عادية أم VIP؟"

    if step == "ask_car":
        session["car"] = user_input
        session["step"] = "ask_audio"
        return (
            "هل تود الاستماع إلى شيء أثناء الرحلة؟ "
            "يمكنك اختيار القرآن الكريم، الموسيقى، أو الصمت."
        )

    if step == "ask_audio":
        txt = user_input.strip().lower()
        if txt in {"القرآن", "قرآن", "quran"}:
            session["audio"] = "القرآن"
            session["step"] = "ask_reciter"
            return "هل لديك قارئ مفضل أو نوع تلاوة تفضله؟"
        else:
            session["audio"] = user_input
            session["step"] = "summary"
            return build_summary(session)

    if step == "ask_reciter":
        session["reciter"] = user_input
        session["step"] = "summary"
        return build_summary(session)

    if step == "summary":
        # حساب معلومات الرحلة (المسافة والزمن)
        if session.get("dest_coords"):
            trip = get_trip_info(
                session["lat"], session["lng"],
                session["dest_coords"][0], session["dest_coords"][1]
            )
            session["trip_info"] = trip
        if user_input.strip().lower() in {"نعم", "أجل", "أكيد", "موافق"}:
            session["step"] = "confirmed"
            return "تم تأكيد الحجز! ستصلك السيارة في الوقت المحدد."
        else:
            session["step"] = "canceled"
            return "تم إلغاء الحجز بناءً على طلبك."

    return "عذراً، لم أفهم. هل يمكنك التوضيح؟"

def build_summary(s: Dict[str, Any]) -> str:
    base = (
        f"رحلتك من {s['start_name']} إلى {s['dest_name']} "
        f"في الساعة {s['time']} بسيارة {s['car']}"
    )
    if s["trip_info"]:
        base += f"\nالمسافة: {s['trip_info']['distance']}, الوقت المتوقع: {s['trip_info']['duration']}"
    if s["audio"] == "القرآن":
        base += "، مع تلاوة قرآنية"
        if s["reciter"]:
            base += f" بصوت {s['reciter']}"
    return base + ". هل تريد تأكيد الحجز بهذه التفاصيل؟"

@app.post("/chatbot", response_model=BotResponse)
def chatbot(req: UserRequest):
    if not req.sessionId or req.sessionId not in sessions:
        if req.lat is None or req.lng is None:
            return BotResponse(
                sessionId="",
                botMessage="لا أستطيع تحديد موقعك. الرجاء إرسال الإحداثيات أولاً.",
            )
        sess_id, msg = new_session(req.lat, req.lng)
        return BotResponse(sessionId=sess_id, botMessage=msg)

    sess = sessions[req.sessionId]
    reply = proceed(sess, req.userInput or "")
    done = sess.get("step") in {"confirmed", "canceled"}
    return BotResponse(sessionId=req.sessionId, botMessage=reply, done=done)
