import os, uuid, requests, math
from typing import Optional, Dict, Any
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

# Places API فقط (بدون شرط الشرقية)
def places_search(query: str, user_lat: float, user_lng: float):
    url = (
        "https://maps.googleapis.com/maps/api/place/textsearch/json"
        f"?query={query}&location={user_lat},{user_lng}&radius=30000"
        f"&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    )
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        best = None
        best_dist = float("inf")
        for res in data["results"]:
            place_id = res["place_id"]
            details_url = (
                "https://maps.googleapis.com/maps/api/place/details/json"
                f"?place_id={place_id}&language=ar&region=SA&key={GOOGLE_MAPS_API_KEY}"
            )
            details = requests.get(details_url).json()
            if details.get("status") == "OK":
                address = details["result"].get("formatted_address", "")
                loc = res["geometry"]["location"]
                dist = haversine(user_lat, user_lng, loc["lat"], loc["lng"])
                if dist < best_dist:
                    best = details["result"]["name"] + "، " + address
                    best_dist = dist
        if best:
            return best
    return None

def reverse_places(lat: float, lng: float):
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lng}&radius=100&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    )
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        for res in data["results"]:
            place_id = res["place_id"]
            details_url = (
                "https://maps.googleapis.com/maps/api/place/details/json"
                f"?place_id={place_id}&language=ar&region=SA&key={GOOGLE_MAPS_API_KEY}"
            )
            details = requests.get(details_url).json()
            if details.get("status") == "OK":
                address = details["result"].get("formatted_address", "")
                # هنا ألغينا شرط الشرقية
                return details["result"]["name"] + "، " + address
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
    start_name = reverse_places(lat, lng) if lat and lng else None
    if not start_name:
        return "", "لا أستطيع تحديد موقعك. الرجاء إرسال الإحداثيات أولاً."
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
    }
    return sess_id, "مرحباً! إلى أين تريد الذهاب اليوم؟"

def proceed(session: Dict[str, Any], user_input: str) -> str:
    step = session["step"]

    if step == "ask_destination":
        dest = extract_destination(user_input)
        dest_name = places_search(dest, session["lat"], session["lng"])
        if not dest_name:
            return "تعذر تحديد موقع الوجهة. يرجى كتابة اسم أوضح أو أقرب حي/شارع."
        session["dest_name"] = dest_name
        session["step"] = "ask_start"
        return (
            f"هل تريد أن نأخذك من موقعك الحالي ({session['start_name']})"
            " أم تفضل الانطلاق من مكان آخر؟"
        )

    if step == "ask_start":
        txt = user_input.strip().lower()
        if txt in {"موقعي", "موقعي الحالي", "الموقع الحالي"}:
            pass  # استخدم start_name المحسوب تلقائياً
        else:
            start_name = places_search(user_input, session["lat"], session["lng"])
            if not start_name:
                return "تعذر تحديد موقع الانطلاق. يرجى كتابة اسم أوضح أو أقرب حي/شارع."
            session["start_name"] = start_name
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
