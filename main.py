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

def places_search(query: str, user_lat: float, user_lng: float, max_results=5) -> Optional[List[Dict[str, Any]]]:
    url = (
        "https://maps.googleapis.com/maps/api/place/textsearch/json"
        f"?query={query}&location={user_lat},{user_lng}&radius=30000"
        f"&region=SY&language=ar&key={GOOGLE_MAPS_API_KEY}"
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
                "distance": round(dist, 2)
            })
        results.sort(key=lambda x: x['distance'])
        return results[:max_results]
    return None

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
            "pickup": None,
            "time": None,
            "car": None,
            "audio": None
        }
        return BotResponse(sessionId=sess_id, botMessage="مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖")

    sess = sessions[req.sessionId]
    user_msg = (req.userInput or "").strip()
    step = sess.get("step", "ask_destination")

    # -------- الخطوة 1: طلب الوجهة --------
    if step == "ask_destination":
        places = places_search(user_msg, sess["lat"], sess["lng"])
        if not places:
            coords = geocode(user_msg)
            if coords:
                address = reverse_geocode(coords["lat"], coords["lng"]) or user_msg
                sess["chosen_place"] = {"name": user_msg, "address": address, "lat": coords["lat"], "lng": coords["lng"], "distance": haversine(sess["lat"], sess["lng"], coords["lat"], coords["lng"])}
                sess["step"] = "confirm_destination"
                return BotResponse(sessionId=req.sessionId, botMessage=f"تم تحديد الوجهة: {address}\nهل تريد المتابعة؟", done=False)
            else:
                return BotResponse(sessionId=req.sessionId, botMessage="لم أستطع تحديد الموقع، حاول كتابة اسم أوضح.", done=False)
        if len(places) > 1:
            sess["step"] = "choose_destination"
            sess["possible_places"] = places
            options = "\n".join([f"{i+1}. {p['name']} - {p['address']} (يبعد {p['distance']} كم)" for i, p in enumerate(places)])
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"وجدت أكثر من مكان:\n{options}\nيرجى اختيار رقم المكان الصحيح.",
                done=False
            )
        else:
            sess["chosen_place"] = places[0]
            sess["step"] = "confirm_destination"
            return BotResponse(sessionId=req.sessionId, botMessage=f"تم تحديد الوجهة: {places[0]['name']} - {places[0]['address']}\nهل تريد المتابعة؟", done=False)

    # -------- الخطوة 2: اختيار الوجهة من القائمة --------
    if step == "choose_destination":
        idx = -1
        try:
            idx = int(user_msg) - 1
        except:
            return BotResponse(sessionId=req.sessionId, botMessage="يرجى اختيار رقم صحيح.", done=False)
        places = sess.get("possible_places", [])
        if 0 <= idx < len(places):
            sess["chosen_place"] = places[idx]
            sess["step"] = "confirm_destination"
            return BotResponse(sessionId=req.sessionId, botMessage=f"تم اختيار: {places[idx]['name']} - {places[idx]['address']}\nهل تريد المتابعة؟", done=False)
        else:
            return BotResponse(sessionId=req.sessionId, botMessage="يرجى اختيار رقم صحيح.", done=False)

    # -------- الخطوة 3: تأكيد الوجهة --------
    if step == "confirm_destination":
        if user_msg.strip().lower() in ["نعم", "أجل", "اكيد", "أيوه", "yes", "ok"]:
            sess["step"] = "ask_pickup"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"✔️ تم اختيار الوجهة: {sess['chosen_place']['name']} - {sess['chosen_place']['address']}.\nمن أين تود الانطلاق؟ من موقعك الحالي ({sess['loc_txt']}) أم من مكان آخر؟",
                done=False
            )
        else:
            sess["step"] = "ask_destination"
            return BotResponse(sessionId=req.sessionId, botMessage="يرجى كتابة اسم الوجهة مرة أخرى.", done=False)

    # -------- الخطوة 4: نقطة الانطلاق --------
    if step == "ask_pickup":
        if user_msg.lower() in ["موقعي", "موقعي الحالي", "الموقع الحالي"]:
            sess["pickup"] = sess["loc_txt"]
        else:
            coords = geocode(user_msg)
            if coords:
                pickup_addr = reverse_geocode(coords["lat"], coords["lng"]) or user_msg
                sess["pickup"] = pickup_addr
            else:
                return BotResponse(sessionId=req.sessionId, botMessage="تعذر تحديد نقطة الانطلاق. أكتب اسم أوضح أو أقرب شارع.", done=False)
        sess["step"] = "ask_time"
        return BotResponse(sessionId=req.sessionId, botMessage="متى تود الانطلاق؟ الآن أم في وقت محدد؟", done=False)

    # -------- الخطوة 5: وقت الانطلاق --------
    if step == "ask_time":
        sess["time"] = user_msg
        sess["step"] = "ask_car"
        return BotResponse(sessionId=req.sessionId, botMessage="أي نوع سيارة تفضل؟ عادية أم VIP؟", done=False)

    # -------- الخطوة 6: نوع السيارة --------
    if step == "ask_car":
        sess["car"] = user_msg
        sess["step"] = "ask_audio"
        return BotResponse(sessionId=req.sessionId, botMessage="هل تود الاستماع لشيء أثناء الرحلة؟ (قرآن، موسيقى، صمت)", done=False)

    # -------- الخطوة 7: الصوت --------
    if step == "ask_audio":
        sess["audio"] = user_msg
        sess["step"] = "summary"
        return BotResponse(
            sessionId=req.sessionId,
            botMessage=(
                "ملخص رحلتك:\n"
                f"• الوجهة: {sess['chosen_place']['name']} - {sess['chosen_place']['address']}\n"
                f"• الانطلاق: {sess['pickup']}\n"
                f"• الوقت: {sess['time']}\n"
                f"• نوع السيارة: {sess['car']}\n"
                f"• الصوت: {sess['audio']}\n\n"
                "هل تؤكد الحجز؟"
            ),
            done=False
        )

    # -------- الخطوة 8: التأكيد النهائي (هنا الحجز الفعلي) --------
    if step == "summary":
        if user_msg.lower() in ["نعم", "أجل", "أكيد", "موافق", "yes"]:
            booking_id = create_mock_booking(
                sess["pickup"],
                f"{sess['chosen_place']['name']} - {sess['chosen_place']['address']}",
                sess["time"],
                sess["car"],
                sess["audio"],
                user_id=None
            )
            sess["step"] = "confirmed"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"✔️ تم تأكيد حجزك بنجاح! رقم الطلب: {booking_id}\nسيصلك السائق في الوقت المحدد.",
                done=True
            )
        else:
            sess["step"] = "ask_destination"
            return BotResponse(sessionId=req.sessionId, botMessage="تم إلغاء الحجز. اكتب الوجهة من جديد.", done=True)

    # -------- الخطوة 9: انتهاء الجلسة --------
    if step == "confirmed":
        return BotResponse(sessionId=req.sessionId, botMessage="بإمكانك بدء حجز جديد بكتابة الوجهة.", done=True)

    # -------- fallback --------
    return BotResponse(sessionId=req.sessionId, botMessage="حدث خطأ غير متوقع، حاول مجدداً.", done=False)
