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

# ----- Google Maps API helpers -----

def geocode(address: str) -> Optional[Dict[str, float]]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
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
            results.append({
                "name": res.get("name"),
                "address": res.get("formatted_address"),
                "lat": loc["lat"],
                "lng": loc["lng"],
            })
        return results[:max_results]
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

# ----- FASTAPI models -----
class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

# ---- Assistant Scenario ----
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
            "step": "ask_destination",
            "history": [
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "assistant", "content": "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖"}
            ],
            "loc_txt": loc_txt,
            "possible_places": None,
            "chosen_place": None
        }
        return BotResponse(sessionId=sess_id, botMessage="مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖")

    sess = sessions[req.sessionId]
    user_msg = (req.userInput or "").strip()
    step = sess.get("step", "ask_destination")

    # ---- خطوة الوجهة (Places API و Geocoding) ----
    if step == "ask_destination":
        # جرب Places API أولاً
        places = places_search(user_msg, sess["lat"], sess["lng"])
        if not places:
            # fallback: جرب geocode لو ما لقى نتيجة (مثلاً حي أو اسم شارع)
            coords = geocode(user_msg)
            if coords:
                address = reverse_geocode(coords["lat"], coords["lng"]) or user_msg
                sess["step"] = "confirm_destination"
                sess["chosen_place"] = {"name": user_msg, "address": address, "lat": coords["lat"], "lng": coords["lng"]}
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage=f"هل تقصد الوجهة: {address}؟\nيرجى كتابة نعم أو لا.",
                    done=False
                )
            else:
                return BotResponse(sessionId=req.sessionId, botMessage="تعذر العثور على الوجهة. أكتب اسم أوضح أو أقرب حي/شارع.", done=False)
        if len(places) > 1:
            sess["step"] = "choose_destination"
            sess["possible_places"] = places
            options = "\n".join([f"{i+1}. {p['name']} - {p['address']}" for i, p in enumerate(places)])
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"وجدت أكثر من مكان بنفس الاسم:\n{options}\nيرجى اختيار رقم المكان الصحيح.",
                done=False
            )
        else:
            place = places[0]
            sess["chosen_place"] = place
            sess["step"] = "confirm_destination"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"هل تقصد الوجهة: {place['name']} - {place['address']}؟\nيرجى كتابة نعم أو لا.",
                done=False
            )

    # ---- اختيار المكان الصحيح من القائمة ----
    if step == "choose_destination":
        idx = -1
        try:
            idx = int(user_msg) - 1
        except:
            return BotResponse(sessionId=req.sessionId, botMessage="الرجاء إدخال رقم صحيح من القائمة.", done=False)
        places = sess.get("possible_places") or []
        if idx < 0 or idx >= len(places):
            return BotResponse(sessionId=req.sessionId, botMessage="الرجاء اختيار رقم من الخيارات أعلاه.", done=False)
        place = places[idx]
        sess["chosen_place"] = place
        sess["step"] = "confirm_destination"
        return BotResponse(
            sessionId=req.sessionId,
            botMessage=f"هل تقصد الوجهة: {place['name']} - {place['address']}؟\nيرجى كتابة نعم أو لا.",
            done=False
        )

    # ---- تأكيد الوجهة ----
    if step == "confirm_destination":
        if user_msg.strip().lower() in ["نعم", "أجل", "اكيد", "أيوه", "yes", "ok"]:
            place = sess["chosen_place"]
            # هنا تنتقل للخطوة التالية في الحجز أو تستكمل باقي السيناريو
            sess["step"] = "done"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"✔️ تم اختيار الوجهة: {place['name']} - {place['address']}.\n(هنا يكمل باقي سيناريو الحجز حسب الحاجة...)",
                done=True
            )
        else:
            sess["step"] = "ask_destination"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage="يرجى كتابة اسم الوجهة مرة أخرى.",
                done=False
            )

    return BotResponse(sessionId=req.sessionId, botMessage="حدث خطأ غير متوقع، حاول مجدداً.", done=False)
