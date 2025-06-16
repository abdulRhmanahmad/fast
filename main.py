import os
import uuid
import requests
import math
import random
import re
import difflib
import numpy as np
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import time
from datetime import datetime, timedelta
# ------------------------ PINECONE ------------------------
from pinecone import Pinecone, ServerlessSpec
TRIP_CREATE_API_URL = "https://car-booking-api-64ov.onrender.com/api/travel/request/create"
CAR_TYPES_API_URL = "https://car-booking-api-64ov.onrender.com/api/codeTables/priceCategories/all"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "places-index"

# إنشاء الفهرس لو مو موجود
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # لازم نفس أبعاد embedding
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )

pinecone_index = pc.Index(index_name)

# ------------------- OPENAI & FASTAPI ---------------------
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Endpoint GET /
@app.get("/")
def root():
    return {"status": "ok", "msg": "Taxi bot is running"}

# Endpoint HEAD / (لحل مشكلة 405 في فحص الصحة)
@app.head("/")
def root_head():
    return {}

sessions: Dict[str, Dict[str, Any]] = {}
places_cache = {}

# -------------- الأماكن المعرفة محلياً -------------
# لو عندك أماكن كثيرة، استوردها من ملف seed_places.py
known_places_embedding = {
    "الجورة": "الجورة، دمشق، سوريا",
    # ... باقي الأماكن كلها كما عندك
    "سوق الجمرك": "سوق الجمرك، دمشق، سوريا"
}

# ----------- توليد embeddings للأماكن المحلية فقط ------------
known_place_vectors = {
    name: client.embeddings.create(model="text-embedding-3-small", input=[name]).data[0].embedding
    for name in known_places_embedding
}

# --------- وظيفة إضافة أماكن إلى Pinecone (تشغيلها مره واحدة فقط إذا حبيت تعتمد البحث السريع) ---------
def seed_places_to_pinecone():
    for name, address in known_places_embedding.items():
        emb = client.embeddings.create(model="text-embedding-3-small", input=[name]).data[0].embedding
        pinecone_index.upsert(vectors=[(
            str(uuid.uuid4()),
            emb,
            {"name": name, "address": address}
        )])

# ============= Helpers & Core Functions =================
def calculate_estimated_price(distance_km, car_type_id):
    car_types = get_cached_car_types()
    car_type = next((c for c in car_types if str(c["Id"]) == str(car_type_id)), None)
    if not car_type:
        return 0
    min_price = float(car_type.get("Min_Price", 0))
    # جدول الأسعار التفصيلي
    for price_cat in car_type.get("A_Price_Catg", []):
        if price_cat["From_Dis"] <= distance_km < price_cat["To_Dis"]:
            return max(min_price, float(price_cat["Price"]) * distance_km)
    # لو مافي رينج مناسب استعمل الحد الأدنى
    return min_price

def get_embedding(text: str) -> list:
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return response.data[0].embedding

def cosine_similarity(vec1: list, vec2: list) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def geocode(address: str) -> Optional[Dict[str, float]]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&region=SY&language=ar&components=locality:دمشق&key={GOOGLE_MAPS_API_KEY}"
    data = requests.get(url).json()
    if data["status"] == "OK" and data["results"]:
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
    return None

def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&region=SY&language=ar&components=locality:دمشق&key={GOOGLE_MAPS_API_KEY}"
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

def remove_country(text):
    if not text:
        return ""
    return re.sub(r"(،?\s*سوريا)$", "", text.strip())

def get_location_text(lat, lng):
    address = reverse_geocode(lat, lng)
    if not address:
        return "موقعك الحالي"
    return format_address(address)

def clean_arabic_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = [
        'من', 'إلى', 'في', 'على', 'عند', 'بدي', 'أريد', 'أروح',
        'أذهب', 'بدك', 'تريد', 'تروح', 'تذهب', 'الى', 'انا', 'أنا'
    ]
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def expand_location_query(query: str) -> List[str]:
    query = clean_arabic_text(query)
    expanded_queries = [query]
    if query:
        if "شارع" not in query and "طريق" not in query:
            expanded_queries.append(f"شارع {query}")
            expanded_queries.append(f"{query} شارع")
        expanded_queries.append(f"{query} دمشق")
        expanded_queries.append(f"{query}, دمشق")
    return list(set(expanded_queries))

def get_distance_km(origin: str, destination: str) -> float:
    def get_latlng(address):
        geo = geocode(address)
        if geo:
            return f"{geo['lat']},{geo['lng']}"
        return address  # fallback
    origin_latlng = get_latlng(origin)
    destination_latlng = get_latlng(destination)
    url = (
        "https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin_latlng}"
        f"&destination={destination_latlng}"
        f"&mode=driving"
        f"&region=SY"
        f"&language=ar"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )
    resp = requests.get(url)
    data = resp.json()
    if data.get("status") == "OK" and data.get("routes"):
        leg = data["routes"][0]["legs"][0]
        distance_m = leg["distance"]["value"]
        return round(distance_m / 1000, 2)
    return 0.0

# --------- بحث Pinecone: استخدمه بدل/مع smart_places_search حسب رغبتك -----------
def search_places_with_pinecone(query):
    emb = get_embedding(query)
    results = pinecone_index.query(
        vector=emb,
        top_k=3,
        include_metadata=True
    )
    if results and results.matches:
        matches = []
        for match in results.matches:
            mdata = match['metadata']
            matches.append({
                "description": mdata["address"],
                "place_id": f"pinecone_{mdata['name']}",
                "is_pinecone": True,
            })
        return matches
    return []

def smart_places_search(query: str, user_lat: float, user_lng: float, max_results=5) -> list:
    cache_key = f"{query.lower().strip()}"
    if cache_key in places_cache:
        return places_cache[cache_key]
    # جرب البحث في Pinecone أولاً
    pinecone_results = search_places_with_pinecone(query)
    if pinecone_results:
        places_cache[cache_key] = pinecone_results
        return pinecone_results
    # باقي البحث المحلي أو Google Places API
    expanded_queries = expand_location_query(query)
    all_results = []
    for search_query in expanded_queries:
        results = places_autocomplete(search_query, user_lat, user_lng, max_results)
        all_results.extend(results)
        if len(results) >= 3:
            break
    unique_results = []
    seen_ids = set()
    for result in all_results:
        if result['place_id'] not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result['place_id'])
    if not unique_results:
        # بحث embedding محلي
        query_emb = get_embedding(query)
        best_match = None
        best_score = 0.0
        for name, emb in known_place_vectors.items():
            score = cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_match = name
        if best_score > 0.75:
            unique_results = [{
                "description": known_places_embedding[best_match],
                "place_id": f"embed_{best_match}",
                "is_local": True
            }]
    places_cache[cache_key] = unique_results
    return unique_results[:max_results]

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
        for e in data["predictions"][:max_results * 2]:  # جلب أكثر من المطلوب لأنك راح تصفي بعدين
            results.append({
                "description": e.get("description"),
                "place_id": e.get("place_id"),
            })
    # هنا أضف الفلترة
    filtered_results = [r for r in results if "دمشق" in r["description"]]
    return filtered_results[:max_results]

# ---------- جلب أنواع السيارات مع كاش ----------
def fetch_car_types():
    try:
        resp = requests.get(CAR_TYPES_API_URL, timeout=5)
        data = resp.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
    except Exception as e:
        print("خطأ في جلب أنواع السيارات:", e)
    return []

car_types_cache = {
    "data": [],
    "timestamp": 0
}
def get_cached_car_types():
    now = time.time()
    if not car_types_cache["data"] or now - car_types_cache["timestamp"] > 600:
        car_types_cache["data"] = fetch_car_types()
        car_types_cache["timestamp"] = now
    return car_types_cache["data"]

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

def get_place_details_enhanced(place_id: str) -> dict:
    if place_id.startswith("pinecone_"):
        name = place_id.replace("pinecone_", "")
        return {
            "address": known_places_embedding.get(name, f"{name}، دمشق، سوريا"),
            "lat": 33.5138,
            "lng": 36.2765,
        }
    if place_id.startswith("embed_") or place_id.startswith("local_"):
        location_name = place_id.replace("local_", "").replace("embed_", "")
        return {
            "address": known_places_embedding.get(location_name, f"{location_name}، دمشق، سوريا"),
            "lat": 33.5138,
            "lng": 36.2765,
        }
    else:
        return get_place_details(place_id)

def ask_gpt(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "أجب بشكل ودود ومختصر دائماً."},
            {"role": "user", "content": message}
        ],
        max_tokens=60,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def extract_time_from_text(user_msg):
    m = re.search(r'بعد\s+(\d+)\s+دق(ي|ا)ق', user_msg)
    if m:
        mins = int(m.group(1))
        return (datetime.now() + timedelta(minutes=mins)).strftime("%H:%M")
    return user_msg

# ================ API MODELS =================
class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

# ================ Messages =================
step_messages = {
    "ask_destination": [
        "أهلين وسهلين فيك 😊! وين بتحب نوصلك اليوم؟ 🚕",
        "يا هلا! خبرني لوين رايح اليوم، وأرتبلك كل شي 👍",
        "جاهز لمشوار جديد؟! قولي وين وجهتك اليوم 😊",
        "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير 🚖.\nوين حابب تروح اليوم؟",
        "هلا فيك! حددلي وجهتك لو سمحت.",
        "أهلين، شو عنوان المكان الي رايح عليه؟",
        "يسعد مساك! خبرني وين وجهتك اليوم.",
        "وين بدك أوصلك اليوم؟"
    ],
    "not_found": [
        "ما قدرت لاقي العنوان 😅، ممكن تعطيني اسم أوضح أو تختار من الأماكن المقترحة تحت 👇",
        "العنوان مو واضح، بتحب تعطيني اسم شارع أو منطقة؟ أو جرب تكتب بطريقة تانية ✍️"
    ],
    "choose_previous": [
        "تحب أرجعك لمكانك السابق؟\n",
        "هاي قائمة بأماكنك السابقة، بدك تروح لواحد منهم؟\n"
    ],
    "ask_pickup": [
        "من وين نوصلك؟ من موقعك الحالي ولا في نقطة ثانية؟ 🗺️",
        "حابب أجيك ععنوانك الحالي ولا حابب تغير؟",
        "اختر نقطة الانطلاق: موقعك الحالي أو مكان آخر.",
        "حددلي من وين حابب تبدأ الرحلة."
    ],
    "ask_time": [
        "وقت الرحلة متى تفضّل؟ الآن ولا بتوقيت محدد؟ ⏱",
        "تحب ننطلق فوراً ولا تحدد وقت لاحق؟",
        "خبرني متى الوقت المناسب للانطلاق."
    ],
    "ask_car_type": [
        "أي نوع سيارة بدك؟ عادية ولا VIP؟ 🚗",
        "تفضّل سيارة عادية ولا بدك تجربة فاخرة (VIP)؟",
        "خبرني نوع السيارة: عادية أم VIP؟"
    ],
    "ask_audio": [
        "تحب نسمع شي أثناء الرحلة؟ قرآن، موسيقى، أو تفضّل الصمت؟ 🎵",
        "اختر نوع الصوت: قرآن، موسيقى، أم بلا صوت.",
        "حابب نضيف لمسة موسيقية أو تحب الجو هادي؟"
    ],
    "confirm_booking": [
        "راجع ملخص الطلب وأكد إذا كل شي تمام 👇",
        "هذي تفاصيل رحلتك! إذا في شي مو واضح صححلي، أو أكد الحجز.",
        "قبل نأكد الحجز، شوف التفاصيل بالأسفل."
    ],
    "wait_loader": [
        "لحظة صغيرة، عم دورلك عالعنوان 😊 ...",
        "ثواني وعم لاقيلك المكان الأنسب 😄 ..."
    ]
}

def random_step_message(step):
    return random.choice(step_messages.get(step, ["كيف أقدر أخدمك؟"]))

# ============= دورة حياة البوت والتعامل مع الخطوات ==============

def is_out_of_booking_context(user_msg, step):
    general_words = [
        "كيفك", "شلونك", "السلام عليكم", "مرحبا", "هاي", "من أنت", "مين أنت",
        "شو بتسوي", "شو في", "كيف الجو", "شو أخبارك", "شخبارك", "وينك", "شكرا", "يسلمو",
        "ثانكس", "thanks", "thx", "good", "nice", "help", "مساعدة"
    ]
    msg = user_msg.strip().lower()
    if any(word in msg for word in general_words):
        return True
    if step in ["ask_destination", "ask_pickup"] and len(msg) < 5:
        return True
    return False

def current_step_question(sess):
    step = sess.get('step', '')
    if step in step_messages:
        return random_step_message(step)
    return "كيف أقدر أخدمك؟"

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

# ============= FastAPI Endpoint ==============

@app.post("/chatbot", response_model=BotResponse)
def chatbot(req: UserRequest):
    try:
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
                    {"role": "assistant", "content": random_step_message("ask_destination")}
                ],
                "loc_txt": loc_txt,
                "last_places": [],
                "possible_places": None,
                "chosen_place": None,
                "possible_pickup_places": None,
                "pickup": None,
                "time": None,
                "car": None,
                "audio": None
            }
            # عرض الأماكن السابقة
            last_places = sessions[sess_id]["last_places"]
            if last_places:
                prev_msg = random_step_message("choose_previous")
                for i, p in enumerate(last_places):
                    prev_msg += f"{i+1}. {remove_country(p)}\n"
                prev_msg += "أكتب رقم المكان أو اسم جديد."
                return BotResponse(sessionId=sess_id, botMessage=prev_msg)
            else:
                return BotResponse(sessionId=sess_id, botMessage=random_step_message("ask_destination"))

        sess = sessions[req.sessionId]
        user_msg = (req.userInput or "").strip()
        step = sess.get("step", "ask_destination")
        last_places = sess.get("last_places", [])

        # ========== إلغاء أو إعادة تشغيل ==========
        if user_msg.lower() in ["إلغاء", "إلغاء الحجز", "ابدأ من جديد", "restart", "cancel"]:
            del sessions[req.sessionId]
            return BotResponse(sessionId="", botMessage="ولا يهمك! إذا حابب تبدأ حجز جديد خبرني وين بتروح 😊", done=True)

        # ========== كلام خارج السياق ==========
        if is_out_of_booking_context(user_msg, step):
            gpt_reply = ask_gpt(user_msg)
            step_q = current_step_question(sess)
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"{gpt_reply}\n\n{step_q}",
                done=False
            )

        # ========== خطوة الوجهة + أماكن سابقة ==========
        if step == "ask_destination":
            if user_msg.isdigit() and last_places:
                idx = int(user_msg) - 1
                if 0 <= idx < len(last_places):
                    place_info = get_place_details_enhanced(f"embed_{last_places[idx].split('،')[0]}")
                    sess["chosen_place"] = place_info
                    
                    sess["step"] = "ask_pickup"
                    return BotResponse(sessionId=req.sessionId, botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])} 🚕\n{random_step_message('ask_pickup')}", done=False)

            match_prev = difflib.get_close_matches(user_msg, [p.split("،")[0] for p in last_places], n=1, cutoff=0.8)
            if match_prev:
                return BotResponse(sessionId=req.sessionId, botMessage=f"على فكرة، هذا نفس المكان يلي رحت عليه قبل: {match_prev[0]} 😉\nأكيد تتابع ولا تكتب عنوان آخر؟", done=False)

            places = smart_places_search(user_msg, sess["lat"], sess["lng"])
            if not places:
                typo_msg = difflib.get_close_matches(user_msg, known_places_embedding.keys(), n=1, cutoff=0.6)
                if typo_msg:
                    return BotResponse(sessionId=req.sessionId, botMessage=f"يمكن قصدك: {typo_msg[0]}؟ أكتب 'نعم' للتأكيد أو جرب تكتب عنوان تاني. 😊", done=False)
                return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("not_found")[0], done=False)
            if len(places) > 1:
                sess["step"] = "choose_destination"
                sess["possible_places"] = places
                options = "\n".join([f"{i+1}. {remove_country(p['description'])}" for i, p in enumerate(places)])
                return BotResponse(sessionId=req.sessionId, botMessage=f"لقيت أكتر من مكان يشبه طلبك 👇\n{options}\nاختر رقم أو اسم المكان المطلوب.", done=False)
            else:
                place_info = get_place_details_enhanced(places[0]['place_id'])
                sess["chosen_place"] = place_info
                sess["to_lat"] = place_info.get("lat", 0)
                sess["to_lng"] = place_info.get("lng", 0)
                sess["step"] = "ask_pickup"
                return BotResponse(sessionId=req.sessionId, botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])} 🚕\n{random_step_message('ask_pickup')}", done=False)

        # ========== اختيار من قائمة ==========
        if step == "choose_destination":
            places = sess.get("possible_places", [])
            if user_msg.isdigit():
                idx = int(user_msg) - 1
                if 0 <= idx < len(places):
                    place_info = get_place_details_enhanced(places[idx]['place_id'])
                    sess["chosen_place"] = place_info
                    sess["to_lat"] = place_info.get("lat", 0)
                    sess["to_lng"] = place_info.get("lng", 0)
                    sess["step"] = "ask_pickup"
                    return BotResponse(sessionId=req.sessionId, botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])} 🚕\n{random_step_message('ask_pickup')}", done=False)
            typo_msg = difflib.get_close_matches(user_msg, [p['description'].split("،")[0] for p in places], n=1, cutoff=0.6)
            if typo_msg:
                
                return BotResponse(sessionId=req.sessionId, botMessage=f"يمكن قصدك: {typo_msg[0]}؟ أكتب 'نعم' للتأكيد أو جرب تكتب عنوان تاني. 😊", done=False)
            return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("not_found")[0], done=False)

        # ========== نقطة الانطلاق ==========
        if step == "ask_pickup":
            if user_msg in ["موقعي", "موقعي", "موقعي الحالي", "الموقع الحالي"]:
                sess["pickup"] = sess["loc_txt"]
                sess["pickup_lat"] = sess["lat"]
                sess["pickup_lng"] = sess["lng"]
                sess["step"] = "ask_time"

            
                return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)
            else:
                places = smart_places_search(user_msg, sess["lat"], sess["lng"])
                if not places:
                    return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("not_found")[1], done=False)
                if len(places) > 1:
                    sess["step"] = "choose_pickup"
                    sess["possible_pickup_places"] = places
                    options = "\n".join([f"{i+1}. {remove_country(p['description'])}" for i, p in enumerate(places)])
                    return BotResponse(sessionId=req.sessionId, botMessage=f"لقيت أكتر من مكان كنقطة انطلاق 👇\n{options}\nاختر رقم أو اسم المكان.", done=False)
                else:
                    place_info = get_place_details_enhanced(places[0]['place_id'])
                    sess["pickup"] = place_info['address']
                    
                    sess["pickup_lat"] = place_info.get("lat", 0)
                    sess["pickup_lng"] = place_info.get("lng", 0)

                    sess["step"] = "ask_time"
                    return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)

        # ========== اختيار نقطة الانطلاق ==========
        if step == "choose_pickup":
            places = sess.get("possible_pickup_places", [])
            if user_msg.isdigit():
                idx = int(user_msg) - 1
                if 0 <= idx < len(places):
                    place_info = get_place_details_enhanced(places[idx]['place_id'])
                    sess["pickup"] = place_info['address']
                    sess["pickup_lat"] = place_info.get("lat", 0)
                    sess["pickup_lng"] = place_info.get("lng", 0)
                    sess["step"] = "ask_time"
                    return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)
            typo_msg = difflib.get_close_matches(user_msg, [p['description'].split("،")[0] for p in places], n=1, cutoff=0.6)
            if typo_msg:
                return BotResponse(sessionId=req.sessionId, botMessage=f"يمكن قصدك: {typo_msg[0]}؟ أكتب 'نعم' للتأكيد أو جرب تكتب عنوان تاني. 😊", done=False)
            return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("not_found")[1], done=False)

        # ========== وقت الرحلة ==========
        if step == "ask_time":
            parsed_time = extract_time_from_text(user_msg)
            sess["time"] = parsed_time
            sess["step"] = "ask_car_type"
            return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_car_type"), done=False)

        # ========== نوع السيارة ==========
        if step == "ask_car_type":
            car_types = get_cached_car_types()
            if not car_types:
                sess["car"] = "عادية"
                sess["step"] = "ask_audio"
                return BotResponse(sessionId=req.sessionId, botMessage="ما قدرت أجيب أنواع السيارات حالياً. نكمل بسيارة عادية.", done=False)

           
            options = "\n".join([f"{i+1}. {ct.get('Ar_Name', 'نوع غير معروف')}" for i, ct in enumerate(car_types)])

            sess["car_types"] = car_types
            sess["step"] = "choose_car_type"
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"اختر نوع السيارة اللي يناسبك:\n{options}\n(أرسل رقم النوع)",
                done=False
            )
        if step == "choose_car_type":
            car_types = sess.get("car_types", [])
            if user_msg.isdigit():
                idx = int(user_msg) - 1
                if 0 <= idx < len(car_types):
                    

                    sess["car"] = car_types[idx].get("Ar_Name", "غير معروف")
                    sess["car_id"] = car_types[idx].get("Id")

                    sess["step"] = "ask_audio"
                    return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_audio"), done=False)
            return BotResponse(sessionId=req.sessionId, botMessage="يرجى اختيار رقم من القائمة أعلاه.", done=False)

        if step == "ask_audio":
            # تحديد الصوت
            if "قرآن" in user_msg or "قران" in user_msg:
                sess["audio"] = "قرآن"
            elif "موسيقى" in user_msg or "موسيقا" in user_msg or "أغاني" in user_msg:
                sess["audio"] = "موسيقى"
            else:
                sess["audio"] = "صمت"
            sess["step"] = "confirm_booking"

            pickup_address = sess['pickup']
            dest_address = sess['chosen_place']['address']
            distance_km = get_distance_km(pickup_address, dest_address)
            sess['distance_km'] = distance_km  # (اختياري)
            car_id = sess.get('car_id', 1)
            estimated_price = calculate_estimated_price(distance_km, car_id)
            sess['estimated_price'] = estimated_price

            summary = f"""
ملخص طلبك:
- من: {remove_country(pickup_address)}
- إلى: {remove_country(dest_address)}
- المسافة التقريبية: {distance_km if distance_km else "غير متوفرة"} كم
- نوع السيارة: {sess.get('car')}
- السعر التقديري: {int(estimated_price)} ل.س
هل ترغب بتأكيد الحجز؟
"""

    
            return BotResponse(sessionId=req.sessionId, botMessage=summary, done=False)
        
        # ========== التأكيد ==========
        if step == "confirm_booking":
            if user_msg in ["نعم", "موافق", "أكد", "تأكيد", "yes", "ok"]:
                pickup_address = sess['pickup']
                dest_address = sess['chosen_place']['address']
                distance_km = sess.get('distance_km', 0)
                estimated_price = sess.get('estimated_price', 0)
                car_id = sess.get('car_id', 1)
                estimated_duration = int(distance_km * 4)  # أو احسبها من API المسافة والزمن
                estimated_distance = int(distance_km * 1000)

                from_lat = sess.get('pickup_lat', 0)      # إذا تقدر خزّنها من قبل
                from_lng = sess.get('pickup_lng', 0)
                to_lat = sess.get('to_lat', 0)
                to_lng = sess.get('to_lng', 0)
                payload = {
                    "From_Location": remove_country(pickup_address),
                    "To_Location": remove_country(dest_address),
                    "From_Lat": from_lat,
                    "From_Lng": from_lng,
                    "To_Lat": to_lat,
                    "To_Lng": to_lng,
                    "Catg_Id": int(car_id),
                    "Pref_Music": sess.get("audio", ""),
                    "Estimated_Price": float(estimated_price),
                    "Estimated_Duration": estimated_duration,
                    "Estimated_Distance": estimated_distance,
                    "Start_at": None,
                    "Type_Id": 4,
                    "Rem": "حجز من الشات بوت"
                }
                try:
                    api_response = requests.post(TRIP_CREATE_API_URL, json=payload, timeout=10)
                    resp_json = api_response.json()
                except Exception as e:
                    resp_json = {"error": str(e)}
                msg = f"""
🎉 تم تأكيد حجزك بنجاح!
🚗 السائق في الطريق إليك!
⏱️ الوقت المتوقع: 5-10 دقائق

بيانات الرحلة (API): {resp_json}

لو بدك حجز جديد خبرني وين بتروح 😉
"""
                del sessions[req.sessionId]
                return BotResponse(sessionId=req.sessionId, botMessage=msg, done=True)
            else:
                 return BotResponse(sessionId=req.sessionId, botMessage="تم إلغاء الحجز. إذا حابب تبدأ من جديد خبرني 😊", done=True)
    except Exception as e:
        return BotResponse(
        sessionId = getattr(req, "sessionId", ""),
        botMessage = f"⚠️ حصل خطأ غير متوقع أثناء معالجة الطلب: {str(e)}",
        done = True
    )


        

# ========== تشغيل السيرفر محلياً لو أردت ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
