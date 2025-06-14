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

# ------------------------ PINECONE ------------------------
from pinecone import Pinecone, ServerlessSpec

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
sessions: Dict[str, Dict[str, Any]] = {}

# -------------- الأماكن المعرفة محلياً -------------
# لو عندك أماكن كثيرة، استوردها من ملف seed_places.py
known_places_embedding = {
"الجورة": "الجورة، دمشق، سوريا",
    "العمارة الجوانية": "العمارة الجوانية، دمشق، سوريا",
    "باب توما": "باب توما، دمشق، سوريا",
    "القيمرية": "القيمرية، دمشق، سوريا",
    "الحميدية": "الحميدية، دمشق، سوريا",
    "الحريقة": "الحريقة، دمشق، سوريا",
    "الأمين": "الأمين، دمشق، سوريا",
    "مئذنة الشحم": "مئذنة الشحم، دمشق، سوريا",
    "شاغور جواني": "شاغور جواني، دمشق، سوريا",
    "سوق ساروجة": "سوق ساروجة، دمشق، سوريا",
    "العقيبة": "العقيبة، دمشق، سوريا",
    "العمارة البرانية": "العمارة البرانية، دمشق، سوريا",
    "مسجد الأقصاب": "مسجد الأقصاب، دمشق، سوريا",
    "القصاع": "القصاع، دمشق، سوريا",
    "العدوي": "العدوي، دمشق، سوريا",
    "القصور": "القصور، دمشق، سوريا",
    "فارس الخوري": "فارس الخوري، دمشق، سوريا",
    "القنوات": "القنوات، دمشق، سوريا",
    "الحجاز": "الحجاز، دمشق، سوريا",
    "البرامكة": "البرامكة، دمشق، سوريا",
    "باب الجابية": "باب الجابية، دمشق، سوريا",
    "باب سريجة": "باب سريجة، دمشق، سوريا",
    "السويقة": "السويقة، دمشق، سوريا",
    "قبر عاتكة": "قبر عاتكة، دمشق، سوريا",
    "المجتهد": "المجتهد، دمشق، سوريا",
    "الأنصاري": "الأنصاري، دمشق، سوريا",
    "جوبر الشرقي": "جوبر الشرقي، دمشق، سوريا",
    "جوبر الغربي": "جوبر الغربي، دمشق، سوريا",
    "المأمونية": "المأمونية، دمشق، سوريا",
    "الاستقلال": "الاستقلال، دمشق، سوريا",
    "ميدان وسطاني": "ميدان وسطاني، دمشق، سوريا",
    "الزاهرة": "الزاهرة، دمشق، سوريا",
    "الحقلة": "الحقلة، دمشق، سوريا",
    "الدقاق": "الدقاق، دمشق، سوريا",
    "القاعة": "القاعة، دمشق، سوريا",
    "باب مصلى": "باب مصلى، دمشق، سوريا",
    "شاغور براني": "شاغور براني، دمشق، سوريا",
    "باب شرقي": "باب شرقي، دمشق، سوريا",
    "ابن عساكر": "ابن عساكر، دمشق، سوريا",
    "النضال": "النضال، دمشق، سوريا",
    "الوحدة": "الوحدة، دمشق، سوريا",
    "بلال": "بلال، دمشق، سوريا",
    "روضة الميدان": "روضة الميدان، دمشق، سوريا",
    "الزهور": "الزهور، دمشق، سوريا",
    "التضامن": "التضامن، دمشق، سوريا",
    "السيدة عائشة": "السيدة عائشة، دمشق، سوريا",
    "القدم": "القدم، دمشق، سوريا",
    "المصطفى": "المصطفى، دمشق، سوريا",
    "الشريباتي": "الشريباتي، دمشق، سوريا",
    "العسالي": "العسالي، دمشق، سوريا",
    "القدم الشرقي": "القدم الشرقي، دمشق، سوريا",
    "كفرسوسة البلد": "كفرسوسة البلد، دمشق، سوريا",
    "الإخلاص": "الإخلاص، دمشق، سوريا",
    "الواحة": "الواحة، دمشق، سوريا",
    "الفردوس": "الفردوس، دمشق، سوريا",
    "اللوان": "اللوان، دمشق، سوريا",
    "الربوة": "الربوة، دمشق، سوريا",
    "المزة القديمة": "المزة القديمة، دمشق، سوريا",
    "الجلاء": "الجلاء، دمشق، سوريا",
    "مزة جبل": "مزة جبل، دمشق، سوريا",
    "فيلات شرقية": "فيلات شرقية، دمشق، سوريا",
    "فيلات غربية": "فيلات غربية، دمشق، سوريا",
    "مزة 86": "مزة 86، دمشق، سوريا",
    "مزة بساتين": "مزة بساتين، دمشق، سوريا",
    "مشروع دمر": "مشروع دمر، دمشق، سوريا",
    "دمر الشرقية": "دمر الشرقية، دمشق، سوريا",
    "دمر الغربية": "دمر الغربية، دمشق، سوريا",
    "العرين": "العرين، دمشق، سوريا",
    "الورود": "الورود، دمشق، سوريا",
    "برزة البلد": "برزة البلد، دمشق، سوريا",
    "مساكن برزة": "مساكن برزة، دمشق، سوريا",
    "المنارة": "المنارة، دمشق، سوريا",
    "العباس": "العباس، دمشق، سوريا",
    "النزهة": "النزهة، دمشق، سوريا",
    "عش الورور": "عش الورور، دمشق، سوريا",
    "تشرين": "تشرين، دمشق، سوريا",
    "القابون": "القابون، دمشق، سوريا",
    "المصانع": "المصانع، دمشق، سوريا",
    "أسد الدين": "أسد الدين، دمشق، سوريا",
    "النقشبندي": "النقشبندي، دمشق، سوريا",
    "الأيوبية": "الأيوبية، دمشق، سوريا",
    "الفيحاء": "الفيحاء، دمشق، سوريا",
    "قاسيون": "قاسيون، دمشق، سوريا",
    "أبو جرش": "أبو جرش، دمشق، سوريا",
    "الشيخ محي الدين": "الشيخ محي الدين، دمشق، سوريا",
    "المدارس": "المدارس، دمشق، سوريا",
    "المزرعة": "المزرعة، دمشق، سوريا",
    "الشهداء": "الشهداء، دمشق، سوريا",
    "شورى": "شورى، دمشق، سوريا",
    "المصطبة": "المصطبة، دمشق، سوريا",
    "المرابط": "المرابط، دمشق، سوريا",
    "الروضة": "الروضة، دمشق، سوريا",
    "أبو رمانة": "أبو رمانة، دمشق، سوريا",
    "المالكي": "المالكي، دمشق، سوريا",
    "الحبوبي": "الحبوبي، دمشق، سوريا",
    "الكرمل": "الكرمل، دمشق، سوريا",
    "شارع الثورة": "شارع الثورة، دمشق، سوريا",
    "شارع الحمراء": "شارع الحمراء، دمشق، سوريا",
    "شارع بغداد": "شارع بغداد، دمشق، سوريا",
    "شارع خالد بن الوليد": "شارع خالد بن الوليد، دمشق، سوريا",
    "شارع شكري القوتلي": "شارع شكري القوتلي، دمشق، سوريا",
    "شارع العابد": "شارع العابد، دمشق، سوريا",
    "شارع النصر": "شارع النصر، دمشق، سوريا",
    "شارع الصالحية": "شارع الصالحية، دمشق، سوريا",
    "شارع البدوي": "شارع البدوي، دمشق، سوريا",
    "سوق الحميدية": "سوق الحميدية، دمشق، سوريا",
    "سوق مدحت باشا": "سوق مدحت باشا، دمشق، سوريا",
    "سوق الحريقة": "سوق الحريقة، دمشق، سوريا",
    "سوق العصرونية": "سوق العصرونية، دمشق، سوريا",
    "سوق الصاغة": "سوق الصاغة، دمشق، سوريا",
    "سوق المناخلية": "سوق المناخلية، دمشق، سوريا",
    "سوق الخياطين": "سوق الخياطين، دمشق، سوريا",
    "سوق السروجية": "سوق السروجية، دمشق، سوريا",
    "سوق القباقبية": "سوق القباقبية، دمشق، سوريا",
    "سوق الهال": "سوق الهال، دمشق، سوريا",
    "سوق الجزماتية": "سوق الجزماتية، دمشق، سوريا",
    "سوق الجمعة": "سوق الجمعة، دمشق، سوريا",
    "سوق الدرويشية": "سوق الدرويشية، دمشق، سوريا",
    "سوق السنانية": "سوق السنانية، دمشق، سوريا",
    "سوق السويقة": "سوق السويقة، دمشق، سوريا",
    "سوق العتيق": "سوق العتيق، دمشق، سوريا",
    "سوق النحاسين": "سوق النحاسين، دمشق، سوريا",
    "سوق النحاتين": "سوق النحاتين، دمشق، سوريا",
    "سوق المهن اليدوية": "سوق المهن اليدوية، دمشق، سوريا",
    "سوق باب الجابية": "سوق باب الجابية، دمشق، سوريا",
    "سوق صاروجا": "سوق صاروجا، دمشق، سوريا",
    "سوق القبيبات": "سوق القبيبات، دمشق، سوريا",
    "سوق الخجا": "سوق الخجا، دمشق، سوريا",
    "سوق السكرية": "سوق السكرية، دمشق، سوريا",
    "سوق السنجقدار": "سوق السنجقدار، دمشق، سوريا",
    "سوق المسكية": "سوق المسكية، دمشق، سوريا",
    "سوق الصقالين": "سوق الصقالين، دمشق، سوريا",
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
# شغلها مرة واحدة فقط (أول ما تنشئ الفهرس)، ثم علقها أو احذفها
# seed_places_to_pinecone()

# ============= Helpers & Core Functions =================

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
    # جرب البحث في Pinecone أولاً
    pinecone_results = search_places_with_pinecone(query)
    if pinecone_results:
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
        for e in data["predictions"][:max_results]:
            results.append({
                "description": e.get("description"),
                "place_id": e.get("place_id"),
            })
        return results
    return []

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

# ============= دورة حياة البوت والتعامل مع الخطوات ==============

class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

step_messages = {
    "ask_destination": [
        "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير 🚖.\nوين حابب تروح اليوم؟",
        "هلا فيك! حددلي وجهتك لو سمحت.",
        "أهلين، شو عنوان المكان الي رايح عليه؟",
        "يسعد مساك! خبرني وين وجهتك اليوم.",
        "وين بدك أوصلك اليوم؟"
    ],
    "ask_pickup": [
        "من وين نوصلك؟ من موقعك الحالي ولا في نقطة ثانية؟",
        "اختر نقطة الانطلاق: موقعك الحالي أو مكان آخر.",
        "حابب أجيك ععنوانك الحالي ولا حابب تغير؟",
        "حددلي من وين حابب تبدأ الرحلة."
    ],
    "ask_time": [
        "وقت الرحلة متى تفضّل؟ الآن ولا بتوقيت محدد؟",
        "تحب ننطلق فوراً ولا تحدد وقت لاحق؟",
        "خبرني متى الوقت المناسب للانطلاق."
    ],
    "ask_car_type": [
        "أي نوع سيارة بدك؟ عادية ولا VIP؟",
        "تفضّل سيارة عادية ولا بدك تجربة فاخرة (VIP)؟",
        "خبرني نوع السيارة: عادية أم VIP؟"
    ],
    "ask_audio": [
        "تحب نسمع شي أثناء الرحلة؟ قرآن، موسيقى، أو تفضّل الصمت؟",
        "اختر نوع الصوت: قرآن، موسيقى، أم بلا صوت.",
        "حابب نضيف لمسة موسيقية أو تحب الجو هادي؟"
    ],
    "confirm_booking": [
        "راجع ملخص الطلب وأكد إذا كل شي تمام 👇",
        "هذي تفاصيل رحلتك! إذا في شي مو واضح صححلي، أو أكد الحجز.",
        "قبل نأكد الحجز، شوف التفاصيل بالأسفل."
    ]
}

def random_step_message(step):
    msgs = step_messages.get(step, ["كيف أقدر أخدمك؟"])
    return random.choice(msgs)

def ask_gpt(message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "أجب بشكل ودود ومختصر. إذا الموضوع خارج حجز التاكسي، جاوب بلطف ثم ذكر المستخدم أنه بإمكانه حجز مشوار."},
            {"role": "user", "content": message}
        ],
        max_tokens=60,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

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
                "possible_places": None,
                "chosen_place": None,
                "possible_pickup_places": None,
                "pickup": None,
                "time": None,
                "car": None,
                "audio": None
            }
            return BotResponse(sessionId=sess_id, botMessage=random_step_message("ask_destination"))

        sess = sessions[req.sessionId]
        user_msg = (req.userInput or "").strip()
        step = sess.get("step", "ask_destination")

        if is_out_of_booking_context(user_msg, step):
            gpt_reply = ask_gpt(user_msg)
            step_q = current_step_question(sess)
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"{gpt_reply}\n\n{step_q}",
                done=False
            )

        if step == "ask_destination":
            places = smart_places_search(user_msg, sess["lat"], sess["lng"])
            if not places:
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage="لم أتمكن من العثور على هذا المكان. جرب مكان آخر أو أعد كتابة العنوان.\nمثال: 'الشعلان'، 'المزة'، 'ساحة الأمويين'",
                    done=False
                )
            if len(places) > 1:
                sess["step"] = "choose_destination"
                sess["possible_places"] = places
                options = "\n".join([f"{i+1}. {remove_country(p['description'])}" for i, p in enumerate(places)])
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage=f"وجدت أكثر من مكان:\n{options}\nاختر رقم أو اسم المكان المطلوب.",
                    done=False
                )
            else:
                if places[0].get('is_local') or places[0].get('is_pinecone'):
                    place_info = get_place_details_enhanced(places[0]['place_id'])
                else:
                    place_info = get_place_details(places[0]['place_id'])
                sess["chosen_place"] = place_info
                sess["step"] = "ask_pickup"
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])}\n{random_step_message('ask_pickup')}",
                    done=False
                )

        if step == "choose_destination":
            places = sess.get("possible_places", [])
            user_reply = user_msg.strip().lower()
            found = False
            try:
                idx = int(user_reply) - 1
                if 0 <= idx < len(places):
                    if places[idx].get('is_local') or places[idx].get('is_pinecone'):
                        place_info = get_place_details_enhanced(places[idx]['place_id'])
                    else:
                        place_info = get_place_details(places[idx]['place_id'])
                    sess["chosen_place"] = place_info
                    sess["step"] = "ask_pickup"
                    found = True
            except:
                pass
            if not found:
                new_places = smart_places_search(user_msg, sess["lat"], sess["lng"])
                if new_places:
                    if len(new_places) == 1:
                        if new_places[0].get('is_local') or new_places[0].get('is_pinecone'):
                            place_info = get_place_details_enhanced(new_places[0]['place_id'])
                        else:
                            place_info = get_place_details(new_places[0]['place_id'])
                        sess["chosen_place"] = place_info
                        sess["step"] = "ask_pickup"
                        return BotResponse(
                            sessionId=req.sessionId,
                            botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])}\n{random_step_message('ask_pickup')}",
                            done=False
                        )
                    else:
                        sess["possible_places"] = new_places
                        options = "\n".join([f"{i+1}. {remove_country(p['description'])}" for i, p in enumerate(new_places)])
                        return BotResponse(
                            sessionId=req.sessionId,
                            botMessage=f"وجدت أكثر من مكان:\n{options}\nاختر رقم أو اسم المكان المطلوب.",
                            done=False
                        )
                else:
                    return BotResponse(sessionId=req.sessionId, botMessage="ما لقيت المكان، جرب تكتب عنوان أوضح أو مختلف.", done=False)
            else:
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage=f"✔️ تم اختيار الوجهة: {remove_country(sess['chosen_place']['address'])}\n{random_step_message('ask_pickup')}",
                    done=False
                )

        if step == "ask_pickup":
            user_reply = user_msg.strip().lower()
            if user_reply in ["موقعي", "موقعي الحالي", "الموقع الحالي"]:
                sess["pickup"] = sess["loc_txt"]
                sess["step"] = "ask_time"
                return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)
            else:
                places = smart_places_search(user_msg, sess["lat"], sess["lng"])
                if not places:
                    return BotResponse(
                        sessionId=req.sessionId,
                        botMessage="لم أتمكن من العثور على هذا المكان كنقطة انطلاق. جرب عنوان آخر.",
                        done=False
                    )
                if len(places) > 1:
                    sess["step"] = "choose_pickup"
                    sess["possible_pickup_places"] = places
                    options = "\n".join([f"{i+1}. {remove_country(p['description'])}" for i, p in enumerate(places)])
                    return BotResponse(
                        sessionId=req.sessionId,
                        botMessage=f"وجدت أكثر من مكان كنقطة انطلاق:\n{options}\nاختر رقم أو اسم المكان.",
                        done=False
                    )
                else:
                    if places[0].get('is_local') or places[0].get('is_pinecone'):
                        place_info = get_place_details_enhanced(places[0]['place_id'])
                    else:
                        place_info = get_place_details(places[0]['place_id'])
                    sess["pickup"] = place_info['address']
                    sess["step"] = "ask_time"
                    return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)

        if step == "choose_pickup":
            places = sess.get("possible_pickup_places", [])
            user_reply = user_msg.strip().lower()
            found = False
            try:
                idx = int(user_reply) - 1
                if 0 <= idx < len(places):
                    if places[idx].get('is_local') or places[idx].get('is_pinecone'):
                        place_info = get_place_details_enhanced(places[idx]['place_id'])
                    else:
                        place_info = get_place_details(places[idx]['place_id'])
                    sess["pickup"] = place_info['address']
                    sess["step"] = "ask_time"
                    found = True
            except:
                pass
            if not found:
                new_places = smart_places_search(user_msg, sess["lat"], sess["lng"])
                if new_places:
                    if len(new_places) == 1:
                        if new_places[0].get('is_local') or new_places[0].get('is_pinecone'):
                            place_info = get_place_details_enhanced(new_places[0]['place_id'])
                        else:
                            place_info = get_place_details(new_places[0]['place_id'])
                        sess["pickup"] = place_info['address']
                        sess["step"] = "ask_time"
                        return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)
                    else:
                        sess["possible_pickup_places"] = new_places
                        options = "\n".join([f"{i+1}. {remove_country(p['description'])}" for i, p in enumerate(new_places)])
                        return BotResponse(
                            sessionId=req.sessionId,
                            botMessage=f"وجدت أكثر من مكان كنقطة انطلاق:\n{options}\nاختر رقم أو اسم المكان.",
                            done=False
                        )
                else:
                    return BotResponse(sessionId=req.sessionId, botMessage="ما لقيت المكان كنقطة انطلاق. جرب عنوان أوضح أو مختلف.", done=False)
            else:
                return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_time"), done=False)

        if step == "ask_time":
            user_reply = user_msg.strip().lower()
            if user_reply in ["الآن", "حالا", "حاضر", "فوري"]:
                sess["time"] = "الآن"
            else:
                sess["time"] = user_msg.strip()
            sess["step"] = "ask_car_type"
            return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_car_type"), done=False)

        if step == "ask_car_type":
            user_reply = user_msg.strip().lower()
            if "vip" in user_reply or "في آي بي" in user_reply or "فاخرة" in user_reply:
                sess["car"] = "VIP"
            else:
                sess["car"] = "عادية"
            sess["step"] = "ask_audio"
            return BotResponse(sessionId=req.sessionId, botMessage=random_step_message("ask_audio"), done=False)

        if step == "ask_audio":
            user_reply = user_msg.strip().lower()
            if "قرآن" in user_reply or "قران" in user_reply:
                sess["audio"] = "قرآن"
            elif "موسيقى" in user_reply or "موسيقا" in user_reply or "أغاني" in user_reply:
                sess["audio"] = "موسيقى"
            else:
                sess["audio"] = "صمت"
            sess["step"] = "confirm_booking"
            summary = f"""
✔️ ملخص طلبك:
📍 من: {remove_country(sess['pickup'])}
🎯 إلى: {remove_country(sess['chosen_place']['address'])}
⏰ الوقت: {sess['time']}
🚗 نوع السيارة: {sess['car']}
🎵 الصوت: {sess['audio']}

هل تؤكد الحجز؟ (نعم/لا)
"""
            return BotResponse(sessionId=req.sessionId, botMessage=summary, done=False)

        if step == "confirm_booking":
            user_reply = user_msg.strip().lower()
            if user_reply in ["نعم", "موافق", "أكد", "تأكيد", "yes", "ok"]:
                booking_id = random.randint(10000, 99999)
                success_msg = f"""
🎉 تم تأكيد حجزك بنجاح!
رقم الحجز: {booking_id}
📱 ستصلك رسالة تأكيد قريباً
🚗 السائق في الطريق إليك
⏱️ الوقت المتوقع: 5-10 دقائق

شكراً لاستخدامك خدمة يا هو! 🚖
"""
                del sessions[req.sessionId]
                return BotResponse(sessionId=req.sessionId, botMessage=success_msg, done=True)
            else:
                return BotResponse(sessionId=req.sessionId, botMessage="تم إلغاء الحجز. هل تود بدء حجز جديد؟", done=True)

        return BotResponse(sessionId=req.sessionId, botMessage="عذراً، حدث خطأ. حاول مرة أخرى.", done=False)
    except Exception as e:
        print('خطأ في السيرفر:', e)
        return BotResponse(sessionId=req.sessionId if req.sessionId else '', botMessage="حصل خطأ بالسيرفر. جرب بعد قليل.", done=True)

