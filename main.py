
import os, uuid, requests, math, random, re, difflib
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

# Helpers
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

def remove_country(text):
    if not text:
        return ""
    return re.sub(r"(،?\s*سوريا)$", "", text.strip())

def get_location_text(lat, lng):
    address = reverse_geocode(lat, lng)
    if not address:
        return "موقعك الحالي"
    return format_address(address)

# NLP Helpers
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
        syrian_cities = ["دمشق", "حلب", "حمص", "حماة", "اللاذقية", "طرطوس"]
        for city in syrian_cities:
            if city not in query:
                expanded_queries.append(f"{query} {city}")
                expanded_queries.append(f"{query}, {city}")
        if "شعلان" in query.lower():
            expanded_queries.extend([
                "الشعلان دمشق",
                "شارع الشعلان",
                "حي الشعلان",
                "منطقة الشعلان"
            ])
        common_corrections = {
            "شعلان": ["الشعلان", "شارع الشعلان"],
            "مزه": ["المزة", "حي المزة"],
            "جسر": ["الجسر الأبيض", "جسر فيكتوريا"],
            "ساحة": ["ساحة الأمويين", "ساحة المحافظة"],
        }
        for mistake, corrections in common_corrections.items():
            if mistake in query.lower():
                expanded_queries.extend(corrections)
    return list(set(expanded_queries))

def smart_places_search(query: str, user_lat: float, user_lng: float, max_results=5) -> list:
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
        unique_results = fuzzy_location_search(query, user_lat, user_lng)
    return unique_results[:max_results]

def fuzzy_location_search(query: str, user_lat: float, user_lng: float) -> list:
    known_places = {
        "الشعلان": "الشعلان، دمشق، سوريا",
        "شعلان": "الشعلان، دمشق، سوريا", 
        "المزة": "المزة، دمشق، سوريا",
        "مزه": "المزة، دمشق، سوريا",
        "الحمدانية": "الحمدانية، حلب، سوريا",
        "حمدانية": "الحمدانية، حلب، سوريا",
        "صلاح الدين": "صلاح الدين، حلب، سوريا",
        "الأزبكية": "الأزبكية، حلب، سوريا",
        "أزبكية": "الأزبكية، حلب، سوريا",
        "كفرسوسة": "كفرسوسة، دمشق، سوريا",
        "جرمانا": "جرمانا، ريف دمشق، سوريا",
        "دوما": "دوما، ريف دمشق، سوريا",
        "حرستا": "حرستا، ريف دمشق، سوريا",
        "معضمية": "المعضمية، ريف دمشق، سوريا",
        "التل": "التل، ريف دمشق، سوريا",
        "صحنايا": "صحنايا، ريف دمشق، سوريا"
    }
    query_clean = clean_arabic_text(query.lower())
    for key, value in known_places.items():
        if key.lower() in query_clean or query_clean in key.lower():
            return [{
                "description": value,
                "place_id": f"local_{key}",
                "is_local": True
            }]
    matches = difflib.get_close_matches(query_clean, known_places.keys(), n=3, cutoff=0.6)
    results = []
    for match in matches:
        results.append({
            "description": known_places[match],
            "place_id": f"local_{match}",
            "is_local": True
        })
    return results

def get_place_details_enhanced(place_id: str) -> dict:
    if place_id.startswith("local_"):
        location_name = place_id.replace("local_", "")
        local_coordinates = {
            "الشعلان": {"lat": 33.5138, "lng": 36.2765},
            "شعلان": {"lat": 33.5138, "lng": 36.2765},
            "المزة": {"lat": 33.5024, "lng": 36.2213},
            "مزه": {"lat": 33.5024, "lng": 36.2213},
        }
        coords = local_coordinates.get(location_name, {"lat": 33.5138, "lng": 36.2765})
        return {
            "address": f"{location_name}، دمشق، سوريا",
            "lat": coords["lat"],
            "lng": coords["lng"],
        }
    else:
        return get_place_details(place_id)

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

class UserRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class BotResponse(BaseModel):
    sessionId: str
    botMessage: str
    done: bool = False

# رسائل متنوعة
step_messages = {
    "ask_destination": [
        "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير 🚖.\nوين حابب تروح اليوم؟",
        "هلا فيك! حددلي وجهتك لو سمحت.",
        "أهلين، شو عنوان المكان يلي رايح عليه؟",
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

        # منطق ChatGPT للردود العامة خارج سيناريو الحجز
        if is_out_of_booking_context(user_msg, step):
            gpt_reply = ask_gpt(user_msg)
            step_q = current_step_question(sess)
            return BotResponse(
                sessionId=req.sessionId,
                botMessage=f"{gpt_reply}\n\n{step_q}",
                done=False
            )

        # ---- منطق الحجز ----
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
                if places[0].get('is_local'):
                    place_info = get_place_details_enhanced(places[0]['place_id'])
                else:
                    place_info = get_place_details(places[0]['place_id'])
                sess["chosen_place"] = place_info
                sess["step"] = "ask_pickup"
                return BotResponse(
                    sessionId=req.sessionId,
                    botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])}.\n{random_step_message('ask_pickup')}",
                    done=False
                )

        if step == "choose_destination":
            places = sess.get("possible_places", [])
            user_reply = user_msg.strip().lower()
            found = False
            try:
                idx = int(user_reply) - 1
                if 0 <= idx < len(places):
                    if places[idx].get('is_local'):
                        place_info = get_place_details_enhanced(places[idx]['place_id'])
                    else:
                        place_info = get_place_details(places[idx]['place_id'])
                    sess["chosen_place"] = place_info
                    sess["step"] = "ask_pickup"
                    found = True
            except:
                pass
            if not found:
                # لو كتب خيار مو ضمن القائمة، رجّع يبحث عنه!
                new_places = smart_places_search(user_msg, sess["lat"], sess["lng"])
                if new_places:
                    if len(new_places) == 1:
                        if new_places[0].get('is_local'):
                            place_info = get_place_details_enhanced(new_places[0]['place_id'])
                        else:
                            place_info = get_place_details(new_places[0]['place_id'])
                        sess["chosen_place"] = place_info
                        sess["step"] = "ask_pickup"
                        return BotResponse(
                            sessionId=req.sessionId,
                            botMessage=f"✔️ تم اختيار الوجهة: {remove_country(place_info['address'])}.\n{random_step_message('ask_pickup')}",
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
                    botMessage=f"✔️ تم اختيار الوجهة: {remove_country(sess['chosen_place']['address'])}.\n{random_step_message('ask_pickup')}",
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
                    if places[0].get('is_local'):
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
                    if places[idx].get('is_local'):
                        place_info = get_place_details_enhanced(places[idx]['place_id'])
                    else:
                        place_info = get_place_details(places[idx]['place_id'])
                    sess["pickup"] = place_info['address']
                    sess["step"] = "ask_time"
                    found = True
            except:
                pass
            if not found:
                # لو كتب خيار مو ضمن القائمة، رجع يبحث عنه!
                new_places = smart_places_search(user_msg, sess["lat"], sess["lng"])
                if new_places:
                    if len(new_places) == 1:
                        if new_places[0].get('is_local'):
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
                booking_id = create_mock_booking(
                    pickup=sess['pickup'],
                    destination=sess['chosen_place']['address'],
                    time=sess['time'],
                    car_type=sess['car'],
                    audio_pref=sess['audio']
                )
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

