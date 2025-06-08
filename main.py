import os, json, asyncio, re
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from openai import OpenAI
import logging
import aiohttp

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Taxi Booking API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()

# ========== النماذج ==========
class ChatMessage(BaseModel):
    role: str
    content: str

    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v

class MessageRequest(BaseModel):
    user_id: str
    messages: List[ChatMessage]
    lat: Optional[float] = None
    lng: Optional[float] = None

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('User ID cannot be empty')
        return v.strip()

class MessageResponse(BaseModel):
    response: str
    booking_id: Optional[str] = None
    maps_link: Optional[str] = None
    status: str = "success"

# ========== برومبت السيناريو السوري ==========
system_prompt_base = """
أنت مساعد صوتي ذكي لتطبيق تاكسي، ترد باللهجة السورية فقط وتلتزم بالسيناريو والخطوات التالية بالتحديد:

- رحب بالمستخدم أولاً بواحدة من هذه الجمل (بدون تكرار نفس الجملة دائماً):
    - "أهلين! وين حابب تروح اليوم؟"
    - "هلا فيك! لوين رايح؟"
    - "أهلاً وسهلاً! على وين المشوار اليوم؟"

- إذا ذكر المستخدم وجهة فقط ولم يحدد مكان الانطلاق:
    - اسأله بواحدة من هذه الجمل (غير مكررة):
        - "تحب نجي ناخدك من موقعك الحالي ([اسم الموقع]) أو من مكان تاني؟"
        - "أرسللك السيارة ع موقعك الحالي ولا في عنوان تاني ببالك؟"
        - "من وين بنبلش الرحلة؟ من عندك هلق ولا من محل تاني؟"

- إذا لم يحدد وقت الرحلة:
    - اسأله بواحدة من هذه الجمل:
        - "إيمتى حابب تطلع؟"
        - "شو الوقت اللي ناسبك للمشوار؟"
        - "حابب تطلع هلق ولا بعد شوي؟"

- إذا لم يحدد نوع السيارة:
    - اسأله بواحدة من هذه الجمل:
        - "شو نوع السيارة يلي بترتاحله، عادية ولا VIP؟"
        - "بتحب سيارة عادية أو شي مميز أكتر متل VIP؟"
        - "نوع السيارة يلي بتفضله عادي ولا VIP؟"

- إذا لم يحدد الموسيقى أو الصوت:
    - اسأله بواحدة من هذه الجمل:
        - "بتحب نسمع شي بالمشوار؟"
        - "حابب نضيف تلاوة أو موسيقى بالطريق؟"
        - "بتحب نشتغل شي صوتي خلال الرحلة؟"

- إذا قال المستخدم قرآن أو تلاوة:
    - اسأله: "شو نوع التلاوة يلي بتحبها؟ أو في قارئ مفضل؟"

- لما تجمع كل المعلومات المطلوبة، اعرض له ملخص الرحلة واختر واحدة من هذه الصيغ:
    - "رحلتك من [الانطلاق] إلى [الوجهة] الساعة [الوقت] بسيارة [نوع السيارة]{، مع تلاوة قرآنية}."
    - "بننطلق من [الانطلاق] ورايحين على [الوجهة] الساعة [الوقت]، والسيارة [نوع السيارة]{، وفي تلاوة قرآن مثل ما طلبت}."
    - "المشوار من [الانطلاق] لـ [الوجهة] بـ [نوع السيارة] الساعة [الوقت]{، مع تلاوة قرآنية}."
    (إذا المستخدم طلب قرآن أضف جملة عن التلاوة)

- بعد الملخص اسأله للتأكيد بواحدة من هالجمل:
    - "ثبتلك الحجز بهالمعلومات؟"
    - "أمشي بالحجز عهالبيانات؟"
    - "جاهز أأكدلك الرحلة؟"

- إذا وافق المستخدم أو كتب نعم، أكد الحجز ورد عليه بواحدة من هالجمل:
    - "✔️ تم! رح أحجزلك الرحلة فوراً."
    - "تمام! حجزتك جاهزة هلأ."
    - "انتهينا! السيارة بالطريق إلك."

- أي سؤال إضافي أو غموض جاوبه ببساطة وود، ودائماً باللهجة السورية، ولا تخلط مع أي لغة ثانية.
- لا تخرج أبداً عن هذا السيناريو حتى لو سأل المستخدم عن شيء خارج الحجز.

مهم: استخدم عبارات عفوية وواقعية وكأنك فعلاً موظف استقبال ودود في مكتب تاكسي سوري.
"""

# ========== كشف اللغة واللهجة ==========
def detect_language(text: str) -> str:
    if not text or not text.strip():
        return 'arabic'
    text_clean = text.strip().lower()
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    arabic_keywords = [
        'بدي', 'أروح', 'وين', 'من', 'إلى', 'الى', 'تاكسي', 'سيارة', 'عادي', 'vip', 'مطار', 'جامعة', 'بيت', 'شغل',
        'خذني', 'وديني', 'وصلني', 'ابغى', 'ابي', 'أريد', 'قرآن', 'موسيقى', 'نعم', 'لا', 'شكرا', 'مرحبا',
        'صباح', 'مساء', 'يلا', 'الحين', 'هلا', 'دغري', 'بسرعة', 'ش', 'شو', 'بدك', 'أنت', 'انا', 'توصلني'
    ]
    english_keywords = [
        'want', 'go', 'from', 'to', 'at', 'taxi', 'car', 'airport', 'university', 'home', 'work', 'take', 'me',
        'yes', 'no', 'thanks', 'hello', 'hi', 'music', 'quran', 'morning', 'evening', 'please', 'right now', 'now'
    ]
    arabic_score = arabic_chars * 2
    english_score = english_chars * 2
    for word in arabic_keywords:
        if word in text_clean:
            arabic_score += 3
    for word in english_keywords:
        if word in text_clean:
            english_score += 3
    if len(text_clean) < 3:
        return 'arabic'
    return 'arabic' if arabic_score >= english_score else 'english'

def extract_intent(text: str) -> str:
    booking_keywords = ['حجز', 'احجز', 'اريد', 'بدي', 'ابغى', 'أريد', 'book', 'reservation', 'ride', 'go', 'وصلني', 'وديني']
    cancel_keywords = ['الغاء', 'إلغاء', 'cancel', 'وقف', 'لا بدي', 'لا أريد', 'stop', 'إلغاء الحجز']
    if any(k in text for k in booking_keywords):
        return "booking"
    if any(k in text for k in cancel_keywords):
        return "cancel"
    return "unknown"

def extract_entities_gpt(text: str) -> Dict[str, str]:
    prompt = f"""
    حلل الرسالة التالية من مستخدم تطبيق تاكسي باللهجة السورية أو السعودية أو الإنجليزية، واستخرج فقط هذه الحقول بصيغة JSON:
    destination, pickup_location, ride_time, car_type, music, notes.
    إذا لم يوجد حقل اتركه فارغ.
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
    except Exception as e:
        logger.warning(f"Failed to extract entities by GPT: {e}")
        return {}

def parse_time_from_text(text: str) -> Optional[str]:
    time_patterns = [
        r'(\d{1,2}:\d{1,2})', r'(\d{1,2}\s*(?:ص|م|am|pm))', r'(الآن|هلق|الحين|الساعة\s+\d{1,2})',
        r'(بكرة|غداً|غدا|غدًا|اليوم|بعد المغرب|بعد العصر)'
    ]
    for pattern in time_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def clarify_if_ambiguous(entities: Dict[str, str]) -> Optional[str]:
    ambiguous_places = ["الجامعة", "المطار", "المول", "المستشفى", "البيت", "المدرسة"]
    if entities.get("destination", "").strip() in ambiguous_places:
        options = {
            "الجامعة": "جامعة دمشق، الجامعة الافتراضية، الجامعة العربية الدولية ...",
            "المطار": "مطار دمشق، مطار الملك خالد، مطار الملك عبدالعزيز ...",
            "المول": "شام سيتي سنتر، مول العرب ...",
            "البيت": "الرجاء تحديد عنوان البيت بالتفصيل.",
            "المستشفى": "مشفى المواساة، مشفى الشامي ...",
            "المدرسة": "مدرسة دار السلام، مدرسة الإخلاص ..."
        }
        extra = options.get(entities["destination"], "")
        return f"أي {entities['destination']} تقصد؟ {extra}"
    return None

async def get_location_name(lat: float, lng: float) -> Optional[str]:
    if not GOOGLE_MAPS_API_KEY:
        return None
    try:
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "latlng": f"{lat},{lng}",
            "key": GOOGLE_MAPS_API_KEY,
            "language": "ar",
            "result_type": "locality|sublocality|administrative_area_level_2"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
                if data.get("status") == "OK" and data.get("results"):
                    for result in data["results"]:
                        for component in result.get("address_components", []):
                            types = component.get("types", [])
                            if any(t in ["locality", "sublocality", "administrative_area_level_2"] for t in types):
                                return component.get("long_name", "")
                    if data["results"][0].get("formatted_address"):
                        address = data["results"][0]["formatted_address"]
                        parts = address.split(",")
                        if len(parts) > 1:
                            return parts[1].strip()
    except Exception as e:
        logger.error(f"Reverse geocoding error: {e}")
    return None

async def find_nearest_place(place: str, lat: Optional[float] = None, lng: Optional[float] = None) -> Dict[str, Any]:
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("Google Maps API key not provided")
        return {"exists": True, "name": place}
    try:
        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": place,
            "key": GOOGLE_MAPS_API_KEY,
            "language": "ar"
        }
        if lat and lng:
            params["location"] = f"{lat},{lng}"
            params["radius"] = 50000
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
                if data.get("status") == "OK" and data.get("results"):
                    result = data["results"][0]
                    return {
                        "exists": True,
                        "name": result.get("name", place),
                        "address": result.get("formatted_address", ""),
                        "lat": result["geometry"]["location"]["lat"],
                        "lng": result["geometry"]["location"]["lng"],
                        "place_id": result.get("place_id", "")
                    }
                else:
                    logger.warning(f"Places API returned: {data.get('status', 'Unknown error')}")
    except asyncio.TimeoutError:
        logger.error("Google Places API timeout")
    except Exception as e:
        logger.error(f"Unexpected error in find_nearest_place: {e}")
    return {"exists": False}

def get_google_maps_link(lat: float, lng: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"

async def save_booking_to_file(booking_data: Dict[str, Any]) -> Optional[str]:
    async with file_lock:
        try:
            booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
            booking_record = {
                "booking_id": booking_id,
                "timestamp": datetime.now().isoformat(),
                "status": "pending",
                **booking_data
            }
            bookings = []
            if os.path.exists(BOOKINGS_FILE):
                try:
                    with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            bookings = json.loads(content)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in bookings file, starting fresh")
                    bookings = []
            bookings.append(booking_record)
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(bookings, f, ensure_ascii=False, indent=2)
            logger.info(f"Booking saved successfully: {booking_id}")
            return booking_id
        except Exception as e:
            logger.error(f"Error saving booking: {e}")
            return None

# ========== نقطة نهاية المحادثة ==========
@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest, background_tasks: BackgroundTasks):
    try:
        messages = [msg.model_dump() for msg in request.messages]
        last_user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        language = detect_language(last_user_message)
        intent = extract_intent(last_user_message)
        entities = extract_entities_gpt(last_user_message)
        current_location_name = None
        if request.lat and request.lng:
            current_location_name = await get_location_name(request.lat, request.lng)
        if not entities.get("ride_time"):
            parsed_time = parse_time_from_text(last_user_message)
            if parsed_time:
                entities["ride_time"] = parsed_time
        ambiguous_msg = clarify_if_ambiguous(entities)
        if ambiguous_msg:
            return MessageResponse(response=ambiguous_msg, status="clarify")
        maps_link = None
        if entities.get("destination"):
            place_info = await find_nearest_place(entities["destination"], request.lat, request.lng)
            if not place_info["exists"]:
                return MessageResponse(
                    response="الموقع غير واضح. حاول تكتب اسم المكان بدقة أو فعل الموقع.",
                    status="location_not_found"
                )
            elif "lat" in place_info and "lng" in place_info:
                maps_link = get_google_maps_link(place_info["lat"], place_info["lng"])
        
        # 👇 بناء الرسالة للنموذج باستخدام البرومبت المخصص
        system_message = (
            system_prompt_base
            + f"\n\nلغة المستخدم: {'عربية' if language == 'arabic' else 'English'}"
            + f"\nمعرف المستخدم: {request.user_id}"
            + f"\nالوقت: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            + (f"\nالموقع الحالي: {current_location_name}" if current_location_name else "")
            + f"\nالنية: {intent}"
            + f"\nالكيانات: {json.dumps(entities, ensure_ascii=False)}"
        )
        model_messages = [{"role": "system", "content": system_message}] + messages
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=model_messages,
                temperature=0.6,  # لردود عفوية
                max_tokens=400,
                timeout=30
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(status_code=503, detail="صار خطأ بالمعالجة. جرب مرة تانية.")
        booking_id = None
        booking_confirmed = any(x in response_text for x in [
            "✔️ تم", "تمام! حجزتك جاهزة", "انتهينا! السيارة بالطريق إلك"
        ])
        if booking_confirmed and entities.get("destination"):
            entities["user_id"] = request.user_id
            booking_id = await save_booking_to_file(entities)
            if booking_id:
                response_text += f"\n\n📱 رقم حجزك: {booking_id}"
        if maps_link:
            map_text = f"\n\n🗺️ {'رابط الموقع على الخريطة' if language == 'arabic' else 'View location on map'}:\n{maps_link}"
            response_text += map_text
        return MessageResponse(
            response=response_text,
            booking_id=booking_id,
            maps_link=maps_link,
            status="success"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="صار خطأ بالمعالجة. جرب مرة تانية.")

@app.get("/bookings")
async def get_all_bookings():
    try:
        if not os.path.exists(BOOKINGS_FILE):
            return []
        with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            bookings = json.loads(content)
            return sorted(bookings, key=lambda x: x.get("timestamp", ""), reverse=True)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in bookings file")
        return []
    except Exception as e:
        logger.error(f"Error reading bookings: {e}")
        raise HTTPException(status_code=500, detail="خطأ في جلب الحجوزات")

@app.get("/booking/{booking_id}")
async def get_booking_by_id(booking_id: str):
    try:
        bookings = await get_all_bookings()
        booking = next((b for b in bookings if b.get("booking_id") == booking_id), None)
        if not booking:
            raise HTTPException(status_code=404, detail="الحجز غير موجود")
        return booking
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching booking {booking_id}: {e}")
        raise HTTPException(status_code=500, detail="خطأ في جلب الحجز")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=1
    )
