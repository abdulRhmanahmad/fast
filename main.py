import os, json, asyncio, re, requests
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from openai import OpenAI
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Taxi Booking API", version="1.0.0")

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# المتغيرات العامة
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()

# النماذج المحسنة
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

class BookingDetails(BaseModel):
    destination: Optional[str] = None
    pickup_location: Optional[str] = None
    car_type: Optional[str] = None
    ride_time: Optional[str] = None
    music: Optional[str] = None
    notes: Optional[str] = None
    user_id: str

# وظائف محسنة لكشف اللغة
def detect_language(text: str) -> str:
    """كشف اللغة بدقة أكبر"""
    if not text or not text.strip():
        return 'arabic'
    
    text_clean = text.strip().lower()
    
    # عدد الأحرف العربية والإنجليزية
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # كلمات مفتاحية عربية وإنجليزية محسنة
    arabic_keywords = [
        'بدي', 'أروح', 'وين', 'من', 'إلى', 'الى', 'الساعة', 'تاكسي', 'سيارة', 
        'عادي', 'مطار', 'جامعة', 'بيت', 'شغل', 'خذني', 'أبي', 'أبغى', 'أريد',
        'قرآن', 'موسيقى', 'نعم', 'لا', 'شكرا', 'مرحبا', 'صباح', 'مساء'
    ]
    english_keywords = [
        'want', 'go', 'from', 'to', 'at', 'taxi', 'car', 'airport', 'university', 
        'home', 'work', 'take', 'me', 'yes', 'no', 'thanks', 'hello', 'hi',
        'music', 'quran', 'morning', 'evening', 'please'
    ]
    
    arabic_score = arabic_chars * 2
    english_score = english_chars * 2
    
    # إضافة نقاط للكلمات المفتاحية
    for word in arabic_keywords:
        if word in text_clean:
            arabic_score += 3
    
    for word in english_keywords:
        if word in text_clean:
            english_score += 3
    
    # إذا كان النص قصير جداً، افترض العربية
    if len(text_clean) < 3:
        return 'arabic'
    
    return 'arabic' if arabic_score >= english_score else 'english'

def get_response_templates(language: str) -> Dict[str, str]:
    """قوالب الردود حسب اللغة"""
    if language == 'english':
        return {
            'greeting': "Hello! I'm Yaho, your ride assistant. Where would you like to go today? 🚖",
            'location_error': "The entered location could not be found on the map. Please try a clearer name or enable location services.",
            'processing_error': "Sorry, there was an error processing your request. Please try again."
        }
    else:
        return {
            'greeting': "مرحباً! أنا يا هو، مساعدك للمشاوير. أين تود الذهاب اليوم؟ 🚖",
            'location_error': "الموقع المدخل غير موجود أو غير واضح على الخريطة. حاول كتابة اسم المكان بدقة أو فعّل خدمة الموقع.",
            'processing_error': "عذراً، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى."
        }

# تحسين وظيفة البحث عن الأماكن
async def find_nearest_place(place: str, lat: Optional[float] = None, lng: Optional[float] = None) -> Dict[str, Any]:
    """البحث عن أقرب مكان مع معالجة أفضل للأخطاء"""
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("Google Maps API key not provided")
        return {"exists": True, "name": place}  # افتراض أن المكان موجود إذا لم يكن لدينا API key
    
    try:
        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": place,
            "key": GOOGLE_MAPS_API_KEY,
            "language": "ar"  # تفضيل اللغة العربية
        }
        
        # إضافة الموقع إذا كان متوفراً
        if lat and lng:
            params["location"] = f"{lat},{lng}"
            params["radius"] = 50000  # زيادة نصف القطر إلى 50 كم
        
        async with asyncio.timeout(10):  # timeout لتجنب التعليق
            response = requests.get(base_url, params=params, timeout=8)
            response.raise_for_status()
            
        data = response.json()
        
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
    except requests.RequestException as e:
        logger.error(f"Places API request error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in find_nearest_place: {e}")
    
    return {"exists": False}

def get_google_maps_link(lat: float, lng: float) -> str:
    """إنشاء رابط خرائط جوجل"""
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"

# تحسين النظام
system_prompt_base = """
أنت مساعد صوتي ذكي اسمك "يا هو" داخل تطبيق تاكسي متطور. مهمتك مساعدة المستخدمين في حجز المشاوير بطريقة سهلة وودودة.

## القواعد الأساسية:
- استخدم نفس لغة المستخدم في كل رد (عربي أو إنجليزي)
- لا تخلط بين اللغات في نفس الرد
- اسأل سؤالاً واحداً واضحاً في كل مرة
- كن ودوداً ومفيداً
- تذكر المعلومات السابقة في المحادثة

## خطوات الحجز:
1. الترحيب والسؤال عن الوجهة
2. تحديد نقطة الانطلاق (الموقع الحالي أم مكان آخر)
3. السؤال عن وقت الانطلاق
4. نوع السيارة (عادية أم VIP)
5. تفضيلات الصوت (قرآن، موسيقى، صمت)
6. ملخص الحجز والتأكيد

## ردود الترحيب:
- عربي: "مرحباً! أنا يا هو، مساعدك الذكي للمشاوير. أين تود الذهاب اليوم؟ 🚖"
- إنجليزي: "Hello! I'm Yaho, your smart ride assistant. Where would you like to go today? 🚖"

## أمثلة على الأسئلة:
- "من أين تود الانطلاق؟ من موقعك الحالي أم من مكان آخر؟"
- "متى تود الانطلاق؟ الآن أم في وقت محدد؟"
- "أي نوع سيارة تفضل؟ عادية أم VIP؟"
- "هل تود الاستماع لشيء أثناء الرحلة؟"

## التأكيد النهائي:
عند اكتمال المعلومات، اعرض ملخصاً كاملاً واطلب التأكيد:
"ملخص رحلتك:
• الوجهة: [الوجهة]
• الانطلاق: [نقطة الانطلاق]
• الوقت: [وقت الانطلاق]
• نوع السيارة: [عادية/VIP]
• الصوت: [قرآن/موسيقى/صمت]

هل تؤكد الحجز؟"

## بعد التأكيد:
"✔️ تم تأكيد حجزك بنجاح! سيصلك السائق في الوقت المحدد."

تذكر: كن طبيعياً وودوداً، واستخدم الرموز التعبيرية بشكل مناسب.
"""

def extract_last_qa(messages: List[Dict[str, str]]) -> tuple[str, str]:
    """استخراج آخر سؤال وجواب"""
    for i in reversed(range(len(messages) - 1)):
        if messages[i]["role"] == "assistant" and messages[i + 1]["role"] == "user":
            return messages[i]["content"], messages[i + 1]["content"]
    return "", ""

def parse_booking_summary(text: str) -> Dict[str, str]:
    """استخراج تفاصيل الحجز من النص"""
    details = {}
    
    # أنماط البحث للعربية والإنجليزية
    patterns = {
        "destination": [r"الوجهة[:\s]*([^\n•]+)", r"destination[:\s]*([^\n•]+)"],
        "pickup_location": [r"الانطلاق[:\s]*([^\n•]+)", r"من[:\s]*([^\n•]+)", r"pickup[:\s]*([^\n•]+)", r"from[:\s]*([^\n•]+)"],
        "car_type": [r"نوع السيارة[:\s]*([^\n•]+)", r"السيارة[:\s]*([^\n•]+)", r"car[:\s]*([^\n•]+)"],
        "ride_time": [r"الوقت[:\s]*([^\n•]+)", r"وقت الانطلاق[:\s]*([^\n•]+)", r"time[:\s]*([^\n•]+)"],
        "music": [r"الصوت[:\s]*([^\n•]+)", r"الموسيقا[:\s]*([^\n•]+)", r"music[:\s]*([^\n•]+)", r"audio[:\s]*([^\n•]+)"],
        "notes": [r"ملاحظات[:\s]*([^\n•]+)", r"notes[:\s]*([^\n•]+)"]
    }
    
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip(" :[]•-")
                if value and value != "-":
                    details[key] = value
                break
    
    return details

async def save_booking_to_file(booking_data: Dict[str, Any]) -> Optional[str]:
    """حفظ الحجز في الملف مع معالجة أفضل للأخطاء"""
    async with file_lock:
        try:
            booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # إنشاء سجل الحجز
            booking_record = {
                "booking_id": booking_id,
                "timestamp": datetime.now().isoformat(),
                "status": "pending",
                **booking_data
            }
            
            # قراءة الحجوزات الموجودة
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
            
            # إضافة الحجز الجديد
            bookings.append(booking_record)
            
            # حفظ الملف
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(bookings, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Booking saved successfully: {booking_id}")
            return booking_id
            
        except Exception as e:
            logger.error(f"Error saving booking: {e}")
            return None

# نقاط النهاية المحسنة
@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest, background_tasks: BackgroundTasks):
    """نقطة نهاية المحادثة المحسنة"""
    try:
        messages = [msg.model_dump() for msg in request.messages]
        
        # الحصول على آخر رسالة من المستخدم
        last_user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # كشف اللغة
        language = detect_language(last_user_message)
        templates = get_response_templates(language)
        
        # التحقق من الأماكن إذا لزم الأمر
        place_keywords = ["من", "الى", "إلى", "to", "from", "destination", "pickup", "الوجهة", "مكان", "location"]
        needs_place_verification = any(keyword in last_user_message.lower() for keyword in place_keywords)
        
        maps_link = None
        if needs_place_verification:
            place_info = await find_nearest_place(last_user_message, request.lat, request.lng)
            if not place_info["exists"]:
                return MessageResponse(
                    response=templates['location_error'],
                    status="location_not_found"
                )
            elif "lat" in place_info and "lng" in place_info:
                maps_link = get_google_maps_link(place_info["lat"], place_info["lng"])
        
        # بناء الرسالة للنموذج
        system_message = system_prompt_base
        
        # إضافة معلومات السياق
        if len(messages) <= 1:
            greeting = templates['greeting']
            system_message += f"\n\nابدأ المحادثة بـ: {greeting}"
        
        # إضافة آخر سؤال وجواب
        last_question, last_answer = extract_last_qa(messages)
        if last_question and last_answer:
            system_message += f"\n\nآخر سؤال: {last_question}\nآخر جواب: {last_answer}\nانتقل للخطوة التالية."
        
        # إضافة معلومات إضافية
        system_message += f"\n\nلغة المستخدم: {'عربية' if language == 'arabic' else 'English'}"
        system_message += f"\nمعرف المستخدم: {request.user_id}"
        system_message += f"\nالوقت: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # إعداد الرسائل للنموذج
        model_messages = [{"role": "system", "content": system_message}] + messages
        
        # استدعاء النموذج
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=model_messages,
                temperature=0.3,
                max_tokens=400,
                timeout=30
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(status_code=503, detail=templates['processing_error'])
        
        # التحقق من تأكيد الحجز
        booking_id = None
        booking_confirmed = any(indicator in response_text for indicator in [
            "تم تأكيد حجزك", "✔️", "تم!", "confirmed", "booking confirmed"
        ])
        
        if booking_confirmed:
            booking_details = parse_booking_summary(response_text)
            if booking_details.get("destination"):
                booking_details["user_id"] = request.user_id
                booking_id = await save_booking_to_file(booking_details)
                if booking_id:
                    response_text += f"\n\n📱 رقم حجزك: {booking_id}"
        
        # إضافة رابط الخريطة إذا كان متوفراً
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
        language = detect_language(request.messages[-1].content if request.messages else "")
        templates = get_response_templates(language)
        raise HTTPException(status_code=500, detail=templates['processing_error'])

@app.get("/bookings")
async def get_all_bookings():
    """جلب جميع الحجوزات"""
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
    """جلب حجز محدد بالمعرف"""
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
    """فحص صحة الخدمة"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
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
