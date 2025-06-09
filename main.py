import os, uuid, requests
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Eastern Province Taxi Booking", version="2.0.0")
sessions: Dict[str, Dict[str, Any]] = {}

# قائمة المدن والمناطق المدعومة في الشرقية
EASTERN_PROVINCE_CITIES = [
    "الدمام", "الخبر", "الظهران", "الجبيل", "القطيف", "رأس تنورة", 
    "سيهات", "الأحساء", "الهفوف", "المبرز", "بقيق", "العديد",
    "النعيرية", "خفجي", "حفر الباطن", "الخفجي"
]

def geocode_and_check_eastern(name: str) -> Optional[str]:
    """
    التحقق من أن الموقع ضمن المنطقة الشرقية
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("Google Maps API key not provided")
        return None
        
    try:
        url = (
            "https://maps.googleapis.com/maps/api/geocode/json"
            f"?address={name}&region=SA&language=ar&key={GOOGLE_MAPS_API_KEY}"
        )
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data["status"] != "OK" or not data["results"]:
            logger.warning(f"Geocoding failed for: {name}")
            return None
            
        result = data["results"][0]
        address = result["formatted_address"]
        
        # التحقق من وجود "الشرقية" أو "Eastern Province" في مكونات العنوان
        for comp in result["address_components"]:
            long_name = comp.get("long_name", "")
            if ("الشرقية" in long_name) or ("Eastern Province" in long_name):
                logger.info(f"Location confirmed in Eastern Province: {address}")
                return address
                
        # إذا لم نجد "الشرقية" مباشرة، نتحقق من أسماء المدن المعروفة
        for city in EASTERN_PROVINCE_CITIES:
            if city in address:
                logger.info(f"Location confirmed by city match: {address}")
                return address
                
        logger.warning(f"Location not in Eastern Province: {address}")
        return None
        
    except requests.RequestException as e:
        logger.error(f"Request error in geocoding: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in geocoding: {e}")
        return None

def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """
    تحويل الإحداثيات إلى عنوان والتحقق من المنطقة الشرقية
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("Google Maps API key not provided")
        return None
        
    try:
        url = (
            "https://maps.googleapis.com/maps/api/geocode/json"
            f"?latlng={lat},{lng}&language=ar&region=SA&key={GOOGLE_MAPS_API_KEY}"
        )
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data["status"] != "OK" or not data["results"]:
            logger.warning(f"Reverse geocoding failed for: {lat}, {lng}")
            return None
            
        result = data["results"][0]
        address = result["formatted_address"]
        
        # التحقق من وجود "الشرقية" أو "Eastern Province"
        for comp in result["address_components"]:
            long_name = comp.get("long_name", "")
            if ("الشرقية" in long_name) or ("Eastern Province" in long_name):
                logger.info(f"Current location confirmed in Eastern Province: {address}")
                return address
                
        # التحقق من أسماء المدن المعروفة
        for city in EASTERN_PROVINCE_CITIES:
            if city in address:
                logger.info(f"Current location confirmed by city match: {address}")
                return address
                
        logger.warning(f"Current location not in Eastern Province: {address}")
        return None
        
    except requests.RequestException as e:
        logger.error(f"Request error in reverse geocoding: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in reverse geocoding: {e}")
        return None

def extract_destination(text: str) -> str:
    """
    استخراج اسم الوجهة من نص المستخدم
    """
    try:
        prompt = f'استخرج اسم الوجهة من الرسالة التالية بدون أي كلمات إضافية:\n"{text}"'
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أجب بالاسم فقط."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=50,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error extracting destination: {e}")
        return text.strip()

def get_eastern_province_examples() -> str:
    """
    إرجاع أمثلة من المدن المدعومة
    """
    examples = EASTERN_PROVINCE_CITIES[:6]  # أول 6 مدن
    return "، ".join(examples)

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
    """
    إنشاء جلسة جديدة مع التحقق من الموقع الحالي
    """
    # التحقق من صحة الإحداثيات
    if lat is None or lng is None:
        return "", "لا أستطيع تحديد موقعك. الرجاء إرسال الإحداثيات أولاً."
    
    # التحقق أن الموقع الحالي داخل الشرقية
    start_name = reverse_geocode(lat, lng)
    if not start_name:
        examples = get_eastern_province_examples()
        return "", (
            f"عذراً، هذه الخدمة متوفرة فقط في المنطقة الشرقية. "
            f"يرجى تحديد موقعك داخل إحدى هذه المدن: {examples}."
        )
    
    # إنشاء جلسة جديدة
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
    
    logger.info(f"New session created: {sess_id} at {start_name}")
    return sess_id, "مرحباً! إلى أين تريد الذهاب اليوم؟"

def proceed(session: Dict[str, Any], user_input: str) -> str:
    """
    معالجة خطوات المحادثة
    """
    step = session["step"]

    # 1) السؤال عن الوجهة
    if step == "ask_destination":
        dest = extract_destination(user_input)
        dest_name = geocode_and_check_eastern(dest)
        
        if not dest_name:
            examples = get_eastern_province_examples()
            return (
                f"تعذر تحديد موقع الوجهة أو أنها ليست ضمن المنطقة الشرقية. "
                f"يرجى كتابة اسم أوضح من هذه المناطق المدعومة: {examples}."
            )
        
        session["dest_name"] = dest_name
        session["step"] = "ask_start"
        logger.info(f"Destination set: {dest_name}")
        
        return (
            f"هل تريد أن نأخذك من موقعك الحالي ({session['start_name']})"
            " أم تفضل الانطلاق من مكان آخر؟"
        )

    # 2) السؤال عن جهة الانطلاق
    elif step == "ask_start":
        txt = user_input.strip().lower()
        
        # إذا اختار الموقع الحالي
        if txt in {"موقعي", "موقعي الحالي", "الموقع الحالي", "من هنا", "من موقعي"}:
            # الموقع الحالي تم التحقق منه مسبقاً
            logger.info(f"Using current location: {session['start_name']}")
        else:
            # التحقق من الموقع الجديد
            start_name = geocode_and_check_eastern(user_input)
            if not start_name:
                examples = get_eastern_province_examples()
                return (
                    f"تعذر تحديد موقع الانطلاق أو أنه ليس ضمن المنطقة الشرقية. "
                    f"يرجى كتابة اسم أوضح من هذه المناطق المدعومة: {examples}."
                )
            session["start_name"] = start_name
            logger.info(f"Pickup location updated: {start_name}")
        
        session["step"] = "ask_time"
        return "متى تريد الانطلاق؟"

    # 3) السؤال عن الوقت
    elif step == "ask_time":
        session["time"] = user_input
        session["step"] = "ask_car"
        logger.info(f"Time set: {user_input}")
        return "ما نوع السيارة التي تفضلها؟ عادية أم VIP؟"

    # 4) السؤال عن نوع السيارة
    elif step == "ask_car":
        session["car"] = user_input
        session["step"] = "ask_audio"
        logger.info(f"Car type set: {user_input}")
        return (
            "هل تود الاستماع إلى شيء أثناء الرحلة؟ "
            "يمكنك اختيار القرآن الكريم، الموسيقى، أو الصمت."
        )

    # 5) السؤال عن الصوت
    elif step == "ask_audio":
        txt = user_input.strip().lower()
        if txt in {"القرآن", "قرآن", "القران", "quran"}:
            session["audio"] = "القرآن"
            session["step"] = "ask_reciter"
            logger.info("Audio preference: Quran")
            return "هل لديك قارئ مفضل أو نوع تلاوة تفضله؟"
        else:
            session["audio"] = user_input
            session["step"] = "summary"
            logger.info(f"Audio preference: {user_input}")
            return build_summary(session)

    # 6) السؤال عن القارئ (إذا اختار القرآن)
    elif step == "ask_reciter":
        session["reciter"] = user_input
        session["step"] = "summary"
        logger.info(f"Reciter preference: {user_input}")
        return build_summary(session)

    # 7) عرض الملخص وطلب التأكيد
    elif step == "summary":
        txt = user_input.strip().lower()
        if txt in {"نعم", "أجل", "أكيد", "موافق", "تمام", "اوك", "ok", "yes"}:
            session["step"] = "confirmed"
            logger.info(f"Booking confirmed for session: {session}")
            return "تم تأكيد الحجز! ستصلك السيارة في الوقت المحدد. شكراً لك!"
        else:
            session["step"] = "canceled"
            logger.info("Booking canceled by user")
            return "تم إلغاء الحجز بناءً على طلبك. شكراً لك!"

    # حالة غير معروفة
    return "عذراً، لم أفهم. هل يمكنك التوضيح أو إعادة المحاولة؟"

def build_summary(session: Dict[str, Any]) -> str:
    """
    بناء ملخص الرحلة
    """
    base = (
        f"رحلتك من {session['start_name']} إلى {session['dest_name']} "
        f"في الساعة {session['time']} بسيارة {session['car']}"
    )
    
    if session["audio"] == "القرآن":
        base += "، مع تلاوة قرآنية"
        if session["reciter"]:
            base += f" بصوت {session['reciter']}"
    elif session["audio"]:
        base += f"، مع {session['audio']}"
    
    return base + ". هل تريد تأكيد الحجز بهذه التفاصيل؟"

@app.post("/chatbot", response_model=BotResponse)
def chatbot(req: UserRequest):
    """
    نقطة الدخول الرئيسية للشات بوت
    """
    try:
        # إذا لم تكن هناك جلسة أو الجلسة غير موجودة
        if not req.sessionId or req.sessionId not in sessions:
            sess_id, msg = new_session(req.lat, req.lng)
            if not sess_id:  # فشل في إنشاء الجلسة
                return BotResponse(sessionId="", botMessage=msg)
            return BotResponse(sessionId=sess_id, botMessage=msg)

        # معالجة الجلسة الموجودة
        session = sessions[req.sessionId]
        reply = proceed(session, req.userInput or "")
        done = session.get("step") in {"confirmed", "canceled"}
        
        # تنظيف الجلسة إذا انتهت
        if done:
            logger.info(f"Session {req.sessionId} completed and cleaned up")
            sessions.pop(req.sessionId, None)
        
        return BotResponse(sessionId=req.sessionId, botMessage=reply, done=done)
        
    except Exception as e:
        logger.error(f"Unexpected error in chatbot endpoint: {e}")
        return BotResponse(
            sessionId=req.sessionId or "",
            botMessage="عذراً، حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.",
            done=True
        )

@app.get("/health")
def health_check():
    """
    فحص صحة الخدمة
    """
    return {
        "status": "healthy",
        "service": "Eastern Province Taxi Booking",
        "version": "2.0.0",
        "active_sessions": len(sessions)
    }

@app.get("/supported-cities")
def get_supported_cities():
    """
    إرجاع قائمة المدن المدعومة
    """
    return {
        "eastern_province_cities": EASTERN_PROVINCE_CITIES,
        "total_cities": len(EASTERN_PROVINCE_CITIES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
