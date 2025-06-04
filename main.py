import os, json, asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ملف الحجوزات المؤقَّت
BOOKINGS_FILE = "bookings.json"
file_lock = asyncio.Lock()

# ───────── system prompt محسّن باللهجة الشامية ─────────
system_prompt = """
أنت "يا هو" - مساعد حجز تاكسي ذكي باللهجة الشامية السورية. دورك تنظيم المشاوير بطريقة احترافية ومريحة للزبون.

## شخصيتك:
- ودود ومهذب، بس مش مبالغ بالكلام
- تحكي باللهجة الشامية الطبيعية (مثل أهل دمشق)
- تركز على الموضوع وما تطول بردودك
- تتعامل بصبر مع الزبائن حتى لو كانوا مشوشين
- تتابع المحادثة بذكاء وما تعيد نفس السؤال

## خطوات العمل بالترتيب:
1. **بداية المحادثة**: إذا ما في محادثة سابقة، سلم وقول "أهلين، وين بدك تروح؟"
2. **الوجهة**: لما يقول الوجهة، اقبلها وقول "حلو، [الوجهة]" واسأل التفاصيل إذا لازم
3. **نقطة الانطلاق**: اسأل "ومن وين رح نجيك؟" 
4. **نوع السيارة**: اسأل "بتفضل سيارة عادية ولا VIP؟"
5. **وقت الانطلاق**: "بدك تطلع هلأ ولا بوقت معين؟"
6. **الأغاني**: "إيش نوع الموسيقا اللي بتحبها؟"
7. **تفاصيل إضافية**: "في أي طلبات خاصة؟"

## مهم جداً - تتبع المحادثة:
- إذا المستخدم جاوب على سؤال، انتقل للسؤال التالي
- ما تعيد نفس السؤال اللي المستخدم جاوب عليه
- إذا قال "التل" أو أي مكان، هاد جواب على سؤال الوجهة، كمل للسؤال التالي
- إذا قال مكان الانطلاق، كمل لنوع السيارة
- إذا قال نوع السيارة، كمل للوقت... وهكذا

## أمثلة على التفاعل الصحيح:
المستخدم: "بدي روح ع التل"
الرد: "حلو، التل. ومن وين رح نجيك؟"

المستخدم: "من الشام الجديدة"  
الرد: "تمام، من الشام الجديدة للتل. بتفضل سيارة عادية ولا VIP؟"

## التلخيص النهائي:
لما تخلص كل المعلومات، لخص هيك:
```
تمام، خلينا نراجع:
• الوجهة: [العنوان]
• من وين: [نقطة الانطلاق]
• نوع السيارة: [عادية/VIP]  
• وقت الانطلاق: [هلأ/وقت محدد]
• الأغاني: [النوع أو بدون]
• ملاحظات: [أي طلبات خاصة]

تم! رح أحجزلك وأخبرك لما تكون السيارة جاهزة ✅
```

## قواعد مهمة:
- اقرأ المحادثة كاملة قبل ما ترد
- ما تعيد أسئلة المستخدم جاوب عليها
- إذا سأل عن شي ما بيخص التاكسي، قول: "عفواً، أنا بس بساعد بحجز التاكسي"
- إذا المعلومات ناقصة، اسأل عن اللي ناقص بس
- ما تسأل أسئلة كتيرة مرة وحدة، سؤال سؤال

## أمثلة على الردود:
- بدل "يمكنني مساعدتك" → "قدر ساعدك"
- بدل "ما هي وجهتك" → "وين بدك تروح؟"
- بدل "هل تريد" → "بتريد؟" أو "بدك؟"
- بدل "شكراً لك" → "يسلمو" أو "تسلم"

تذكر: كن طبيعي وودود، واحرص تاخد كل المعلومات المطلوبة قبل ما تقول "تم!"
"""

# ───────── نماذج Pydantic ─────────
class ChatMessage(BaseModel):
    role: str     # "user" أو "assistant"
    content: str

class MessageRequest(BaseModel):
    user_id: str
    messages: list[ChatMessage]     # سجلُّ المحادثة كاملًا

class MessageResponse(BaseModel):
    response: str

# ───────── دوال مساعدة محسنة ─────────
def parse_summary(text: str) -> dict:
    """استخراج المعلومات من النص النهائي بطريقة أكثر دقة"""
    def grab(key):
        # البحث عن النص باللغة العربية
        patterns = [f"• {key}:", f"•{key}:", f"{key}:"]
        for pattern in patterns:
            idx = text.find(pattern)
            if idx != -1:
                start = idx + len(pattern)
                end = text.find("\n", start)
                result = text[start:end].strip(" :[]") if end != -1 else text[start:].strip(" :[]")
                return result if result and result != "-" else ""
        return ""
    
    return {
        "destination": grab("الوجهة") or grab("الوجهه"),
        "pickup_location": grab("من وين") or grab("نقطة الانطلاق") or grab("الانطلاق"),
        "car_type": grab("نوع السيارة") or grab("السيارة"),
        "ride_time": grab("وقت الانطلاق") or grab("الوقت"),
        "music": grab("الأغاني") or grab("الموسيقا"),
        "notes": grab("ملاحظات") or grab("طلبات خاصة")
    }

async def save_booking(data: dict) -> str | None:
    """حفظ الحجز مع معالجة أفضل للأخطاء"""
    async with file_lock:
        booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        record = {
            "booking_id": booking_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",  # إضافة حالة الحجز
            **data
        }
        
        try:
            bookings = []
            if os.path.exists(BOOKINGS_FILE):
                with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # تحقق من أن الملف ليس فارغ
                        bookings = json.loads(content)
            
            bookings.append(record)
            
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(bookings, f, ensure_ascii=False, indent=2)
            
            print(f"✅ تم حفظ الحجز: {booking_id}")
            return booking_id
            
        except json.JSONDecodeError as e:
            print(f"❌ خطأ في قراءة JSON: {e}")
            # إنشاء ملف جديد إذا كان فاسد
            with open(BOOKINGS_FILE, "w", encoding="utf-8") as f:
                json.dump([record], f, ensure_ascii=False, indent=2)
            return booking_id
            
        except Exception as exc:
            print(f"❌ خطأ في حفظ الحجز: {exc}")
            return None

# ───────── endpoint /chat محسن ─────────
@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    try:
        # إضافة معلومات السياق للـ system prompt
        enhanced_system = system_prompt + f"\n\nمعرف المستخدم: {req.user_id}\nالوقت الحالي: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        messages = [
            {"role": "system", "content": enhanced_system}
        ] + [m.model_dump() for m in req.messages]
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.3,  # تقليل العشوائية للحصول على ردود أكثر اتساقاً
            max_tokens=300,   # تحديد طول الرد
            messages=messages
        )
        
        reply = completion.choices[0].message.content
        
        # تحسين شرط حفظ الحجز
        booking_indicators = ["رح أحجزلك", "تم!", "✅"]
        if any(indicator in reply for indicator in booking_indicators):
            details = parse_summary(reply)
            # التأكد من وجود المعلومات الأساسية
            if details.get("destination"):
                details["user_id"] = req.user_id
                if (booking_id := await save_booking(details)):
                    reply += f"\n\n📱 رقم حجزك: {booking_id}"
        
        return MessageResponse(response=reply)
        
    except Exception as e:
        print(f"❌ خطأ في معالجة الرسالة: {e}")
        raise HTTPException(
            status_code=500, 
            detail="عذراً، صار خطأ تقني. جرب مرة تانية."
        )

# ───────── endpoint /bookings محسن ─────────
@app.get("/bookings", response_model=list[dict])
async def get_bookings():
    """جلب جميع الحجوزات مع معالجة الأخطاء"""
    try:
        if not os.path.exists(BOOKINGS_FILE):
            return []
        
        with open(BOOKINGS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
            
    except json.JSONDecodeError:
        print("❌ ملف الحجوزات تالف، سيتم إنشاء ملف جديد")
        return []
    except Exception as e:
        print(f"❌ خطأ في قراءة الحجوزات: {e}")
        raise HTTPException(status_code=500, detail="خطأ في جلب الحجوزات")

# ───────── endpoint إضافي لحالة الحجز ─────────
@app.get("/booking/{booking_id}")
async def get_booking_status(booking_id: str):
    """جلب حالة حجز معين"""
    try:
        bookings = await get_bookings()
        booking = next((b for b in bookings if b["booking_id"] == booking_id), None)
        
        if not booking:
            raise HTTPException(status_code=404, detail="الحجز غير موجود")
        
        return booking
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
