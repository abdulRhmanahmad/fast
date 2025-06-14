import pinecone
from openai import OpenAI
import os
import time

# === 🛠️ مفاتيح API (يفضل استخدام Environment Variables في Render) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")  # مثلاً: gcp-starter أو us-east4-gcp

# === 🔗 ربط OpenAI و Pinecone ===
client = OpenAI(api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "places-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)
index = pinecone.Index(index_name)

# === 🔤 دالة توليد Embedding لأي نص ===
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

# === 📍 قائمة الأماكن (اختصرنا، فيك تكمل) ===
known_places_embedding = {
    "المزة": "المزة، دمشق، سوريا",
    "باب توما": "باب توما، دمشق، سوريا",
    "حي الشعلان": "حي الشعلان، دمشق، سوريا",
    "سوق الحميدية": "سوق الحميدية، دمشق، سوريا",
    "شارع بغداد": "شارع بغداد، دمشق، سوريا",
    "كفرسوسة": "كفرسوسة، دمشق، سوريا",
    "مزة جبل": "مزة جبل، دمشق، سوريا",
    "مشروع دمر": "مشروع دمر، دمشق، سوريا",
    "حي القصور": "حي القصور، دمشق، سوريا",
    "ساحة الأمويين": "ساحة الأمويين، دمشق، سوريا"
    # 🔁 بإمكانك إضافة باقي الأماكن من القائمة اللي جهزناها سابقًا
}

# === ⬆️ رفع الأماكن إلى Pinecone ===
for name, address in known_places_embedding.items():
    try:
        vector = get_embedding(name)
        index.upsert([(f"place-{name}", vector, {"address": address})])
        print(f"✅ تم رفع: {name}")
        time.sleep(0.5)  # تأخير بسيط لراحة الـ API
    except Exception as e:
        print(f"❌ خطأ في {name}: {e}")
