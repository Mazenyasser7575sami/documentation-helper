from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. بنعرف نفس المترجم الذكي اللي استخدمناه في الحفظ
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. بنفتح الفولدر اللي على الهارد ديسك
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 3. بنطلب من قاعدة البيانات تورينا الداتا اللي جواها (هنطلب أول 3 عناصر كمثال)
# الجملة دي بتجيب لنا البيانات المخزنة جوه Chroma
db_data = vectorstore.get(limit=3)

# 4. بنطبع النتائج بشكل نظيف عشان نشوفها
print(f"📊 إجمالي عدد القطع المخزنة في الـ DB حالياً: {len(vectorstore.get()['ids'])}")
print("\n--- عينة من البيانات المخزنة ---")

for i in range(len(db_data["ids"])):
    print(f"\n📍 قطعة رقم {i+1}:")
    print(f"🔗 المصدر (URL): {db_data['metadatas'][i]['source']}")
    print(
        f"📝 جزء من النص: {db_data['documents'][i][:200]}..."
    )  # بنطبع أول 200 حرف بس عشان الشاشة متبقاش زحمة
    print("-" * 30)
