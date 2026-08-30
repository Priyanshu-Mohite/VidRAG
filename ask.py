import os
import cohere
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# APIs & Clients initialize
co = cohere.Client(os.environ.get("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Hybrid Search ke liye Keyword (BM25) model load kar rahe hain (Local chalega, no API cost)
print("Loading Keyword Search Model...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

COLLECTION_NAME = "dsa_lectures_hybrid"
LLM_MODEL = "openai/gpt-oss-120b"

def search_and_answer(query):
    print("\n🔍 Tera sawal Cohere (Vector) aur BM25 (Keyword) dono me convert ho raha hai...")
    
    # 1. DENSE VECTOR (Cohere - Meaning ke liye)
    query_vector = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # 2. SPARSE VECTOR (BM25 - Exact Keyword ke liye)
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_query = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )

    print("📚 Qdrant DB se Hybrid matches (Meaning + Exact Words) dhoondh rahe hain...")
    
    # 3. HYBRID SEARCH (Prefetch & Fusion)
    # 3. HYBRID SEARCH (Prefetch & Fusion)
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            # Vector search se top 10 nikalo (using hata diya taaki default unnamed vector use ho)
            models.Prefetch(query=query_vector, limit=10),
            
            # Keyword search se top 10 nikalo (ye "sparse" naam se hi save kiya tha)
            models.Prefetch(query=sparse_query, using="sparse", limit=10),
        ],
        # Dono ko mix karo RRF (Reciprocal Rank Fusion) se aur best 5 do
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=5
    ).points

    # ⚠️ NOTE ON THRESHOLD: Jab dono mix hote hain, toh Qdrant score 0.5 (cosine) nahi deta. 
    # Wo rank ke hisaab se chota score deta hai. Isliye yahan hum strict threshold nahi laga rahe.
    # Humara LLM ka Prompt hi usko "Refuse" karne me help karega agar chunk irrelevant hai.
    
    if not search_results:
        print("\n❌ REFUSAL: Koi relevant chunks nahi mile.")
        return

    print(f"🎯 {len(search_results)} hybrid chunks mil gaye! Ab LLM ko prompt bhej rahe hain...\n")

    # 4. Context build karna
    context_text = ""
    for i, chunk in enumerate(search_results):
        payload = chunk.payload
        context_text += f"\n--- Chunk {i+1} (Video ID: {payload['video_id']}, Time: {payload['start_time']}s) ---\n"
        context_text += payload['text'] + "\n"

    # 5. Strict LLM Prompt
    # 5. Smart LLM Prompt (UPDATED)
    prompt = f"""
    Tu ek expert DSA Teaching Assistant hai. Tera kaam students ke doubts solve karna hai aur unhe sahi lectures ki taraf guide karna hai.
    Neeche kuch video transcripts ka context diya gaya hai.

    RULES:
    1. Agar student koi technical concept puchta hai, toh context ko use karke samjha de.
    2. Agar student videos ya topics suggest karne ko bolta hai (jaise "DP videos suggest kar"), aur context me wo topic available hai, toh refuse mat karna. Usko bolna: "Haan bhai, Pratyush bhai ne in videos me ye topic mast cover kiya hai. Aap neeche diye gaye links check kar sakte ho."
    3. Apne mann se koi technical gyan mat pelna.
    4. Agar context me topic se related KUCH BHI nahi hai, SIRF TABHI exact ye line bolna: "Sorry, Pratyush bhai ne iske baare me specifically nahi bataya hai."
    5. Output ekdum clean, point-wise aur SIRF "Hinglish" (Roman characters) me hona chahiye. Hindi script use mat karna.
    6. IMPORTANT: Tu jis context chunk se answer nikal raha hai, us chunk ka 'Video ID' apne answer me zaroor mention karna.

    7."CRITICAL RULE: Agar student koi code likhne ya problem solve karne bole, aur uska EXACT solution in transcripts me nahi hai, toh tu apne mann se ek line ka code nahi likhega. Seedha refuse karega."

    8."Tujhe strictly sirf utna hi bolna hai jitna context me hai. Apni pre-trained knowledge ka 1% bhi use nahi karna hai."

    CONTEXT:
    {context_text}

    STUDENT KA SAWAL: {query}
    """

    # 6. LLM API Call
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful DSA AI Assistant. Respond in natural, helpful language."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2 
    )

    answer = response.choices[0].message.content
    print("🤖 RAG ENGINE KA FINAL ANSWER:\n")
    print("="*60)
    print(answer)
    print("="*60 + "\n")
    
    # 7. THE GROUNDING GUARD (Is baar ye zaroor daalna!)
    # 7. THE GROUNDING GUARD & SMART CITATION
    if "Sorry, Pratyush bhai" in answer:
        pass
    else:
        print("📺 Video References (Yahan click karke exact video section dekh):")
        unique_videos = set()
        for chunk in search_results:
            v_id = chunk.payload['video_id']
            
            # ASLI FIX YAHAN HAI: Agar video_id LLM ke answer me hai, tabhi link banao
            if v_id in answer:
                start_t = max(0, int(float(chunk.payload['start_time'])) - 5) 
                unique_videos.add(f"https://youtu.be/{v_id}?t={start_t}")
                
        # Agar filtering ke baad koi link bacha hai toh hi print karo
        if unique_videos:
            for link in unique_videos:
                print(f"👉 {link}")
        else:
            # Fallback (Agar LLM ne ID nahi likha galti se, toh sabse top wala 1st link de do)
            top_v_id = search_results[0].payload['video_id']
            top_start_t = max(0, int(float(search_results[0].payload['start_time'])) - 5)
            print(f"👉 https://youtu.be/{top_v_id}?t={top_start_t}")

if __name__ == "__main__":
    print("🚀 RAG Engine Active! Terminate karne ke liye 'exit' likhna.")
    while True:
        user_query = input("\n🤔 Apna DSA ka sawal puch: ")
        if user_query.strip().lower() in ['exit', 'quit']:
            print("Chalte hain bhai!")
            break
        if user_query.strip():
            search_and_answer(user_query)