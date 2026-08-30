import os
import cohere
from qdrant_client import QdrantClient
# from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# 1. Sabhi APIs initialize kar lo
co = cohere.Client(os.environ.get("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)



# 🛑 Yahan dhyan de: OpenRouter ka setup kiya hai tera LLM call karne ke liye
# llm_client = OpenAI(
#     base_url="https://openrouter.ai/api/v1", 
#     api_key=os.environ.get("OPENROUTER_API_KEY") 
# )

COLLECTION_NAME = "dsa_lectures_1024"
LLM_MODEL = "openai/gpt-oss-120b" # Tera pasandida model

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key = GROQ_API_KEY)

# Qdrant Cosine Similarity me score 0 se 1 ke beech aata hai. 1.0 = Exact Match
# Ye raha apna Evaluation wala Threshold (Iske neeche wala kachra hai)
MIN_SCORE = 0.5 

def search_and_answer(query):
    print("\n🔍 Tera sawal Cohere ke paas vector banne jaa raha hai...")
    
    # Step 1: Sawal ko vector me convert karo (Cohere)
    query_vector = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings[0]

    print("📚 Qdrant DB se sabse best matches dhoondh rahe hain...")
    
    # Step 2: Vector ko Qdrant me search maaro
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        with_payload=True,
        limit=5, # Top 5 relevant chunks uthayenge
    ).points

    # Step 3: Threshold Filter (Precision maintain karne ke liye)
    relevant_chunks = []
    for result in search_results:
        # Check maar rahe hain ki kya answer strict criteria match kar raha hai
        if result.score >= MIN_SCORE:
            relevant_chunks.append(result)

    # Agar score 0.5 se upar ek bhi chunk nahi aaya, matlab out of syllabus
    if not relevant_chunks:
        print("\n❌ REFUSAL: Bhai, ye topic apne videos me cover nahi hua hai ya score bohot kam aaya. (Hallucination se bachne ke liye LLM call nahi kar rahe)")
        return

    print(f"🎯 {len(relevant_chunks)} solid chunks mil gaye! Ab LLM ko prompt bhej rahe hain...\n")

    # Step 4: Context build karna (Pura kachcha chitha)
    context_text = ""
    for i, chunk in enumerate(relevant_chunks):
        payload = chunk.payload
        context_text += f"\n--- Chunk {i+1} (Video ID: {payload['video_id']}, Time: {payload['start_time']}s) ---\n"
        context_text += payload['text'] + "\n"

    # Step 5: LLM ko strict prompt dena (The Prompt Engineering)
    prompt = f"""
    Tu ek expert DSA Teaching Assistant hai. Tera kaam students ke doubts solve karna hai.
    Neeche kuch video transcripts ka context diya gaya hai.
    Tujhe STRICTLY sirf is context ke basis par answer dena hai. Apne mann se koi technical gyan mat pelna.
    Agar context me answer nahi hai, toh seedha bol de: "Sorry, Pratyush bhai ne iske baare me specifically nahi bataya hai."

    ⚠️ CRITICAL RULE: Tera poora answer SIRF aur SIRF "Hinglish" me hona chahiye (English alphabets use karke). Hindi/Devanagari script (जैसे की ऐसे शब्द) ka use bhool kar bhi mat karna. Har ek word Roman letters me hi likhna. Output ekdum clean aur point-wise hona chahiye.

    CONTEXT:
    {context_text}

    STUDENT KA SAWAL: {query}
    """

    # LLM API Call
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful DSA AI Assistant. Respond in natural, helpful language."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2 # Ekdum kam temperature taaki point-to-point sateek answer de, creativity nahi chahiye idhar
    )

    answer = response.choices[0].message.content
    print("🤖 RAG ENGINE KA FINAL ANSWER:\n")
    print("="*60)
    print(answer)
    print("="*60 + "\n")
    
    # Step 6: Magic Timestamps (Clickable YouTube links banayenge)
    print("📺 Video References (Yahan click karke exact video section dekh):")
    unique_videos = set()
    for chunk in relevant_chunks:
        v_id = chunk.payload['video_id']
        # 5 second piche se start karenge taaki video start hone par context mil jaye
        start_t = max(0, int(chunk.payload['start_time']) - 5) 
        unique_videos.add(f"https://youtu.be/{v_id}?t={start_t}")
        
    for link in unique_videos:
        print(f"👉 {link}")

if __name__ == "__main__":
    print("🚀 RAG Engine Active! Terminate karne ke liye 'exit' likhna.")
    while True:
        user_query = input("\n🤔 Apna DSA ka sawal puch: ")
        if user_query.strip().lower() in ['exit', 'quit']:
            print("Chalte hain bhai!")
            break
        if user_query.strip():
            search_and_answer(user_query)