import os
import cohere
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

app = FastAPI(title="DSA RAG Engine API")

# Frontend (React) ko allow karne ke liye CORS middleware lagaya hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production me isko apne localhost:5173 ya domain se replace karenge
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys aur Clients ka setup
co = cohere.Client(os.environ.get("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading Keyword Search Model for API...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# COLLECTION_NAME = "dsa_lectures_hybrid"
COLLECTION_NAME = "dsa_lectures_clean"

# LLM_MODEL = "llama3-70b-8192" # Groq ka supported model
LLM_MODEL = "openai/gpt-oss-120b" 

# Pydantic Schema (Input data validate karne ke liye)
class QueryRequest(BaseModel):
    question: str

# Ye apna naya POST route hai jo frontend call karega
@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    query = request.question
    original_query = request.question
    
    try:
        print(f"🤔 Original Query: {original_query}")
        
        rewrite_prompt = f"""
        You are a search query optimizer. 
        Translate the following Hinglish/Hindi query to pure English. 
        Fix any spelling mistakes in technical terms (e.g., 'ransome' -> 'ransom').
        Return ONLY the translated english search query string. 
        Do NOT add any extra words, parentheses, brackets, or conversational text at all.
        Query: '{original_query}'
        """
        
        rewrite_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b", # Groq ka sabse fast model pre-processing ke liye
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1
        )
        
        search_query = rewrite_response.choices[0].message.content.strip().strip('"\'')
        print(f"✨ Optimized Search Query: {search_query}")
        # 1. DENSE VECTOR (Cohere se)
        query_vector = co.embed(
            texts=[search_query],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        ).embeddings[0]

        # 2. SPARSE VECTOR (BM25 se)
        sparse_result = list(sparse_model.embed([search_query]))[0]
        sparse_query = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist()
        )

        # 3. HYBRID SEARCH (RRF Fusion ke sath)
        # search_results = qdrant.query_points(
        #     collection_name=COLLECTION_NAME,
        #     prefetch=[
        #         models.Prefetch(query=query_vector, limit=10),
        #         models.Prefetch(query=sparse_query, using="sparse", limit=10),
        #     ],
        #     query=models.FusionQuery(fusion=models.Fusion.RRF),
        #     with_payload=True,
        #     limit=5
        # ).points

        # if not search_results:
        #     return {"answer": "Sorry, Pratyush bhai ne iske baare me specifically nahi bataya hai. (Syllabus ke bahar ka sawal)", "links": []}

        # 3. HYBRID SEARCH (RRF Fusion ke sath)
        # raw_search_results = qdrant.query_points(
        #     collection_name=COLLECTION_NAME,
        #     prefetch=[
        #         models.Prefetch(query=query_vector, limit=10),
        #         models.Prefetch(query=sparse_query, using="sparse", limit=10),
        #     ],
        #     query=models.FusionQuery(fusion=models.Fusion.RRF),
        #     with_payload=True,
        #     limit=5
        # ).points

        # # 🚀 NAYA FIX: RRF Score Threshold Filter (Kachra hatane ke liye)
        # RRF_THRESHOLD = 0.015 # Isko tune kar sakta hai baad me logs dekh kar
        # search_results = []
        
        # print("\n📊 RRF Scores for retrieved chunks:")
        # for result in raw_search_results:
        #     print(f"Video: {result.payload['video_id']}, Score: {result.score:.4f}")
        #     if result.score >= RRF_THRESHOLD:
        #         search_results.append(result)

        # # Agar score threshold pass karne wala ek bhi chunk nahi bacha
        # if not search_results:
        #     return {"answer": "Sorry, Pratyush bhai ne iske baare me specifically nahi bataya hai. (Koi relevant chunk nahi mila)", "links": []}

        # 3. PURE VECTOR SEARCH WITH STRICT SCORE THRESHOLD
        # ---------------------------------------------------------
        # PHASE 1: TWO-STAGE RETRIEVAL (BROAD FETCH + RERANK)
        # ---------------------------------------------------------
        # print("🔍 Stage 1: Fetching top 15 broad chunks from Qdrant...")
        
        # # 1. BROAD FETCH: Qdrant se top 15 chunks uthao (bina strict threshold ke)
        # raw_search_results = qdrant.query_points(
        #     collection_name=COLLECTION_NAME,
        #     query=query_vector, 
        #     limit=15 
        # ).points

        # ---------------------------------------------------------
        # PHASE 1: TWO-STAGE RETRIEVAL (HYBRID FETCH + RERANK)
        # ---------------------------------------------------------
        print("🔍 Stage 1: Fetching top 15 chunks using HYBRID Search (BM25 + Cohere)...")
        
        # 1. BROAD HYBRID FETCH: Qdrant se RRF (Reciprocal Rank Fusion) use karke top 15 chunks uthao
        raw_search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # Meaning samajhne ke liye Cohere Vector
                models.Prefetch(query=query_vector, limit=15),
                
                # Exact spelling/keyword pakadne ke liye BM25 Sparse Vector
                models.Prefetch(query=sparse_query, using="sparse", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), # Dono ke results ko mix kar dega
            with_payload=True,
            limit=15
        ).points

        if not raw_search_results:
            return {"answer": "Sorry, Pratyush bhai ne iske baare me kuch mention nahi kiya hai.", "links": []}

        print(f"📦 Got {len(raw_search_results)} chunks. Sending to Cohere Rerank...")

        # 2. EXTRACT TEXTS: Reranker ko bhejne ke liye sirf text chahiye
        docs_to_rerank = [chunk.payload['text'] for chunk in raw_search_results]

        # 3. COHERE RERANK CALL: Ye identify karega ki actual answer kisme hai
        rerank_response = co.rerank(
            model="rerank-multilingual-v3.0", # Hinglish/English ke liye best
            query=search_query, # Jo user ne pucha hai
            documents=docs_to_rerank,
            top_n=4 # Hum LLM ko sirf top 4 denge
        )

        # 4. STRICT FILTERING (Passing Mention ko yahan maarenge)
        RERANK_THRESHOLD = 0.5 # Is score ke neeche matlab kachra/passing mention
        search_results = [] # Ye hamare final chunks honge
        
        print("\n📊 Cohere Rerank Scores:")
        for result in rerank_response.results:
            score = result.relevance_score
            original_chunk = raw_search_results[result.index]
            
            if score >= RERANK_THRESHOLD:
                search_results.append(original_chunk)
                print(f"✅ PASS -> Score: {score:.4f} | Video: {original_chunk.payload['video_id']}")
            else:
                print(f"❌ FAIL (Passing Mention) -> Score: {score:.4f} | Video: {original_chunk.payload['video_id']}")

        # Agar Reranker ne saare chunks reject kar diye (matlab sirf passing mentions the)
        if not search_results:
            print("🚫 All chunks rejected by Reranker. Returning Not Found.")
            return {
                "answer": "Sorry bhai, Pratyush ne in videos me is topic ka naam zaroor liya hai, par usko detail me padhaya nahi hai.", 
                "links": []
            }

        print(f"🎯 Reranker ne {len(search_results)} solid chunks LLM ke liye approve kiye!\n")
        # ---------------------------------------------------------

        # 4. Context string build karna
        context_text = ""
        for i, chunk in enumerate(search_results):
            payload = chunk.payload
            context_text += f"\n--- Chunk {i+1} (Video ID: {payload['video_id']}, Time: {payload['start_time']}s) ---\n"
            context_text += payload['text'] + "\n"

        # 5. LLM Prompt (Ekdum strict Hinglish rules ke sath)
        # 5. LLM Prompt (Updated with Rule 7 & 8 from ask.py)
        # prompt = f"""
        # Tu ek expert DSA Teaching Assistant hai. Tera kaam students ke doubts solve karna hai aur unhe sahi lectures ki taraf guide karna hai.
        # Neeche kuch video transcripts ka context diya gaya hai.

        # RULES:
        # 1. Agar student koi technical concept puchta hai, toh context ko use karke samjha de.
        # 2. Agar student videos ya topics suggest karne ko bolta hai, aur context me wo topic available hai, toh refuse mat karna. Usko bolna: "Haan bhai, Pratyush bhai ne in videos me ye topic mast cover kiya hai. Aap neeche diye gaye links check kar sakte ho."
        # 3. Apne mann se koi technical gyan mat pelna.
        # 4. Agar context me topic se related KUCH BHI nahi hai, SIRF TABHI exact ye line bolna: "Sorry, Pratyush bhai ne iske baare me specifically nahi bataya hai."
        # 5. Output ekdum clean, point-wise aur SIRF "Hinglish" (Roman characters) me hona chahiye. Hindi script use mat karna.
        # 6. IMPORTANT: Tu jis context chunk se answer nikal raha hai, us chunk ka 'Video ID' apne answer me zaroor mention karna.
        # 7. CRITICAL RULE: Agar student koi code likhne ya problem solve karne bole, aur uska EXACT solution in transcripts me nahi hai, toh tu apne mann se ek line ka code nahi likhega. Seedha refuse karega.
        # 8. Tujhe strictly sirf utna hi bolna hai jitna context me hai. Apni pre-trained knowledge ka 1% bhi use nahi karna hai.
        # 9. SUPER CRITICAL FILTERING RULE: Agar kisi chunk me user ka pucha gaya topic sirf ek "passing mention" hai (jaise "jaise humne ransom note me kiya tha") aur wahan topic ko actually explain, discuss ya code nahi kiya gaya hai, toh us chunk ko POORI TARAH IGNORE kar de. Sirf wahi chunk use kar jahan proper padhaya/explain kiya gaya ho. Agar saare chunks passing mention wale hain, toh tu rule 4 follow karke 'Sorry' bol dega.

        # CONTEXT:
        # {context_text}

        # STUDENT KA SAWAL: {query}
        # """

        # # 6. Groq (LLM) API Call (Same as before)
        # # 6. Groq (LLM) API Call
        # response = groq_client.chat.completions.create(
        #     model=LLM_MODEL,
        #     messages=[
        #         {
        #             "role": "system", 
        #             "content": "You are a STRICT and ruthless DSA Teaching Assistant. Your job is to filter out garbage. If the provided context only contains a 'passing mention', a 'brief example', or just the name of the topic WITHOUT any actual deep explanation, algorithm discussion, or code walkthrough, YOU MUST REJECT IT completely and reply EXACTLY with: 'Sorry, Pratyush bhai ne iske baare me specifically nahi bataya hai.' Do not try to be helpful if the context is weak."
        #         },
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.0 # Creativity bilkul 0 kar de taaki strictly rule follow kare
        # )

        # answer = response.choices[0].message.content

        # ---------------------------------------------------------
        # PHASE 3: THE CALM & SMART PROMPT
        # ---------------------------------------------------------
        
        # 5. LLM Prompt (Relaxed & Helpful)
    #     prompt = f"""
    #     Tu ek helpful aur expert DSA Teaching Assistant hai.
    #     Neeche kuch filtered aur highly relevant video transcripts ka context diya gaya hai.

    #     RULES FOR ANSWERING:
    #     1. Context ka use karke student ke sawal ka detail me answer de.
    #     2. Pura answer SIRF "Hinglish" (Roman characters) me hona chahiye. Devnagari script (Hindi) bilkul use mat karna.
    #     3. Agar student code mangta hai aur context me exact code nahi hai, par logic explain kiya gaya hai, toh logic samjha de. Saath me politely bol de ki "Pratyush bhai ne is video me exact code nahi likha hai, par logic ye hai..." Apne mann se naya code mat generate karna.
    #     4. Tu jis bhi chunk se information nikal raha hai, uska 'Video ID' answer me zaroor mention kar (e.g., "Video [video_id] me bataya gaya hai ki...").
    #     5. Apni external knowledge ka use mat karna. Sirf context par rely kar.

    #     CONTEXT:
    #     {context_text}

    #     STUDENT KA SAWAL: {query}
    #     """

    #     # 6. Groq (LLM) API Call
    #     response = groq_client.chat.completions.create(
    #         model=LLM_MODEL,
    #         messages=[
    #             {
    #                 "role": "system", 
    #                 "content": "You are a helpful and clear DSA AI Teaching Assistant. Your tone is encouraging and technical. Answer questions accurately based ONLY on the provided context."
    #             },
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.2 # 0.0 se badha kar 0.2 kiya hai taaki answer thoda natural aur human-like sound kare, machine jaisa nahi.
    #     )

    #     answer = response.choices[0].message.content

    #     # 7. Grounding Guard (Asli Fix - video_id in answer wala check lagana hai)
    #     unique_links = []
    #     if "Sorry, Pratyush bhai" not in answer:
    #         unique_videos = set()
    #         for chunk in search_results:
    #             v_id = chunk.payload['video_id']
                
    #             # Ye check missing tha teri API me!
    #             if v_id in answer:
    #                 start_t = max(0, int(float(chunk.payload['start_time'])) - 5) 
    #                 unique_videos.add(f"https://youtu.be/{v_id}?t={start_t}")
            
    #         unique_links = list(unique_videos)

    #     # JSON response bhej rahe hain apne frontend ko
    #     return {"answer": answer, "links": unique_links}

    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     raise HTTPException(status_code=500, detail="Internal Server Error")

    # ---------------------------------------------------------
        # PHASE 3: THE CALM & SMART PROMPT
        # ---------------------------------------------------------
        
        # 5. LLM Prompt (Relaxed, Helpful & CONCISE)

        prompt = f"""
        Tu ek helpful aur precise DSA AI Assistant hai. Tera kaam students ke doubts ko strictly diye gaye CONTEXT ke basis par solve karna hai.

        RULES FOR ANSWERING:
        1. CONTEXT STRICTNESS: Sirf context me di gayi information use kar. Agar user ka exact topic context me nahi hai, toh apni pre-trained knowledge use mat kar.
        2. ZERO CODE GENERATION: Agar exact code context transcripts me explicitly nahi bola gaya hai, toh apne dimaag se ek line ka code bhi mat likhna.
        3. EXACT REFUSAL: Agar context me answer nahi hai, toh apna answer EXACTLY in shabdo se shuru kar: "Sorry, Pratyush bhai ne nahi padhaya hai".
        4. CLEAN OUTPUT: Answer strictly Hinglish (Roman characters) me de aur lamba essay mat likh.
        5. CITATION FORMAT: Tu jis Chunk se information le raha hai, sentence ke end me uska index number strictly aise likhna: [1] ya [2]. Koi extra space ya fancy brackets mat lagana. Iske andar koi extra space ya fancy brackets (jaise 【 1 】) bhool kar bhi use mat karna.

        CONTEXT:
        {context_text}

        STUDENT KA SAWAL: {query}
        """

        # 6. Groq (LLM) API Call
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful and clear DSA AI Teaching Assistant. Keep your explanations concise and structured."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 
        )

        answer = response.choices[0].message.content

        # ---------------------------------------------------------
        # PHASE 4: SMART LINK GENERATION (THE FIX)
        # ---------------------------------------------------------
        # unique_links = []
        # if "Sorry, Pratyush bhai" not in answer:
        #     # Ek dictionary banate hain jisme video_id ke sath uska sabse EARLIEST timestamp store hoga
        #     video_timestamps = {}
            
        #     for chunk in search_results:
        #         v_id = chunk.payload['video_id']
                
        #         # Check karo agar LLM ne actually ye video ID use kiya hai
        #         if v_id in answer:
        #             start_t = max(0, int(float(chunk.payload['start_time'])) - 5)
                    
        #             # Agar ye video already list me hai, toh sabse PEECHE wala (earliest) time rakho!
        #             # Taaki student video beech se na dekhe, shuru se concept samjhe.
        #             if v_id in video_timestamps:
        #                 video_timestamps[v_id] = min(video_timestamps[v_id], start_t)
        #             else:
        #                 video_timestamps[v_id] = start_t
                        
        #     # Dictionary se final links bana lo (Sirf Top 1 ya 2 links bhejenge taaki React UI clean rahe)
        #     for v_id, start_t in list(video_timestamps.items())[:4]: 
        #         unique_links.append(f"https://youtu.be/{v_id}?t={start_t}")
                
        #     # Fallback: Agar LLM ne ID cite karna miss kar diya, toh reranker ka number 1 chunk bhej do
        #     if not unique_links and search_results:
        #         top_chunk = search_results[0]
        #         v_id = top_chunk.payload['video_id']
        #         start_t = max(0, int(float(top_chunk.payload['start_time'])) - 5)
        #         unique_links.append(f"https://youtu.be/{v_id}?t={start_t}")

        # ---------------------------------------------------------
        # PHASE 4: SMART LINK GENERATION (THE FIX)
        # ---------------------------------------------------------
        unique_links = []
        if "Sorry, Pratyush bhai" not in answer:
            
            # 1. Regex se answer ke andar se saare brackets wale numbers nikal lo (jaise ['1', '2'])
            used_indices_str = re.findall(r"[\[【]\s*(\d+)\s*[\]】]", answer)
            
            # 2. String numbers ko integer me convert karo aur set use karke duplicate hata do
            unique_indices = set([int(idx) for idx in used_indices_str])
            
            video_timestamps = {}
            
            for idx in unique_indices:
                # LLM ne Chunk 1, Chunk 2 padha hai (1-based), par Python ki list 0 se shuru hoti hai (0-based)
                # Toh Chunk 1 ka matlab search_results[0]
                array_index = idx - 1 
                
                # Check karenge ki index valid hai ya nahi (safety guard)
                if 0 <= array_index < len(search_results):
                    chunk = search_results[array_index]
                    v_id = chunk.payload['video_id']
                    start_t = max(0, int(float(chunk.payload['start_time'])) - 5)
                    
                    # Earliest timestamp save karne wala tera purana mast logic
                    if v_id in video_timestamps:
                        video_timestamps[v_id] = min(video_timestamps[v_id], start_t)
                    else:
                        video_timestamps[v_id] = start_t
                        
            # Dictionary se final links bana lo (Top 4 links)
            for v_id, start_t in list(video_timestamps.items())[:4]: 
                unique_links.append(f"https://youtu.be/{v_id}?t={start_t}")
                
            # Fallback: Agar LLM ne bracket lagana miss kar diya, toh top chunk bhej do (Safety net)
            if not unique_links and search_results:
                top_chunk = search_results[0]
                v_id = top_chunk.payload['video_id']
                start_t = max(0, int(float(top_chunk.payload['start_time'])) - 5)
                unique_links.append(f"https://youtu.be/{v_id}?t={start_t}")

        # JSON response bhej rahe hain frontend ko
        return {"answer": answer, "links": unique_links}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")