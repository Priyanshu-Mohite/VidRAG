import os
import json
import glob
import uuid
import cohere
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()

# Clients initialize kar rahe hain
co = cohere.Client(os.environ.get("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

# Hybrid search ke liye Sparse (BM25) model load kar rahe hain
print("Loading BM25 Sparse Model...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Collection ka naam change kar de taaki purane wale se clash na ho
# COLLECTION_NAME = "dsa_lectures_hybrid"

COLLECTION_NAME = "dsa_lectures_clean"

CHUNK_SEC = 75.0
OVERLAP_SEC = 15.0


def is_valid_chunk(text, min_words=5, repetition_threshold=0.5):
    """
    Kachra aur repeated Whisper hallucinations filter karne ka bouncer.
    """
    words = text.lower().split()
    total_words = len(words)
    
    # 1. Chote chunks (empty ya sirf 'okay') ko reject karo
    if total_words < min_words:
        return False
        
    # 2. Loop-hallucination check (Repetitive words)
    unique_words = len(set(words))
    # Agar unique words total words ke 50% se bhi kam hain, matlab wo ek hi baat repeat kar raha hai
    if total_words > 15 and (unique_words / total_words) < repetition_threshold:
        return False
        
    return True

def setup_qdrant():
    collections = qdrant.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"🛠️ Creating new HYBRID Qdrant collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            # 1. DENSE CONFIG (Cohere)
            vectors_config=models.VectorParams(
                size=1024, 
                distance=models.Distance.COSINE
            ),
            # 2. SPARSE CONFIG (BM25) - Ye miss kiya tha tune
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
    else:
        print(f"✅ Collection {COLLECTION_NAME} already exists.")

# def time_based_chunking(segments):
    # if not segments:
    #     return []

    # chunks = []
    # video_end_time = segments[-1]['end']
    # current_start = 0.0

    # while current_start < video_end_time:
    #     current_end = current_start + CHUNK_SEC
    #     chunk_texts = []
    #     actual_chunk_start = None

    #     for seg in segments:
    #         if seg['start'] < current_end and seg['end'] > current_start:
    #             if actual_chunk_start is None:
    #                 actual_chunk_start = seg['start'] 
    #             chunk_texts.append(seg['text'].strip())

    #     if chunk_texts:
    #         chunks.append({
    #             "start_time": actual_chunk_start,
    #             "text": " ".join(chunk_texts).strip()
    #         })
    #     current_start += (CHUNK_SEC - OVERLAP_SEC)
    # return chunks


# def time_based_chunking(segments):
    # if not segments:
    #     return []

    # chunks = []
    # video_end_time = segments[-1]['end']
    # current_start = 0.0

    # while current_start < video_end_time:
    #     current_end = current_start + CHUNK_SEC
    #     chunk_texts = []
    #     actual_chunk_start = None

    #     for seg in segments:
    #         if seg['start'] < current_end and seg['end'] > current_start:
    #             if actual_chunk_start is None:
    #                 actual_chunk_start = seg['start'] 
    #             chunk_texts.append(seg['text'].strip())

    #     if chunk_texts:
    #         merged_text = " ".join(chunk_texts).strip()
            
    #         # 🚀 NAYA BOUNCER YAHAN LAGA HAI
    #         if is_valid_chunk(merged_text):
    #             chunks.append({
    #                 "start_time": actual_chunk_start,
    #                 "text": merged_text
    #             })
    #         else:
    #             pass # Kachra chunk ignore kar diya
                
    #     current_start += (CHUNK_SEC - OVERLAP_SEC)
    # return chunks

def time_based_chunking(segments):
    if not segments:
        return []

    chunks = []
    # segments array se last item ka 'end' time nikal rahe hain
    video_end_time = segments[-1]['end']
    current_start = 0.0

    while current_start < video_end_time:
        current_end = current_start + CHUNK_SEC
        chunk_texts = []
        actual_chunk_start = None

        for seg in segments:
            if seg['start'] < current_end and seg['end'] > current_start:
                seg_text = seg['text'].strip()
                
                # 🚀 THE REAL FIX: Kachra check sirf ek choti line (segment) par hoga!
                words = seg_text.lower().split()
                # Agar ek choti line me bohot zyada repetition hai (Whisper bug), toh sirf us line ko ignore karo
                if len(words) > 8 and (len(set(words)) / len(words)) < 0.4:
                    continue # Sirf ye kachra line drop hogi, poora video nahi
                
                if actual_chunk_start is None:
                    actual_chunk_start = seg['start'] 
                chunk_texts.append(seg_text)

        if chunk_texts:
            chunks.append({
                "start_time": actual_chunk_start,
                "text": " ".join(chunk_texts).strip()
            })
            
        current_start += (CHUNK_SEC - OVERLAP_SEC)
        
    return chunks

def process_and_upsert():
    setup_qdrant()
    
    json_files = glob.glob("transcripts/*.json")
    print(f"📁 Found {len(json_files)} transcripts to index.")

    indexed_log_file = "indexed_videos_hybrid.txt"
    if os.path.exists(indexed_log_file):
        with open(indexed_log_file, "r") as f:
            indexed_videos = set(f.read().splitlines())
    else:
        indexed_videos = set()

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        video_id = data.get("video_id")
        video_title = data.get("title", "DSA Lecture")
        
        if video_id in indexed_videos:
            print(f"⏭️ Skipping {video_id}, already indexed!")
            continue

        segments = data.get("segments", [])
        if not segments:
            continue
            
        print(f"\n⚙️ Chunking video: {video_id}...")
        chunks = time_based_chunking(segments)
        # texts_to_embed = [c['text'] for c in chunks]
        texts_to_embed = [f"Title: {video_title}\n\n{c['text']}" for c in chunks]

        if not texts_to_embed:
            print(f"⚠️ Video {video_id} me koi solid technical chunk nahi mila (Ya kachra tha). Skipping API call!")
            continue # Loop ko yahi rok kar agle video par jao
        
        try:
            print(f"🧠 Generating DENSE (Cohere) and SPARSE (BM25) embeddings...")
            # Dense Vectors (Meaning)
            dense_response = co.embed(
                texts=texts_to_embed,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            )
            dense_embeddings = dense_response.embeddings
            
            # Sparse Vectors (Exact Keywords)
            sparse_embeddings = list(sparse_model.embed(texts_to_embed))

            points = []
            for i, chunk in enumerate(chunks):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{chunk['start_time']}"))
                
                # Qdrant me ab hum ek dict bhej rahe hain jisme dono vectors hain
                combined_vector = {
                    "": dense_embeddings[i], # Default dense vector
                    "sparse": models.SparseVector(
                        indices=sparse_embeddings[i].indices.tolist(),
                        values=sparse_embeddings[i].values.tolist()
                    )
                }
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=combined_vector,
                        payload={
                            "video_id": video_id,
                            "title": video_title,
                            "start_time": chunk["start_time"],
                            # "text": chunk["text"]
                            "text": texts_to_embed[i]
                        }
                    )
                )

            batch_size = 15
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
            
            with open(indexed_log_file, "a") as f:
                f.write(video_id + "\n")
            indexed_videos.add(video_id)
            
            print(f"✅ Video {video_id} indexed successfully with HYBRID vectors!")
            time.sleep(5)

        except Exception as e:
            print(f"❌ Error on video {video_id}: {e}")
            break

if __name__ == "__main__":
    process_and_upsert()