import os
import json
import glob
import uuid
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

import time

load_dotenv()

# Clients initialize kar rahe hain
co = cohere.Client(os.environ.get("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

COLLECTION_NAME = "dsa_lectures_1024"
CHUNK_SEC = 75.0
OVERLAP_SEC = 15.0

def setup_qdrant():
    """Qdrant me collection banayega agar pehle se nahi hai"""
    collections = qdrant.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"🛠️ Creating new Qdrant collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024, # Cohere multilingual ka dimension 1024 hota hai
                distance=models.Distance.COSINE
            )
        )
    else:
        print(f"✅ Collection {COLLECTION_NAME} already exists.")

def time_based_chunking(segments):
    """
    Kachra text-splitter ki jagah proper Time-Window chunking.
    Har chunk 75 seconds ka hoga, 15 seconds overlap ke sath.
    """
    if not segments:
        return []

    chunks = []
    video_end_time = segments[-1]['end']
    current_start = 0.0

    while current_start < video_end_time:
        current_end = current_start + CHUNK_SEC
        chunk_texts = []
        actual_chunk_start = None

        for seg in segments:
            # Check agar segment is 75s window me aata hai
            if seg['start'] < current_end and seg['end'] > current_start:
                if actual_chunk_start is None:
                    actual_chunk_start = seg['start'] # Pehle word ka exact time
                chunk_texts.append(seg['text'].strip())

        if chunk_texts:
            chunks.append({
                "start_time": actual_chunk_start,
                "text": " ".join(chunk_texts).strip()
            })

        # Next window overlap ke baad se shuru hogi (e.g., 0-75, phir 60-135)
        current_start += (CHUNK_SEC - OVERLAP_SEC)

    return chunks

def process_and_upsert():
    setup_qdrant()
    
    json_files = glob.glob("transcripts/*.json")
    print(f"📁 Found {len(json_files)} transcripts to index.")

    # 🛑 JUGAD 1: Track indexed videos taaki wapas API calls waste na ho
    indexed_log_file = "indexed_videos.txt"
    if os.path.exists(indexed_log_file):
        with open(indexed_log_file, "r") as f:
            indexed_videos = set(f.read().splitlines())
    else:
        indexed_videos = set()

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        video_id = data.get("video_id")
        
        # Check agar pehle hi Qdrant me ja chuka hai toh SKIP maaro
        if video_id in indexed_videos:
            print(f"⏭️ Skipping {video_id}, already indexed!")
            continue

        segments = data.get("segments", [])
        if not segments:
            continue
            
        print(f"\n⚙️ Chunking video: {video_id}...")
        chunks = time_based_chunking(segments)
        print(f"✂️ Made {len(chunks)} chunks.")

        texts_to_embed = [c['text'] for c in chunks]
        
        try:
            print(f"🧠 Generating Cohere embeddings for {video_id}...")
            response = co.embed(
                texts=texts_to_embed,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            )
            embeddings = response.embeddings

            points = []
            for i, chunk in enumerate(chunks):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{chunk['start_time']}"))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[i],
                        payload={
                            "video_id": video_id,
                            "start_time": chunk["start_time"],
                            "text": chunk["text"]
                        }
                    )
                )

            print(f"📤 Upserting {len(points)} vectors to Qdrant in batches...")
            
            # QDRANT JUGAD: Ek saath sab bhejne ke bajaye 15-15 ke batch me bhejo
            batch_size = 15
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                print(f"   -> Batch {i//batch_size + 1} pushed successfully")
            
            # ✅ Success ke baad video_id ko txt file me save kar lo
            with open(indexed_log_file, "a") as f:
                f.write(video_id + "\n")
            indexed_videos.add(video_id)
            
            print(f"✅ Video {video_id} indexed successfully!")
            
            # Sleep ko wapas normal kar de, 70 seconds bohot zyada hai
            print("⏳ Sleeping for 5 seconds...")
            import time
            time.sleep(5)

        except Exception as e:
            print(f"❌ Error on video {video_id}: {e}")
            print("⏳ API limit hit ho gayi hai. Script ruk rahi hai. 1 minute baad wapas run karna, baaki bache hue videos wahin se resume honge!")
            break # Loop rok do taaki crash na ho


if __name__ == "__main__":
    process_and_upsert()
    print("\n🎉 Poori playlist ka vector database ready hai bhai!")