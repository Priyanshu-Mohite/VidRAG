import os
import json
import asyncio
from pydub import AudioSegment
from pydub.silence import detect_silence
from groq import AsyncGroq

from dotenv import load_dotenv
load_dotenv() # Ye teri .env file ko read karke keys ko environment me daal dega

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

semaphore = asyncio.Semaphore(2)

TARGET_CHUNK_MS = 10 * 60 * 1000  # 10 mins

async def transcribe_chunk(chunk_path, offset_seconds, chunk_index, retries=4):
    """
    Groq API ko chunk bhejta hai.
    Rate limits ke liye exponential backoff ke sath.
    """

    async with semaphore:
        for attempt in range(retries):
            try:
                with open(chunk_path, "rb") as file:
                    transcription = await client.audio.transcriptions.create(
                        file=(chunk_path, file.read()),
                        model="whisper-large-v3",
                        prompt="Keep technical terms like memoization, tabulation, adjacency list, time complexity in English.",
                        response_format="verbose_json",
                        language="en"
                    )

                # JSON me timestamps ko offset se aage badhana
                for segment in transcription.segments:
                    segment["start"] += offset_seconds
                    segment["end"] += offset_seconds

                return transcription.segments

            except Exception as e:
                wait_time = 2 ** attempt
                print(
                    f"⚠️ Error on chunk {chunk_index} "
                    f"(Attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait_time}s..."
                )

                await asyncio.sleep(wait_time)

    print(f"❌ Chunk {chunk_index} totally failed.")
    return []

async def process_video(video_id, audio_filepath, output_dir="transcripts"):
    os.makedirs(output_dir, exist_ok=True)
    final_json_path = f"{output_dir}/{video_id}.json"
    temp_json_path = f"{output_dir}/{video_id}.tmp.json"

    if os.path.exists(final_json_path):
        return final_json_path

    print(f"🎧 Loading audio for {video_id}...")
    audio = AudioSegment.from_file(audio_filepath)
    
    tasks = []
    chunk_paths = []
    
    current_start = 0
    total_len = len(audio)
    chunk_index = 0

    print("✂️ Cutting audio intelligently at silence points (with Dynamic Fallbacks)...")
    
    while current_start < total_len:
        expected_end = current_start + TARGET_CHUNK_MS
        
        if expected_end >= total_len:
            actual_end = total_len
        else:
            # Smart Search: Last 2 minutes (8th min se 10th min tak) check karenge
            search_start = expected_end - (120 * 1000) 
            search_window = audio[search_start:expected_end]
            
            # TIER 1: Strict silence (500ms lamba, proper shanti)
            silences = detect_silence(search_window, min_silence_len=500, silence_thresh=audio.dBFS-14)
            
            if not silences:
                # TIER 2: Relaxed silence (250ms lamba, thodi background noise allowed)
                # Ye teacher ke quick saans lene wale points pakdega
                silences = detect_silence(search_window, min_silence_len=250, silence_thresh=audio.dBFS-10)
            
            if not silences:
                # TIER 3: Extreme fallback (100ms micro-pause)
                silences = detect_silence(search_window, min_silence_len=100, silence_thresh=audio.dBFS-8)
                
            if silences:
                last_silence_end = silences[-1][1]
                actual_end = search_start + last_silence_end
            else:
                # Agar 2 minute me ek baar bhi 100ms ka pause nahi liya, jo scientifically impossible hai speech me, 
                # tabhi jaake hard cut lagega (Just for code safety)
                actual_end = expected_end

        offset_seconds = current_start / 1000.0
        chunk = audio[current_start:actual_end]
        
        chunk_path = f"{output_dir}/{video_id}_chunk_{chunk_index}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="64k")
        chunk_paths.append(chunk_path)
        
        # Tuple me (chunk_index) bhi pass kar rahe hain logging ke liye
        tasks.append(transcribe_chunk(chunk_path, offset_seconds, chunk_index))
        
        current_start = actual_end 
        chunk_index += 1

    print(f"🚀 Sending {len(tasks)} chunks to Groq API concurrently...")
    results = await asyncio.gather(*tasks)

    # Empty results filter karna (in case of total chunk failure)
    all_segments = []
    for segments in results:
        if segments:
            all_segments.extend(segments)

    final_data = {
        "video_id": video_id,
        "segments": all_segments
    }

    print("💾 Atomically saving final JSON...")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    os.replace(temp_json_path, final_json_path)

    print("🧹 Cleaning up chunks...")
    for path in chunk_paths:
        if os.path.exists(path):
            os.remove(path)

    print(f"✅ Success! Transcript generated at: {final_json_path}")
    return final_json_path