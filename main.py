import os
import asyncio
import yt_dlp
from transcriber import process_video  # Jo code maine pehle diya tha

PLAYLIST_URL = "https://youtube.com/playlist?list=PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2"
AUDIO_DIR = "temp_audio"

def download_audio_from_youtube(video_url, video_id):
    """yt-dlp se ek video ki audio nikalta hai"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_path = f"{AUDIO_DIR}/{video_id}.m4a"
    
    # Agar audio pehle se downloaded hai (last crash ki wajah se), toh skip mat karo yt-dlp khud handle karega
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': audio_path,
        'quiet': True,
        'no_warnings': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
        
    return audio_path

async def ingest_playlist():
    print(f"Fetching playlist details for 125 videos...")
    
    ydl_opts = {
        'extract_flat': True, # Sirf links nikalne ke liye, bina download kiye
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_dict = ydl.extract_info(PLAYLIST_URL, download=False)
        
    entries = playlist_dict.get('entries', [])
    print(f"Total videos found: {len(entries)}")
    
    for index, entry in enumerate(entries):
        video_id = entry['id']
        video_url = entry['url']
        final_json_path = f"transcripts/{video_id}.json"
        
        # IDEMPOTENCY: Agar json already ban gaya hai, toh agle video par jao (Crash proof)
        if os.path.exists(final_json_path):
            print(f"[{index+1}/{len(entries)}] Transcript already exists for {video_id}. Skipping.")
            continue
            
        print(f"\n[{index+1}/{len(entries)}] Processing Video: {video_id}")
        
        try:
            # Step 1: Raw Audio Download
            audio_path = download_audio_from_youtube(video_url, video_id)
            
            # Step 2: VAD Split -> Groq API -> Atomic Save (Tera Engine)
            await process_video(video_id, audio_path, output_dir="transcripts")

            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"🗑️ Cleaned up original audio: {audio_path}")
            
        except Exception as e:
            print(f"Failed on video {video_id}: {e}")
            print("Moving to the next video...")
            # Ek video fail ho jaye toh poora 125 ka run rukna nahi chahiye

if __name__ == "__main__":
    asyncio.run(ingest_playlist())