import os
from groq import Groq
import json
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

# Apni API key yahan daal (real project me hum .env use karenge)
client = Groq(api_key=my_api_key)

# Teri test audio file ka path
filename = "audio/Zy4SJXnYBjI.m4a" # ya .mp3 jo bhi format download hua tha

print("Groq ko audio bhej raha hu... wait kar bhai")

with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
      file=(filename, file.read()),
      model="whisper-large-v3", # Groq ka sabse fast Whisper model
      response_format="verbose_json", # Ye timestamps nikalne ke liye zaroori hai
    )

# Result ko terminal pe print karke check karte hain
print("Transcription Done! Ye dekh output:\n")

# Sirf pehle 2-3 segments print karte hain check karne ke liye
for segment in transcription.segments[:3]:
    print(f"Start: {segment['start']}s | End: {segment['end']}s | Text: {segment['text']}")

# Ek choti JSON file me pura data save kar le
with open("test_transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcription.model_dump(), f, ensure_ascii=False, indent=4)
    
print("\nPura data test_transcript.json me save ho gaya hai!")