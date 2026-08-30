import os
import yt_dlp


video_urls = [
    "https://youtu.be/Zy4SJXnYBjI?si=Pi0BVGNT4d-8HXcH",
    "https://youtu.be/GvSKlcdiwB4?si=VnXx2mftOXKwhwBC"
]


os.makedirs("audio", exist_ok=True)


options = {
    "format": "bestaudio[ext=m4a]/bestaudio",
    "outtmpl": "audio/%(id)s.%(ext)s",
}


for video_url in video_urls:

    print("\nDownloading:", video_url)

    with yt_dlp.YoutubeDL(options) as ydl:

        info = ydl.extract_info(video_url, download=True)

        file_path = ydl.prepare_filename(info)

    print("Downloaded:", file_path)

    # # Audio delete
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    #     print("Deleted:", file_path)
    # else:
    #     print("File not found:", file_path)