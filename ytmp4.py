from moviepy.editor import VideoFileClip
import os

def trim_middle_minute(filepath):
    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
        return

    print(f" Loading video: {filepath}")
    video = VideoFileClip(filepath)

    # Calculate start and end times for the middle 1-minute segment
    total_duration = video.duration
    if total_duration <= 60:
        print(" video is 60 seconds or shorter.")
        return

    mid_point = total_duration / 2
    start_time = max(0, mid_point - 30)
    end_time = min(total_duration, mid_point + 30)

    clip = video.subclip(start_time, end_time)

    # Save to temp file and replace original
    temp_filename = "temp_trimmed_video.mp4"
    clip.write_videofile(temp_filename, codec="libx264", audio_codec="aac")

    video.close()
    clip.close()

    os.remove(filepath)
    os.rename(temp_filename, filepath)

    print(f"Replaced with 1-minute clip ({start_time:.2f}s to {end_time:.2f}s).")

def trim_all_mp4(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".mp4"):
            filepath = os.path.join(folder_path, filename)
            try:
                trim_middle_minute(filepath)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    folder_path = "test_vids"  # Replace with your folder path
    trim_all_mp4(folder_path)
