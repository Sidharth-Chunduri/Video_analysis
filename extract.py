import os
import cv2
import numpy as np
import librosa
import whisper
from moviepy.editor import VideoFileClip
from scipy.ndimage import gaussian_filter1d
import re
import json
import textstat
import soundfile as sf
import mediapipe as mp
from textblob import TextBlob

PATH = "test_vids/1.mp4"  # Replace with video file path

FILLER_WORDS = [
    "um", "uh", "erm", "ah", "hmm", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "literally", "right", "okay", "just", "sort of",
    "kind of", "anyway", "you see", "i guess", "at the end of the day", "to be honest",
    "whatever", "stuff like that", "you know what i'm saying", "know what i mean", "etcetera"
]

PRONOUNS = ["i", "me", "my", "mine", "we", "us", "you", "your", "he", "she", "they", "them", "it"]

AUDIO_THRESHOLD = 0.01
MIN_PAUSE_DURATION = 0.3

# Load Whisper model
whisper_model = whisper.load_model("base")


#trim the clip
def trim_leading_silence_from_audio(audio_path, output_path="trimmed_temp.wav", top_db=30):
    y, sr = librosa.load(audio_path, sr=None)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)

    if not non_silent_intervals.any():
        # No speech found — return original
        return audio_path

    # Trim from the first non-silent frame onward
    start_sample = non_silent_intervals[0][0]
    y_trimmed = y[start_sample:]
    sf.write(output_path, y_trimmed, sr)
    return output_path


#convert to native python types
def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif hasattr(obj, "item"):  
        return obj.item()
    else:
        return obj

#count filler words
def count_fillers(text):
    text = text.lower()
    count = 0
    for word in FILLER_WORDS:
        matches = re.findall(r'\b' + re.escape(word) + r'\b', text)
        count += len(matches)
    return count

#extracts audio
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(duration)

    rms = librosa.feature.rms(y=y)[0]
    avg_rms = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # Estimate pitch
    pitch_series = librosa.yin(y, fmin=75, fmax=300, sr=sr)
    avg_pitch = float(np.mean(pitch_series))
    pitch_range = float(np.max(pitch_series) - np.min(pitch_series))
    pitch_range_normalized = pitch_range / avg_pitch if avg_pitch > 0 else 0.0

    # Compute pause count and durations from silence
    silent = rms < AUDIO_THRESHOLD
    frame_duration = librosa.frames_to_time(1, sr=sr)
    pause_durations = []
    current_pause = 0

    for s in silent:
        if s:
            current_pause += frame_duration
        elif current_pause >= MIN_PAUSE_DURATION:
            pause_durations.append(current_pause)
            current_pause = 0
        else:
            current_pause = 0

    pause_count_normalized = len(pause_durations)/ duration if duration > 0 else 0.0
    pause_durations_mean = float(np.mean(pause_durations)) if pause_durations else 0.0
    pause_durations_std = float(np.std(pause_durations)) if pause_durations else 0.0

    return {
        "duration_sec": duration,
        "avg_rms_energy": avg_rms,
        "rms_std": rms_std,
        "pitch_range": pitch_range_normalized,
        "pause_count_psec": pause_count_normalized,
        "pause_durations_mean": pause_durations_mean,
        "pause_durations_std": pause_durations_std
    }


#transcription
def transcribe_and_analyze_speech(audio_path):
    result = whisper_model.transcribe(audio_path, no_speech_threshold=0.75)
    text = result['text']

    words = text.lower().split()
    word_count = len(words)

    filler_rate = count_fillers(text)/ word_count if word_count > 0 else 0
    pronouns = sum(words.count(p) for p in PRONOUNS)
    p_ct = pronouns / len(words) if words else 0

    # Sentiment
    blob = TextBlob(text)
    sentiment_polarity = float(blob.sentiment.polarity)
    sentiment_subjectivity = float(blob.sentiment.subjectivity)

    # Pronoun ratios
    first_person = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
    second_person = ["you", "your", "yours"]
    first_person_count = sum(words.count(p) for p in first_person)
    second_person_count = sum(words.count(p) for p in second_person)
    first_person_ratio = first_person_count / word_count if word_count else 0
    second_person_ratio = second_person_count / word_count if word_count else 0

    # Complex word ratio (words with 3+ syllables)
    complex_words = [w for w in words if textstat.syllable_count(w) >= 3]
    complex_word_ratio = len(complex_words) / word_count if word_count else 0

    return {
        "transcript": text,
        "word_count": word_count,
        "filler_rate": filler_rate,
        "pronoun_rate": p_ct,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "first_person_ratio": first_person_ratio,
        "second_person_ratio": second_person_ratio,
        "complex_word_ratio": complex_word_ratio
    }

#extract body motion
def extract_video_motion(video_path, frame_skip=3, min_flow_mag=0.5):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(video_path)
    prev_landmarks = None
    motion_magnitudes = []
    frame_idx = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            break
        total_frames += 1
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        curr_landmarks = []
        ih, iw = frame.shape[0], frame.shape[1]
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                curr_landmarks.append([lm.x * iw, lm.y * ih])
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                curr_landmarks.append([lm.x * iw, lm.y * ih])
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                curr_landmarks.append([lm.x * iw, lm.y * ih])
    
        curr_landmarks = np.array(curr_landmarks, dtype=np.float32) if curr_landmarks else None

        if prev_landmarks is not None and curr_landmarks is not None and len(prev_landmarks) == len(curr_landmarks):
            flow = curr_landmarks - prev_landmarks
            mag = np.linalg.norm(flow, axis=1)
            mag = mag[mag > min_flow_mag]
            if len(mag) > 0:
                motion_magnitudes.append(np.mean(mag))

        prev_landmarks = curr_landmarks
        frame_idx += 1

    cap.release()
    holistic.close()
    if motion_magnitudes:
        motion_magnitudes = gaussian_filter1d(motion_magnitudes, sigma=2)
        avg_motion = float(np.mean(motion_magnitudes))
        motion_std = float(np.std(motion_magnitudes))
        motion_frames = len(motion_magnitudes)
        motion_density = motion_frames / total_frames if total_frames > 0 else 0.0
        return {
            "avg_motion": avg_motion,
            "motion_std": motion_std,
            "motion_frames": motion_frames,
            "motion_density": motion_density
        }
    else:
        return {
            "avg_motion": 0.0,
            "motion_std": 0.0,
            "motion_frames": 0,
            "motion_density": 0.0
        }

def add_core_speech_features(features):
    transcript = features.get("transcript", "")
    word_count = features.get("word_count", 1)
    duration_sec = features.get("duration_sec", 1)

    words = transcript.split()

    speaking_rate_wpm = word_count / (duration_sec / 60) if duration_sec > 0 else 0
    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    readability_score = textstat.flesch_reading_ease(transcript)

    features.update({
        "speaking_rate_wpm": speaking_rate_wpm,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "readability": readability_score
    })

    features.pop("duration_sec", None)

    return features

#extract from video
def extract_features(video_path):
    print("[*] Extracting audio...")
    audio_path = "temp_audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    print("[*] Trimming leading silence...")
    trimmed_audio_path = trim_leading_silence_from_audio(audio_path)
    
    print("[*] Analyzing audio...")
    audio_feats = extract_audio_features(trimmed_audio_path)
    
    print("[*] Transcribing speech...")
    speech_feats = transcribe_and_analyze_speech(trimmed_audio_path)
    
    print("[*] Analyzing video motion...")
    video_feats = extract_video_motion(video_path)
    
    os.remove(audio_path)
    if trimmed_audio_path != audio_path:
        os.remove(trimmed_audio_path)
    
    return {**audio_feats, **speech_feats, **video_feats}

def filter_numerical_features(features):
    return {k: v for k, v in features.items() if isinstance(v, (int, float, bool))}


def save_transcript(video_filename, transcript_text, output_file="all_transcripts.json"):
    # Load existing transcripts if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            transcripts = json.load(f)
    else:
        transcripts = {}

    # Add or update the transcript entry
    video_key = os.path.splitext(os.path.basename(video_filename))[0]
    transcripts[video_key] = transcript_text

    # Write back to file
    with open(output_file, "w") as f:
        json.dump(transcripts, f, indent=2)

    print(f"Saved transcript for {video_key}")


def process_all_mp4(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp4"):
            video_path = os.path.join(input_folder, filename)
            try:
                features = extract_features(video_path)
                features = add_core_speech_features(features)
                features_cleaned = convert_np_types(features)
                save_transcript(video_path, features_cleaned.get("transcript", ""), "all_transcripts.json")
                features_ml = filter_numerical_features(features_cleaned)

                output_filename = f"features_{os.path.splitext(filename)[0]}.json"
                output_path = os.path.join(output_folder, output_filename)

                with open(output_path, "w") as f:
                    json.dump(features_ml, f, indent=2)

                print(f" Saved features for {filename} to {output_path}")
            except Exception as e:
                print(f" Failed to process {filename}: {e}")

if __name__ == "__main__":
    input_path = "test_vids"
    output_path = "test_features"  # Folder to save extracted features
    process_all_mp4(input_path, output_path)

"""if __name__ == "__main__":
    video_path = PATH
    features = extract_features(video_path)
    features = add_core_speech_features(features)
    features_cleaned = convert_np_types(features)
    features_ml = filter_numerical_features(features_cleaned)

    with open(f"features_{video_path[10:-4]}.json", "w") as f:
        json.dump(features_ml, f, indent=2)        
    print(f"Saved {video_path} features to json")"""



