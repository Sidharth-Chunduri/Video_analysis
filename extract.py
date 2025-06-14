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

INPUT = "1.mp4"  # Replace with video file path

FILLER_WORDS = [
    "um", "uh", "erm", "ah", "hmm", "like", "you know", "i mean", "so", "well",
    "actually", "basically", "literally", "right", "okay", "just", "sort of",
    "kind of", "anyway", "you see", "i guess", "at the end of the day", "to be honest",
    "whatever", "stuff like that", "you know what i'm saying", "know what i mean", "etcetera"
]

PRONOUNS = ["i", "me", "my", "mine", "we", "us", "you", "your", "he", "she", "they", "them", "it"]

AUDIO_THRESHOLD = 0.01
MIN_PAUSE_DURATION = 0.1

# Load Whisper model
whisper_model = whisper.load_model("base")

#save transcripts to json
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
    print(f"duration: {duration}")

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

    return duration, {
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

    lexical_diversity = len(set(words)) / word_count if word_count else -1
    avg_word_length = sum(len(word) for word in words) / word_count if word_count else -1
    readability = textstat.flesch_reading_ease(text)

    return text, word_count, words, {
        
        "filler_rate": filler_rate,
        "pronoun_rate": p_ct,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "first_person_ratio": first_person_ratio,
        "second_person_ratio": second_person_ratio,
        "complex_word_ratio": complex_word_ratio,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "readability_score": readability,
    }

def extract_features(video_path):
    print("[*] Extracting audio...")
    audio_path = "temp_audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    #trim leading silence
    audio_path = trim_leading_silence_from_audio(audio_path)

    #extract features
    print("[*] Extracting audio features... ")
    duration, audio_feats = extract_audio_features(audio_path)
    print("[*] Extracting speech features")
    transcript, word_count, words, speech_feats = transcribe_and_analyze_speech(audio_path)

    #add features
    print("[*] Extracting added features")
    wpm = word_count / (duration / 60) if duration > 0 else -1

    calculated_feats = {
        "speaking_rate": wpm,
    }

    os.remove(audio_path)

    return convert_np_types({**audio_feats, **speech_feats, **calculated_feats})

if __name__ == "__main__":
    video_path = INPUT
    output_path = f"features_{os.path.splitext(video_path)[0]}.json"

    features = extract_features(video_path)
    with open(output_path, "w") as f:
        json.dump(features, f, indent=2)
    print(f"Saved {video_path} features to json")
