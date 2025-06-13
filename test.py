import os
from ytmp4 import trim_middle_minute
from extract import extract_features, add_core_speech_features, convert_np_types, filter_numerical_features
import joblib
import json
import pandas as pd

# 1. Trim the video (replace 'your_video.mp4' with your actual filename)
video_filename = "1.mp4"  # <-- Change this to your actual video filename
trim_middle_minute(video_filename)

# 2. Extract features
features = extract_features(video_filename)
features = add_core_speech_features(features)
features_cleaned = convert_np_types(features)
features_ml = filter_numerical_features(features_cleaned)
print("Extracted features for ML:")
print(json.dumps(features_ml, indent=2))

# 3. Load trained ML models
model_names = ["clarity_fluency", "vocal_variety", "pacing", "confidence", "tone"]
models = {}
for name in model_names:
    models[name] = joblib.load(f"model_{name}.pkl")

# 4. Prepare features for prediction (as DataFrame)
X = pd.DataFrame([features_ml])

# --- Feature alignment check ---
for name in model_names:
    print(f"\nModel '{name}' expects features:")
    if hasattr(models[name], "feature_names_in_"):
        print(list(models[name].feature_names_in_))
    else:
        print("feature_names_in_ not available for this model.")
print("\nFeatures provided for prediction:")
print(list(X.columns))

# 5. Run analysis and output the 5 metrics
print("\nML Analysis Results:")
for name in model_names:
    pred = models[name].predict(X)[0]
    print(f"{name}: {pred:.2f}")
    preds = models[name].predict(X)
    print(f"{name} predictions: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")