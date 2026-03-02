import cv2
import numpy as np
from backend.model_loader import predict_frame


def extract_frames(video_path, frame_interval=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames


def analyze_video(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return {"result": "Invalid Video", "confidence": 0}

    predictions = []

    for frame in frames:
        pred = predict_frame(frame)
        predictions.append(pred)

    avg_prediction = np.mean(predictions)

    if avg_prediction > 0.5:
        return {
            "result": "AI Generated",
            "confidence": round(avg_prediction * 100, 2)
        }
    else:
        return {
            "result": "Real",
            "confidence": round((1 - avg_prediction) * 100, 2)
        }