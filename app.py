import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "deepfake_model.pth"
device = torch.device("cpu")

# Load model
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fake_count = 0
    real_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            if predicted.item() == 0:
                fake_count += 1
            else:
                real_count += 1

        frame_count += 1

    cap.release()

    total = fake_count + real_count
    if total == 0:
        return "No frames processed", 0

    fake_percent = (fake_count / total) * 100

    if fake_percent > 30:
        return "FAKE VIDEO", round(fake_percent, 2)
    else:
        return "REAL VIDEO", round(100 - fake_percent, 2)


@app.route("/detect/", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result, confidence = predict_video(filepath)

    return jsonify({
        "result": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(port=8080, debug=True)