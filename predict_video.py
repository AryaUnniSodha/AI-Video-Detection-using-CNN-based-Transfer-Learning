import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import cv2

MODEL_PATH = "deepfake_model.pth"
device = torch.device("cpu")

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

video_path = input("Enter video path: ")
cap = cv2.VideoCapture(video_path)

fake_count = 0
real_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 15 == 0:  # check every 15th frame
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
    print("No frames processed.")
else:
    fake_percent = (fake_count / total) * 100
    real_percent = (real_count / total) * 100

    print("\n--- RESULT ---")
    print(f"Fake Frames: {fake_percent:.2f}%")
    print(f"Real Frames: {real_percent:.2f}%")

    if fake_percent > 10:
        print("Final Prediction: FAKE VIDEO")
    else:
        print("Final Prediction: REAL VIDEO")