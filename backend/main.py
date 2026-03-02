from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model
from PIL import Image
import cv2
import os

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ MUST match training backbone
base_model = get_model("mobilenetv3_large_w1", pretrained=False)

# Remove classifier
base_model.output = nn.Identity()

# ✅ EXACT architecture from your checkpoint
class DeepfakeModel(nn.Module):
    def __init__(self, base):
        super().__init__()

        # Important: wrapped in Sequential to match "base.0.xxx"
        self.base = nn.Sequential(base)

        self.h1 = nn.ModuleDict({
            "b1": nn.BatchNorm1d(2048),
            "l": nn.Linear(2048, 512),
            "b2": nn.BatchNorm1d(512),
            "o": nn.Linear(512, 1)
        })

    def forward(self, x):
        x = self.base(x)        # 2048
        x = self.h1["b1"](x)    # BN 2048
        x = self.h1["l"](x)     # 2048 → 512
        x = self.h1["b2"](x)    # BN 512
        x = self.h1["o"](x)     # 512 → 1
        return x


model = DeepfakeModel(base_model)

# ✅ Load your renamed model file
state_dict = torch.load("backend/model.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {"message": "AI Deepfake Detector API Running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(temp_video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "Could not read video"}

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    os.remove(temp_video_path)

    label = "Fake" if prob > 0.5 else "Real"

    return {
        "result": label,
        "confidence": round(prob * 100, 2)
    }