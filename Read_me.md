AI Video Detection Using CNN-Based Transfer Learning
📌 Overview

This project implements an AI video detection system using MobileNetV2 and transfer learning. The system extracts video frames and classifies them as Real or Fake using a trained CNN model.

FEATURES:

CNN-based transfer learning

Frame-level classification

Video-level prediction

Flask web interface

~75% validation accuracy

MODEL:

MobileNetV2 (Pre-trained on ImageNet)

Fine-tuned for binary classification

CrossEntropyLoss + Adam optimizer

🛠 TECHNOLOGIES USED:

Python

PyTorch

OpenCV

Flask

PROJECT STRUCTURE:

train_model.py
predict_video.py
app.py
templates/
static/

▶️ How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py
