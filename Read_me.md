🎭 AI Video Detection Using CNN-Based Transfer Learning


📌 Abstract

This project presents an AI video detection system using Convolutional Neural Networks (CNN) with transfer learning. A pre-trained MobileNetV2 model is fine-tuned to classify video frames as Real or Fake. The system extracts frames from uploaded videos, performs frame-level classification, and determines the final video prediction using majority voting. A Flask-based web interface allows users to upload videos and view predictions interactively.

🧠 Introduction

With the rapid advancement of the technology, manipulated videos have become increasingly realistic and difficult to detect. This project aims to build a reliable AI-based system capable of identifying whether a video is authentic or artificially manipulated.

We use a CNN-based transfer learning approach (MobileNetV2) to analyze individual video frames. The predictions from multiple frames are combined to produce a final classification for the entire video.

🚀 Features

✅ CNN-based Transfer Learning

✅ Frame-Level Image Classification

✅ Video-Level Prediction (Majority Voting)

✅ Flask Web Interface

✅ ~75% Validation Accuracy

✅ Real vs Fake Binary Classification


🏗️ System Architecture

       Video upload
             |
Frame Extraction using OpenCV
             |
Image Preprocessing (Resize, Normalize)
             |
CNN Prediction (MobileNetV2)
             |
      Majority Voting
             |
Final Result Displayed on Web Interface


🧠 Model Details

Model: MobileNetV2 (Pre-trained on ImageNet)

Type: CNN with Transfer Learning

Output Classes: Real / Fake

Loss Function: CrossEntropyLoss

Optimizer: Adam

Framework: PyTorch



📊 Results & Performance

Training Accuracy: ~78%

Validation Accuracy: ~75%

Binary Classification: Real vs Fake

Performs well on facial deepfake datasets

The model demonstrates effective feature extraction using transfer learning and provides reliable classification results for manipulated videos.


🛠️ Technologies Used

-Python

-PyTorch

-OpenCV

-Flask

-NumPy

-Pillow

📂 Project Structure

AI_Video_Detection/
│
├── train_model.py

├── predict_video.py

├── app.py

├── requirements.txt

├── Read_me.md

├── templates/

└── static/

▶️ How to Run the Project

1️⃣ Clone the Repository
git clone https://github.com/AryaUnniSodha/AI-Video-Detection-using-CNN-based-Transfer-Learning.git

cd AI-Video-Detection-using-CNN-based-Transfer-Learning

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Flask Application
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000/

🔮 Future Scope

Improve accuracy using EfficientNet / Xception

Integrate face detection before classification

Deploy using cloud platforms (AWS / Heroku)

Extend to multi-class classification

Add real-time webcam detection

🎯 Applications

Social media content verification

News media authenticity checking

Cybersecurity investigations

Digital forensics

Online identity verification systems

👩‍💻 Author

Arya U
AI & Deep Learning Enthusiast
