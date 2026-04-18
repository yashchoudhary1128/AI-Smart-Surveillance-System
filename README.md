# 🚨 AI Smart Surveillance System

## 📌 Overview
This project is an AI-based real-time surveillance system that detects suspicious activities using **YOLOv8**. It enhances security by automatically identifying abnormal behavior and generating alerts.

---

## 🎯 Key Features
- 🎥 Real-time object detection using YOLOv8  
- 🔫 Sharp object detection (knife, scissors)  
- 🧍 Loitering detection (person staying too long)  
- 🚨 Alert system with sound  
- 📸 Automatic screenshot capture (evidence)  
- 📊 Live alert counter dashboard  
- ⚡ Fast and stable performance  

---

## 🛠️ Tech Stack
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Streamlit  

---

## 🏗️ System Architecture

Camera → Frame Capture → YOLO Detection → Suspicious Logic → Alert System → Dashboard

---

## 🚀 How It Works
1. Captures live video (webcam or uploaded video)  
2. YOLOv8 detects objects in real-time  
3. Applies logic:
   - Detects sharp objects → 🚨 alert  
   - Detects loitering → ⚠️ alert  
4. Saves screenshots of suspicious activity  
5. Displays results on Streamlit dashboard  

---

## ▶️ How to Run

```bash
git clone https://github.com/yashchoudhary1128/-AI-Smart-Surveillance-System.git
cd -AI-Smart-Surveillance-System

pip install -r requirements.txt
streamlit run main.py

📊 Applications

* Smart cities
* Bank security
* Hostels & campuses
* Public surveillance

⸻

⚠️ Limitations

* Depends on lighting conditions
* YOLO default model has limited weapon detection
* Basic loitering logic

⸻

🔮 Future Scope

* Face recognition system
* Cloud-based alerts
* Mobile app integration
* Custom-trained weapon detection model

⸻

👨‍💻 Author

Yash Dev Choudhary
