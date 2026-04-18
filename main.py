import cv2
import streamlit as st
import time
import os
from ultralytics import YOLO

st.set_page_config(page_title="AI Surveillance", layout="wide")

st.title("🚨 AI Smart Surveillance System")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------ UI ------------------
source = st.radio("Select Source", ["Webcam", "Upload Video"])

video_source = 0

if source == "Upload Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded:
        with open("temp.mp4", "wb") as f:
            f.write(uploaded.read())
        video_source = "temp.mp4"

beep = st.checkbox("Enable Alert Sound", value=True)

start = st.button("🚀 Start Detection")
stop = st.button("⏹️ Stop Detection")

frame_placeholder = st.empty()
status_placeholder = st.empty()

# ------------------ ALERT COUNTER ------------------
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

st.sidebar.metric("🚨 Alerts", st.session_state.alert_count)

# ------------------ LOITERING SETUP ------------------
person_time = {}
LOITER_THRESHOLD = 5  # seconds

# ------------------ MAIN LOOP ------------------
if start:
    cap = cv2.VideoCapture(video_source)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        current_time = time.time()

        # ------------------ DETECTION LOGIC ------------------
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # 🔫 Weapon detection (sharp objects)
            if label in ["knife", "scissors"]:
                st.session_state.alert_count += 1
                status_placeholder.markdown("### 🚨 SHARP OBJECT DETECTED!")

                # Save evidence
                cv2.imwrite(f"alert_{int(time.time())}.jpg", frame)

                # Sound alert
                if beep:
                    try:
                        os.system('say "SHARP OBJECT DETECTED"')
                    except:
                        print("🔔 ALERT!")

            # 🧍 Loitering detection
            if label == "person":
                track_id = int(box.id[0]) if box.id is not None else 0

                if track_id not in person_time:
                    person_time[track_id] = current_time
                else:
                    duration = current_time - person_time[track_id]

                    if duration > LOITER_THRESHOLD:
                        st.session_state.alert_count += 1
                        status_placeholder.markdown("### ⚠️ LOITERING DETECTED!")

        # ------------------ FPS ------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # ------------------ DISPLAY ------------------
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated, use_container_width=True)

        status_placeholder.markdown(f"### 🟢 Monitoring | FPS: {fps:.1f}")

    cap.release()

if stop:
    st.warning("🛑 Detection Stopped")