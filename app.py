import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
import pandas as pd

# ===============================
# Load Model
# ===============================
model = load_model("emotion_detection_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion counter (for dashboard)
emotion_counter = {emotion: 0 for emotion in emotion_labels}

# ===============================
# Real-Time Video Processing
# ===============================
def process_frame(frame):

    global emotion_counter

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    prob_df = pd.DataFrame({
        "Emotion": emotion_labels,
        "Probability": [0]*len(emotion_labels)
    })

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)[0]

        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = prediction[emotion_index] * 100

        emotion_counter[emotion] += 1

        # Update probability dataframe
        prob_df["Probability"] = prediction * 100

        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    stats_df = pd.DataFrame({
        "Emotion": list(emotion_counter.keys()),
        "Count": list(emotion_counter.values())
    })

    return frame, prob_df, stats_df


# ===============================
# Reset Dashboard
# ===============================
def reset_dashboard():
    global emotion_counter
    emotion_counter = {emotion: 0 for emotion in emotion_labels}
    return pd.DataFrame({
        "Emotion": emotion_labels,
        "Count": [0]*len(emotion_labels)
    })


# ===============================
# Modern UI
# ===============================
with gr.Blocks(theme=gr.themes.Soft(), title="AI Emotion Detection System") as demo:

    gr.Markdown("""
    # 🎭 AI Real-Time Emotion Detection System

    🚀 Continuous live webcam streaming  
    📊 Emotion probability visualization  
    📈 Emotion statistics dashboard  
    🧠 CNN-based facial emotion classifier  
    """)

    with gr.Row():
        video_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Live Camera")

    with gr.Row():
        video_output = gr.Image(label="Processed Output")

    with gr.Row():
        prob_chart = gr.BarPlot(
            x="Emotion",
            y="Probability",
            title="Emotion Probability (%)"
        )

    with gr.Row():
        stats_chart = gr.BarPlot(
            x="Emotion",
            y="Count",
            title="Emotion Statistics Dashboard"
        )

    reset_btn = gr.Button("🔄 Reset Statistics")

    video_input.stream(
        process_frame,
        inputs=video_input,
        outputs=[video_output, prob_chart, stats_chart]
    )

    reset_btn.click(
        reset_dashboard,
        outputs=stats_chart
    )

demo.launch()
