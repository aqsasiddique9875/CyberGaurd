import streamlit as st
import librosa
import json
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# IMPORT YOUR BACKEND LOGIC
# -------------------------
# Replace this line with your real backend:
# from backend import analyze_pipeline

# Temporary backend mock (DELETE when you integrate your model)
def analyze_pipeline(audio_data, manual_text):
    return {
        "transcription": "Your OTP is 54321. Please verify your account.",
        "emotion": "pressured/fast",
        "features": {"rms": 0.04, "pitch": 180, "tempo": 150, "duration": 3.2},
        "classification": {
            "label": "FRAUD",
            "confidence": 88,
            "reasons": [
                "mentions OTP/account verification",
                "urgent tone detected",
                "caller identity unclear"
            ],
            "safe_response": "I cannot verify anything on call. Send details in writing."
        }
    }


# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(
    page_title="AI Scam & Fraud Call Detector",
    page_icon="🚨",
    layout="wide"
)

# Custom Styling for Professional Look
st.markdown("""
    <style>
    .severity-box {
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        text-align: center;
        font-weight: 600;
    }
    .json-box {
        border-radius: 12px;
        padding: 15px;
        background-color: #262730;
        color: white;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# Title Section
# -------------------------
st.title("🚨 AI Scam & Fraud Voice Detector")
st.write("Analyze recorded or uploaded calls for fraud, spam or malicious intent using AI.")


# -------------------------
# Sidebar (Left Panel)
# -------------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Upload/Record audio & provide optional call text.")

manual_text = st.sidebar.text_area(
    "📋 Optional Text from Call:", 
    placeholder="Paste call text here...",
    height=150
)

uploaded_audio = st.sidebar.file_uploader(
    "🎤 Upload Caller Audio",
    type=["wav", "mp3", "m4a"]
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Analyze Audio", use_container_width=True)


# -------------------------
# Main Page Layout
# -------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("🎧 Audio Preview")
    if uploaded_audio:
        st.audio(uploaded_audio)

with col2:
    st.subheader("📊 Scam Severity Visualization")


# -------------------------
# When Analyze Button Clicks
# -------------------------
if run_btn:

    if uploaded_audio is None:
        st.error("Please upload an audio file first.")
        st.stop()

    # Load audio for backend
    y, sr = librosa.load(uploaded_audio, sr=16000, mono=True)
    audio_data = (y, sr)

    # Run backend
    result = analyze_pipeline(audio_data, manual_text)

    label = result["classification"]["label"]
    confidence = result["classification"]["confidence"]

    # Severity color
    if confidence >= 80:
        color = "#ff4d4d"  # RED
    elif confidence >= 60:
        color = "#ff9900"  # ORANGE
    else:
        color = "#00b300"  # GREEN

    st.markdown(
        f"""
        <div class="severity-box" style="background-color:{color};">
            Scam Severity Score: {confidence}%
            <br>
            <small>Prediction: {label}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.bar(["Severity"], [confidence], color=color)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("Scam Severity")

    st.pyplot(fig)

    # -------------------------
    # Text Results
    # -------------------------
    st.subheader("📝 Transcription")
    st.success(result["transcription"])

    st.subheader("🔊 Detected Emotion")
    st.info(result["emotion"])

    st.subheader("🛑 Reasoning")
    st.write(result["classification"]["reasons"])

    st.subheader("🗣 Safe Response Suggestion")
    st.warning(result["classification"]["safe_response"])

    # -------------------------
    # JSON Output
    # -------------------------
    st.subheader("📦 Full JSON Response")
    st.json(result)


else:
    st.info("Upload audio & click **Analyze Audio** to begin.")
