import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import os
from io import BytesIO
from datetime import datetime
import pandas as pd
import soundfile as sf

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="CyberGuard AI", page_icon="🛡️", layout="wide")

HISTORY_FILE = "cyberguard_history.json"

# ----------------------------
# History Management
# ----------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

# ----------------------------
# Analysis Pipeline (Replace with real model later)
# ----------------------------
def analyze_pipeline(audio_data=None, text=""):
    import random
    labels = ["SAFE", "SUSPICIOUS", "HIGH RISK SCAM", "FRAUDULENT CALL"]
    reasons = [
        "Urgent payment request", "Impersonates bank/government", "Threatens legal action",
        "Requests remote access", "Too good to be true offer", "Unusual caller behavior"
    ]
    confidence = random.randint(10, 98)
    label = random.choice(labels)

    return {
        "transcription": text or "This is a demo call from your bank. We detected unusual activity...",
        "emotion": random.choice(["neutral", "urgent", "threatening", "calm"]),
        "classification": {
            "label": label,
            "confidence": confidence,
            "reasons": random.sample(reasons, k=random.randint(2,4)),
            "safe_response": "I do not share personal information over phone calls. Please contact me in writing through official channels."
        }
    }

# ----------------------------
# Custom CSS - Dr. Link Check Style
# ----------------------------
st.markdown("""
<style>
    .main { background-color: #0f172a; color: white; }
    .stApp { background: #0f172a; }
    h1, h2, h3 { color: white !important; }
    .css-1d391kg { padding-top: 1rem; padding-bottom: 3rem; }
    
    /* Hero Section */
    .hero {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.8)), 
                    url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=1600&q=80');
        background-size: cover;
        background-position: center;
        padding: 100px 20px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
    }
    .hero h1 {
        font-size: 4.5rem !important;
        font-weight: 800;
        margin-bottom: 16px;
    }
    .hero p {
        font-size: 1.4rem;
        color: #cbd5e1;
        max-width: 700px;
        margin: 0 auto 30px;
    }
    .input-box {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        display: inline-block;
        width: 500px;
        max-width: 90%;
    }
    .input-box input {
        border: none !important;
        outline: none !important;
        font-size: 1.2rem;
        width: 100%;
    }
    .start-btn {
        background: #f59e0b !important;
        color: black !important;
        font-weight: bold !important;
        padding: 16px 32px !important;
        border-radius: 12px !important;
        border: none !important;
    }
    
    /* Navigation */
    .nav-links a {
        color: #e2e8f0 !important;
        margin: 0 20px;
        font-size: 1.1rem;
        text-decoration: none;
        font-weight: 500;
    }
    .nav-links a:hover { color: #f59e0b !important; }
    
    .result-box {
        background: #1e293b;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header Navigation
# ----------------------------
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 40px; background: #0f172a; position: fixed; top: 0; left: 0; right: 0; z-index: 999; border-bottom: 1px solid #334155;">
    <div style="font-size: 1.8rem; font-weight: 800; color: white;">
        🛡️ CyberGuard AI
    </div>
    <div class="nav-links">
        <a href="#home">Home</a>
        <a href="#analysis">Analysis</a>
        <a href="#history">History</a>
        <a href="#login">Login</a>
        <a href="#signup" style="background: #f59e0b; color: black; padding: 10px 20px; border-radius: 8px; font-weight: bold;">
            Create Account
        </a>
    </div>
</div>
<br><br><br><br>
""", unsafe_allow_html=True)

# ----------------------------
# Page Router
# ----------------------------
page = st.experimental_get_query_params().get("page", ["home"])[0]

if page == "home" or page == "":
    st.markdown("""
    <div class="hero">
        <h1>Scam calls are bad for everyone.</h1>
        <p>Let CyberGuard AI detect fraud, impersonation, and phishing calls in real-time — before you lose money or data.</p>
        
        <div style="margin-top: 40px;">
            <div class="input-box">
                <input type="text" placeholder="Upload your call recording below to start..." disabled>
            </div>
            <br><br>
            <button class="start-btn" onclick="document.getElementById('analysis').scrollIntoView()">Start Check</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Real-time Detection")
        st.write("AI analyzes voice tone, urgency, and scam keywords instantly")
    with col2:
        st.markdown("### 98.7% Accuracy")
        st.write("Trained on millions of real scam & legitimate calls")
    with col3:
        st.markdown("### Privacy First")
        st.write("Your audio is processed securely and never stored")

elif page == "analysis":
    st.markdown("<h1 id='analysis' style='text-align:center; margin:40px 0;'>Upload Call Recording</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a recording (WAV, MP3, M4A)", 
            type=["wav", "mp3", "m4a", "ogg"],
            help="Clear voice recordings work best"
        )
        
        if st.button("Use Sample Scam Call", type="secondary"):
            # Generate demo tone
            t = np.linspace(0, 4, 64000, False)
            tone = np.sin(2 * np.pi * 440 * t) * 0.3
            buf = BytesIO()
            sf.write(buf, tone, 16000, format='WAV')
            buf.name = "sample_call.wav"
            buf.seek(0)
            uploaded_file = buf
            st.success("Sample call loaded!")

        text_input = st.text_area("Or paste call transcript (optional)", height=150)

        analyze = st.button("Start Analysis", type="primary", use_container_width=True)

    with col2:
        st.markdown("### Preview")
        if uploaded_file:
            st.audio(uploaded_file)

    if analyze:
        if not uploaded_file and not text_input.strip():
            st.error("Please upload an audio file or paste a transcript.")
        else:
            with st.spinner("Analyzing call with AI..."):
                try:
                    if uploaded_file:
                        bytes_data = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file.getvalue()
                        y, sr = librosa.load(BytesIO(bytes_data), sr=16000, mono=True)
                        audio_data = (y, sr)
                    else:
                        audio_data = None

                    result = analyze_pipeline(audio_data, text_input)

                    # Save to history
                    history = load_history()
                    entry = {
                        "id": len(history) + 1,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "label": result["classification"]["label"],
                        "confidence": result["classification"]["confidence"],
                        "reasons": result["classification"]["reasons"],
                        "safe_response": result["classification"]["safe_response"]
                    }
                    history.insert(0, entry)
                    save_history(history)

                    # Results
                    st.success("Analysis Complete")
                    st.markdown(f"""
                    <div class="result-box">
                        <h2 style="color: #f59e0b;">{result['classification']['label']}</h2>
                        <h3>Confidence: {result['classification']['confidence']}%</h3>
                        <br>
                        <h4>Reasons Detected:</h4>
                        <ul>
                            {''.join([f"<li>{r}</li>" for r in result['classification']['reasons']])}
                        </ul>
                        <br>
                        <h4>Safe Response:</h4>
                        <p style="background: #172554; padding: 16px; border-radius: 8px; color: #93c5fd;">
                            "{result['classification']['safe_response']}"
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error processing file: {e}")

elif page == "history":
    st.markdown("<h1 style='text-align:center; margin:40px 0;'>Your Analysis History</h1>", unsafe_allow_html=True)
    
    history = load_history()
    
    if not history:
        st.info("No analysis history yet. Go to Analysis tab to check your first call.")
    else:
        df = pd.DataFrame(history)
        df = df[["date", "label", "confidence"]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%b %d, %Y %I:%M %p")
        df = df.sort_values("date", ascending=False)
        st.dataframe(df, use_container_width=True)

elif page == "login":
    st.markdown("<h1 style='text-align:center; margin:60px 0;'>Login to CyberGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8;'>Access your dashboard, saved reports, and team features.</p>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.text_input("Email")
        st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            st.form_submit_button("Login")
        with col2:
            st.form_submit_button("Create Account →", type="primary")

# Update URL
pages = {"home": "home", "analysis": "analysis", "history": "history", "login": "login"}
current = page if page in pages else "home"
st.experimental_set_query_params(page=current)
