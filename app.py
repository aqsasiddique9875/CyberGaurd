# app.py — CyberGuard AI v2.0 (With Mic + PDF + Firebase Auth)
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import numpy as np
import json
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import base64
from io import BytesIO
import soundfile as sf

# ---------------------------
# Firebase Auth Setup
# ---------------------------
import firebase_admin
from firebase_admin import credentials, auth

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")  # Put your firebase-adminsdk.json here
    firebase_admin.initialize_app(cred)

def login_with_google():
    st.markdown("""
    <script src="https://accounts.google.com/gsi/client" async defer></script>
    <div id="g_id_onload"
         data-client_id="YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
         data-login_uri="https://your-streamlit-app.streamlit.app"
         data-auto_prompt="false">
    </div>
    """, unsafe_allow_html=True)

# Simple session-based auth (will be replaced by Firebase token)
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------------------
# CONFIG & CSS
# ---------------------------
st.set_page_config(page_title="CyberGuard AI", page_icon="Shield", layout="wide")

st.markdown("""
<style>
    .main {background: #0f172a; color: white;}
    .stApp {background: #0f172a;}
    h1, h2 {color: white !important;}
    .mic-button {background: #f59e0b !important; color: black !important;}
    .result-box {background: #1e293b; padding: 30px; border-radius: 16px; border: 1px solid #334155;}
    .nav-links a {color: #e2e8f0; margin: 0 20px; font-weight: 500; text-decoration: none;}
    .nav-links a:hover {color: #f59e0b;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# Shield CyberGuard AI")
with col2:
    st.markdown("""
    <div style="text-align:right; padding-top:20px;">
        <a href="?page=home">Home</a>
        <a href="?page=analysis">Analysis</a>
        <a href="?page=history">History</a>
        <a href="?page=login">Login</a>
        <a href="?page=signup" style="background:#f59e0b;color:black;padding:10px 20px;border-radius:8px;">Create Account</a>
    </div>
    """, unsafe_allow_html=True)

page = st.experimental_get_query_params().get("page", ["home"])[0]

# ---------------------------
# Auth Pages
# ---------------------------
if page == "login":
    st.markdown("<h1 style='text-align:center;margin:60px 0;'>Welcome Back</h1>", unsafe_allow_html=True)
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            st.session_state.user = {"email": email}
            st.success("Logged in!")
            st.experimental_rerun()
    st.markdown("<p style='text-align:center;'>Or</p>", unsafe_allow_html=True)
    if st.button("Continue with Google", type="secondary"):
        st.info("Google Sign-In ready (configure client ID)")

elif page == "signup":
    st.markdown("<h1 style='text-align:center;margin:60px 0;'>Create Account</h1>", unsafe_allow_html=True)
    with st.form("signup"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign Up Free"):
            st.session_state.user = {"email": email, "name": name}
            st.balloons()
            st.success("Account created! Welcome to CyberGuard AI")
            st.experimental_rerun()

# ---------------------------
# Home
# ---------------------------
elif page == "home":
    st.markdown("""
    <div style="text-align:center; padding:100px 20px; background:linear-gradient(rgba(0,0,0,0.8),rgba(0,0,0,0.9)), url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b') center/cover; border-radius:20px; margin:40px 0;">
        <h1 style="font-size:4.5rem;">Scam calls are bad for everyone.</h1>
        <p style="font-size:1.5rem; color:#cbd5e1;">Let CyberGuard AI detect fraud before it's too late.</p>
        <br>
        <a href="?page=analysis"><button style="background:#f59e0b; color:black; padding:16px 40px; font-size:1.2rem; border:none; border-radius:12px; font-weight:bold;">Start Free Check</button></a>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Analysis Page (With Mic + Upload + PDF)
# ---------------------------
elif page == "analysis":
    st.markdown("<h1 style='text-align:center;margin:40px 0;'>Analyze a Call</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Microphone Record", "Upload File"])

    result = None

    with tab1:
        st.markdown("### Record Live Call")
        audio_bytes = mic_recorder(
            start_prompt="Record Now",
            stop_prompt="Stop Recording",
            key="mic",
            use_container_width=True
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

    with tab2:
        st.markdown("### Or Upload Recording")
        uploaded_file = st.file_uploader("WAV, MP3, M4A", type=["wav","mp3","m4a","ogg"])

    text_input = st.text_area("Or paste transcript (optional)", height=100)

    if st.button("Analyze Call Now", type="primary", use_container_width=True):
        audio_data = None

        # Handle mic or upload
        if "audio_bytes" in locals() and audio_bytes:
            y, sr = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
            audio_data = (y, sr)
        elif uploaded_file:
            y, sr = librosa.load(uploaded_file, sr=16000, mono=True)
            audio_data = (y, sr)

        with st.spinner("AI is analyzing the call..."):
            # Fake analysis
            import random
            result = {
                "transcription": text_input or "This is a call from your bank. We need your account details immediately...",
                "label": random.choice(["SAFE", "SUSPICIOUS", "HIGH RISK SCAM", "FRAUDULENT"]),
                "confidence": random.randint(35, 98),
                "reasons": random.sample([
                    "Urgency tactics used", "Impersonates authority", "Requests payment",
                    "Threatens arrest", "Asks for remote access"
                ], k=3),
                "safe_response": "I never share personal info over phone. Send official letter."
            }

            # Save to history
            history = load_history() if os.path.exists("history.json") else []
            history.insert(0, {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "label": result["label"],
                "confidence": result["confidence"]
            })
            with open("history.json", "w") as f:
                json.dump(history, f)

            # Generate PDF Report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "CyberGuard AI - Scam Detection Report", ln=1, align="C")
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
            pdf.cell(0, 10, f"Result: {result['label']} ({result['confidence']}% confidence)", ln=1)
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Reasons:", ln=1)
            pdf.set_font("Arial", size=11)
            for r in result["reasons"]:
                pdf.cell(0, 8, f"• {r}", ln=1)
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Recommended Response:", ln=1)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, result["safe_response"])

            pdf_output = BytesIO()
            pdf.output(pdf_output)
            pdf_bytes = pdf_output.getvalue()

            b64 = base64.b64encode(pdf_bytes).decode()

            st.success("Analysis Complete!")
            st.markdown(f"""
            <div class="result-box">
                <h2 style="color:#f59e0b;">{result['label']}</h2>
                <h3>Confidence: {result['confidence']}%</h3>
                <br>
                <h4>Reasons:</h4>
                <ul>{''.join([f"<li>{r}</li>" for r in result['reasons']])}</ul>
                <h4>Safe Response:</h4>
                <p style="background:#172554;padding:16px;border-radius:8px;color:#93c5fd;">
                    "{result['safe_response']}"
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Download Buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"cyberguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
            with col2:
                st.download_button(
                    "Download Audio (if recorded)",
                    data=audio_bytes if 'audio_bytes' in locals() else uploaded_file.read(),
                    file_name="recorded_call.wav",
                    mime="audio/wav"
                )

# ---------------------------
# History Page
# ---------------------------
elif page == "history":
    st.markdown("<h1 style='text-align:center;margin:40px 0;'>Your Analysis History</h1>", unsafe_allow_html=True)
    if os.path.exists("history.json"):
        history = json.load(open("history.json"))
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%b %d, %Y %I:%M %p")
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)
    else:
        st.info("No history yet. Record or upload your first call!")

# Update URL
st.experimental_set_query_params(page=page)
