"""
Ultra-Premium Streamlit UI for SafeCall AI (Scam/Fraud Voice Detector)
- Multi-page: Home | Analyze | History | Admin
- Animated Plotly gauge, waveform visualizer, download report
- Simple session auth for Admin
- Replace `analyze_pipeline` dummy with your real backend function
"""

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
from PIL import Image

# ----------------------------
# CONFIG & PATHS
# ----------------------------
st.set_page_config(page_title="SafeCall AI — Ultra", page_icon="🚨", layout="wide")
HISTORY_PATH = "history.json"

# ----------------------------
# Utilities
# ----------------------------
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_history(history_list):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_list, f, indent=2, default=str)

def bytes_to_audiofile_bytes(uploaded_file):
    # return raw bytes and file-like for librosa if needed
    uploaded_file.seek(0)
    b = uploaded_file.read()
    return b

def make_downloadable_json(obj, name="result.json"):
    b = json.dumps(obj, indent=2).encode("utf-8")
    return st.download_button("⬇️ Download JSON", data=b, file_name=name, mime="application/json")

def make_downloadable_report(result, name_prefix="report"):
    # Simple text/pdf placeholder: we'll create a plain text report
    txt = []
    txt.append("SafeCall AI — Analysis Report")
    txt.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    txt.append("")
    txt.append("Transcription:")
    txt.append(result.get("transcription",""))
    txt.append("")
    txt.append("Emotion: " + str(result.get("emotion","")))
    txt.append("")
    txt.append("Classification:")
    for k,v in result.get("classification",{}).items():
        txt.append(f"- {k}: {v}")
    data = "\n".join(txt).encode("utf-8")
    return st.download_button("⬇️ Download Report (TXT)", data=data, file_name=f"{name_prefix}.txt", mime="text/plain")

def render_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(9,2.2))
    ax.plot(np.linspace(0, len(y)/sr, num=len(y)), y, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Audio waveform")
    plt.tight_layout()
    return fig

def plot_gauge(score:int, title="Scam Score"):
    # score 0-100
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        number = {'suffix':'%','font':{'size':34,'color':'#ffffff'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range':[0,100], 'tickwidth':1, 'tickcolor':"#888"},
            'bar': {'color': "#00cc96"},
            'steps': [
                {'range':[0,50], 'color':'#0fbf6f'},
                {'range':[50,75], 'color':'#ffb020'},
                {'range':[75,100], 'color':'#ff4d4d'}
            ],
            'threshold': {
                'line': {'color': "white", 'width':4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=30,b=0),
        font=dict(color='white')
    )
    return fig

# ----------------------------
# Replace this with your real backend analyze_pipeline function.
# Signature: analyze_pipeline(audio_data: tuple(y,sr) or None, manual_text: str) -> dict
# The returned dict MUST contain keys:
#  - transcription (str)
#  - emotion (str)
#  - features (dict) optional
#  - classification (dict) with label (str), confidence (int 0-100), reasons (list), safe_response (str)
# ----------------------------
def analyze_pipeline(audio_data, manual_text):
    # Demo heuristic/fake result — REPLACE in production
    demo = {
        "transcription": manual_text if manual_text else "Demo: Please replace analyze_pipeline() with your real function.",
        "emotion": "neutral",
        "features": {"rms": 0.03, "tempo": 120},
        "classification": {
            "label": "POTENTIAL_SPAM",
            "confidence": 62,
            "reasons": ["contains promotional language", "caller identity unclear"],
            "safe_response": "I'm not sharing details; please send this in writing."
        }
    }
    # If audio present, tweak demo score a little
    if audio_data is not None:
        y,sr = audio_data
        dur = round(len(y)/sr,2)
        demo["transcription"] = demo["transcription"] + f" (audio duration: {dur}s)"
        demo["features"]["duration"] = dur
        # small heuristic: louder -> higher confidence
        rms = float(np.mean(librosa.feature.rms(y=y)))
        demo["classification"]["confidence"] = int(min(95, demo["classification"]["confidence"] + (rms*100)))
    return demo

# ----------------------------
# Simple session-based admin auth (DEMO)
# Replace with your own auth mechanism for production
# ----------------------------
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", None) if "secrets" in st.__dict__ else None
# For demo convenience, if no secret is provided, default password:
if ADMIN_PASSWORD is None:
    ADMIN_PASSWORD = "adminpass"  # change this in production!

def login_ui():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        st.sidebar.success("Admin: logged in")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()
    else:
        st.sidebar.markdown("---")
        st.sidebar.write("🔐 Admin login")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login as Admin"):
            if pw == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.sidebar.success("Welcome, Admin")
                st.experimental_rerun()
            else:
                st.sidebar.error("Incorrect password")

# ----------------------------
# Top navigation & header
# ----------------------------
logo_placeholder = """
<div style="display:flex;align-items:center;gap:14px">
  <div style="width:56px;height:56px;border-radius:12px;background:linear-gradient(135deg,#06b6d4,#8b5cf6);
              display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:26px">SC</div>
  <div style="line-height:1">
    <div style="font-size:20px;font-weight:700;color:white">SafeCall AI</div>
    <div style="font-size:12px;color:#bfc9d9">Premium Scam & Fraud Voice Detection</div>
  </div>
</div>
"""
st.markdown(logo_placeholder, unsafe_allow_html=True)
st.markdown("""<hr style="border:0;height:1px;background:#1f2937;margin-top:12px;margin-bottom:18px">""", unsafe_allow_html=True)

# Sidebar navigation
nav = st.sidebar.selectbox("Navigation", ["Home","Analyze","History","Admin"])
st.sidebar.markdown("### Quick actions")
st.sidebar.button("Open Docs")
st.sidebar.markdown("---")
login_ui()

# ----------------------------
# HOME PAGE
# ----------------------------
if nav == "Home":
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("## Welcome to SafeCall AI — Ultra")
        st.write("""
        An enterprise-grade interface for detecting scams, fraud and spam in voice calls.
        Upload an audio file or paste a transcript and get a clear, explainable analysis.
        """)
        st.markdown("### Features")
        st.markdown("- Animated gauge with color zones (safe / suspicious / fraudulent)")
        st.markdown("- High-resolution waveform preview & player")
        st.markdown("- Downloadable JSON / report for audits")
        st.markdown("- History & admin analytics")
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("1. Go to **Analyze**.  2. Upload audio (wav/mp3) or paste transcript.  3. Click Analyze.  4. Review results & download.")
    with col2:
        st.markdown("### Tips")
        st.info("For best results upload clean recorded PCM WAV (16kHz). Short clips ≤ 30s work best.")
        st.success("Integrate your backend by replacing the demo analyze_pipeline() in this file.")
        st.warning("Admin password is demo-only; change in production.")
        st.image(Image.new("RGBA",(200,120),(20,24,39,255)))

# ----------------------------
# ANALYZE PAGE
# ----------------------------
elif nav == "Analyze":
    st.markdown("## Analyze a Call")
    left, right = st.columns([1.2,1])
    with left:
        st.markdown("### Input")
        uploaded = st.file_uploader("Upload audio file (wav/mp3/m4a) or drag & drop", type=["wav","mp3","m4a"])
        sample_btn = st.button("Load sample audio")
        manual_text = st.text_area("Optional: paste transcript / notes", height=120)
        analyze_button = st.button("🔎 Analyze")

        # provide sample audio from small generated sine wave if requested
        if sample_btn and uploaded is None:
            duration=3.0
            sr=16000
            t = np.linspace(0,duration,int(sr*duration),False)
            tone = 0.02*np.sin(2*np.pi*400*t)
            buf = BytesIO()
            librosa.output.write_wav(buf, tone, sr) if hasattr(librosa.output,"write_wav") else sf_write_wav(buf, tone, sr)
            buf.seek(0)
            uploaded = buf

    with right:
        st.markdown("### Live Preview")
        preview_card = st.container()
        preview_card.markdown("<div style='padding:14px;border-radius:12px;background:rgba(255,255,255,0.03)'>", unsafe_allow_html=True)
        if uploaded:
            # show audio player
            try:
                st.audio(uploaded)
            except:
                st.info("Audio loaded (preview unavailable in this environment).")
        else:
            st.info("Upload audio to preview waveform & player.")
        preview_card.markdown("</div>", unsafe_allow_html=True)

    # Run analysis on click
    if analyze_button:
        if uploaded is None and (not manual_text or manual_text.strip()==""):
            st.error("Please upload audio or provide transcript text.")
            st.stop()

        # Load audio bytes into librosa if present
        audio_data = None
        if uploaded:
            # streamlit returns UploadedFile type or BytesIO; ensure seek and use librosa
            try:
                uploaded.seek(0)
            except:
                pass
            data_bytes = uploaded.read() if hasattr(uploaded, "read") else bytes_to_audiofile_bytes(uploaded)
            # librosa can load from bytes via BytesIO buffer
            try:
                y, sr = librosa.load(BytesIO(data_bytes), sr=16000, mono=True)
                audio_data = (y, sr)
            except Exception as e:
                st.warning(f"Could not decode audio with librosa: {e}. Proceeding without audio features.")
                audio_data = None

        # Run backend
        with st.spinner("Analyzing… this may take a few seconds"):
            result = analyze_pipeline(audio_data, manual_text or "")
        # store result with timestamp
        hist = load_history()
        entry = {"id": len(hist)+1, "ts": datetime.utcnow().isoformat()+"Z", "result": result}
        hist.insert(0, entry)
        save_history(hist)

        # Display results
        st.markdown("---")
        st.markdown("## Results")
        colA, colB = st.columns([1.4, 1])
        with colA:
            st.markdown("### 🔎 Transcription")
            st.write(result.get("transcription","—"))

            st.markdown("### ✅ Suggested Safe Response")
            st.info(result.get("classification",{}).get("safe_response","—"))

            st.markdown("### 🧾 Reasons")
            reasons = result.get("classification",{}).get("reasons",[])
            for r in reasons:
                st.write(f"- {r}")

            # JSON download
            make_downloadable_json(result, name=f"analysis_{entry['id']}.json")
            make_downloadable_report(result, name_prefix=f"analysis_{entry['id']}_report")
        with colB:
            conf = int(result.get("classification",{}).get("confidence",0))
            label = result.get("classification",{}).get("label","—")

            st.metric(label="Prediction", value=label, delta=f"{conf}% confidence")
            gauge = plot_gauge(conf)
            st.plotly_chart(gauge, use_container_width=True)

            # Waveform
            if audio_data is not None:
                y,sr = audio_data
                fig_w = render_waveform(y, sr)
                st.pyplot(fig_w)
            else:
                st.info("No audio waveform available. (Uploaded file could not be decoded.)")

# ----------------------------
# HISTORY PAGE
# ----------------------------
elif nav == "History":
    st.markdown("## Analysis History")
    history = load_history()
    if not history:
        st.info("No analysis history yet (analyze something to populate records).")
    else:
        # show table and allow filtering
        df = pd.DataFrame([
            {
                "id": item["id"],
                "ts": item["ts"],
                "label": item["result"].get("classification",{}).get("label",""),
                "confidence": item["result"].get("classification",{}).get("confidence",0),
                "transcription": item["result"].get("transcription","")
            } for item in history
        ])
        st.data_editor(df[["id","ts","label","confidence"]], num_rows="dynamic")
        st.markdown("---")
        st.markdown("### Detailed entry")
        sel = st.number_input("Enter record id to view", min_value=1, max_value=len(history), value=1)
        chosen = next((h for h in history if h["id"]==sel), None)
        if chosen:
            st.json(chosen)
            # allow download
            make_downloadable_json(chosen, name=f"history_{sel}.json")

# ----------------------------
# ADMIN PAGE
# ----------------------------
elif nav == "Admin":
    if not st.session_state.get("logged_in", False):
        st.warning("Admin area — please login from the sidebar.")
        st.stop()

    st.markdown("## Admin Dashboard")
    history = load_history()
    st.markdown("### Summary")
    total = len(history)
    avg_conf = int(np.mean([h["result"].get("classification",{}).get("confidence",0) for h in history]) if history else 0)
    fraud_count = sum(1 for h in history if h["result"].get("classification",{}).get("label","").lower() in ["fraud","spam"])
    col1,col2,col3 = st.columns(3)
    col1.metric("Total analyses", total)
    col2.metric("Avg Confidence", f"{avg_conf}%")
    col3.metric("Fraud-like", fraud_count)

    st.markdown("### Confidence distribution")
    if history:
        confs = [h["result"].get("classification",{}).get("confidence",0) for h in history]
        fig, ax = plt.subplots()
        ax.hist(confs, bins=10, color="#7c3aed")
        ax.set_xlabel("Confidence")
        st.pyplot(fig)
    else:
        st.info("No data to visualize")

    st.markdown("### Raw history (JSON)")
    st.write(history)

    # Option to clear history
    if st.button("⚠️ Clear History (admin)"):
        save_history([])
        st.success("History cleared")
