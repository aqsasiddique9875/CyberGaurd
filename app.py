"""
Cyber Guard AI — Ultra Premium Streamlit App
Features:
 - Top link-style navigation (Home, Analyze, History, Login/Create Account)
 - Create account + Login (demo local users.json store)
 - Upload audio from laptop (wav/mp3/m4a)
 - Real-time pipeline: upload -> Whisper transcription (if available) -> Groq (if GROQ_API_KEY present)
 - Animated SVG gauge (needle) embedded via components.html (no Plotly)
 - Waveform + Spectrogram overlays
 - History (table) without raw JSON on the page
 - Download JSON / TXT reports per history entry
"""

import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json, os, tempfile, shutil, hashlib, time, re, requests
from datetime import datetime
import base64
import soundfile as sf
import streamlit.components.v1 as components
from PIL import Image

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Cyber Guard AI", page_icon="🛡️", layout="wide")
HISTORY_PATH = "history.json"
USERS_PATH = "users.json"
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY = os.environ.get("GROQ_API_KEY", None)

# ---------------------------
# Helpers
# ---------------------------
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_history(h):
    with open(HISTORY_PATH, "w") as f:
        json.dump(h, f, indent=2, default=str)

def load_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users(u):
    with open(USERS_PATH, "w") as f:
        json.dump(u, f, indent=2)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def make_download_bytes(obj, filename="result.json"):
    b = json.dumps(obj, indent=2).encode("utf-8")
    st.download_button("⬇️ Download JSON", data=b, file_name=filename, mime="application/json")

def make_download_txt(text:str, filename="report.txt"):
    st.download_button("⬇️ Download Report (TXT)", data=text.encode("utf-8"), file_name=filename, mime="text/plain")

# ---------------------------
# Whisper import (optional)
# ---------------------------
USE_WHISPER = False
try:
    import whisper
    USE_WHISPER = True
    WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
except Exception as e:
    whisper_model = None
    USE_WHISPER = False

# ---------------------------
# Groq call helper
# ---------------------------
def build_groq_prompt(transcription, emotion, features):
    return f"""
You are an assistant that MUST return JSON only. Analyze the short phone call below.

Transcription: \"\"\"{transcription}\"\"\"  
Estimated voice emotion: {emotion}  
Audio features: {json.dumps(features)}

Task:
- Choose label from: "SPAM", "FRAUD", "LEGITIMATE", "UNKNOWN".
- Provide confidence (0-100).
- Provide 2-4 concise reasons (array).
- Suggest one short safe response user can say (<= 20 words).

Return JSON only with keys: label, confidence, reasons, safe_response.
"""

def call_groq_chat(api_key, prompt, model="mixtral-8x7b", max_tokens=300, temperature=0.0, timeout=30):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a JSON-output assistant and must return JSON only."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1
    }
    resp = requests.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# ---------------------------
# Heuristic fallback classifier
# ---------------------------
def extract_audio_features_from_array(y, sr=16000):
    try:
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        tempo = 0.0
        try:
            tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=None)[0])
        except:
            tempo = 0.0
        duration = float(librosa.get_duration(y=y, sr=sr))
        pitch = 0.0
        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0_nonan = f0[~np.isnan(f0)]
            if len(f0_nonan)>0:
                pitch = float(np.median(f0_nonan))
        except:
            pitch = 0.0
        return {"rms": rms, "zcr": zcr, "pitch": pitch, "tempo": tempo, "duration": duration}
    except Exception as e:
        return {"error": str(e)}

def heuristic_emotion_label(features):
    if not isinstance(features, dict):
        return "unknown"
    dur = features.get("duration",0)
    if dur < 0.35:
        return "too_short"
    rms = features.get("rms",0)
    pitch = features.get("pitch",0)
    tempo = features.get("tempo",0)
    if rms > 0.03 and pitch > 200:
        return "angry/urgent"
    if tempo > 140 or (rms > 0.025 and tempo > 110):
        return "pressured/fast"
    if rms < 0.007 and tempo < 80:
        return "calm/soft"
    return "neutral"

def local_heuristic_classifier(transcription, emotion, features):
    txt = (transcription or "").lower()
    reasons = []
    score = 40
    fraud_keywords = ["bank","account","password","otp","verify","ssn","transfer","wire","pin"]
    spam_keywords = ["prize","winner","congratulations","free","claim","limited time","offer"]
    if any(k in txt for k in fraud_keywords):
        reasons.append("mentions account/verification keywords")
        score += 30
    if any(k in txt for k in spam_keywords):
        reasons.append("uses promotional/prize language")
        score += 20
    if "call from" in txt or "we are calling" in txt:
        reasons.append("caller identity unclear")
        score += 10
    if emotion and ("angry" in emotion or "pressured" in emotion):
        reasons.append("pressured/urgent tone")
        score += 10
    if score >= 80:
        label = "FRAUD"
    elif score >= 60:
        label = "SPAM"
    elif score >= 45:
        label = "POTENTIAL_SPAM"
    else:
        label = "LEGITIMATE"
    confidence = int(min(95, score + 5))
    if not reasons:
        reasons = ["No strong indicators detected"]
    return {"label": label, "confidence": confidence, "reasons": reasons, "safe_response": "I don't share personal info on calls; send details in writing."}

# ---------------------------
# SVG Gauge component generator
# ---------------------------
SVG_TEMPLATE = """
<div style="width:100%;display:flex;justify-content:center;">
  <svg id="gauge" viewBox="0 0 200 120" width="{width}" height="{height}">
    <defs>
      <linearGradient id="g1" x1="0" x2="1">
        <stop offset="0%" stop-color="#0fbf6f"/>
        <stop offset="50%" stop-color="#ffb020"/>
        <stop offset="100%" stop-color="#ff4d4d"/>
      </linearGradient>
    </defs>

    <g transform="translate(100,100)">
      <!-- Arc background -->
      <path d="M-80 0 A80 80 0 0 1 80 0" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="18" stroke-linecap="round"/>
      <!-- Color arc -->
      <path d="M-80 0 A80 80 0 0 1 80 0" fill="none" stroke="url(#g1)" stroke-width="18" stroke-linecap="round" stroke-dasharray="250" stroke-dashoffset="{dashoffset}"/>
      <!-- Tick marks -->
      <!-- needle pivot -->
      <circle cx="0" cy="0" r="4" fill="#111827" stroke="#ffffff" stroke-width="1"/>
      <!-- Needle -->
      <g transform="rotate({angle})">
        <rect x="-2.5" y="-2" width="85" height="4" rx="2" ry="2" fill="#111827" />
        <rect x="-2" y="-1.5" width="80" height="3" rx="1.5" ry="1.5" fill="#ffffff" opacity="0.9"/>
      </g>
      <!-- center label -->
      <text x="0" y="30" text-anchor="middle" fill="#cbd5e1" font-size="12">{label}</text>
      <text x="0" y="50" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="700">{value}%</text>
    </g>
  </svg>
</div>

<style>
@keyframes sweep {{
  from {{ transform: rotate(-90deg); }}
  to {{ transform: rotate({angle}deg); }}
}}
svg#gauge rect {{ transition: transform 1.3s cubic-bezier(.2,.9,.2,1); transform-origin: 100px 100px; }}
</style>
"""

def render_svg_gauge(value:int, width="320px", height="200px", label="Scam Score"):
    # angle range: -90deg =>  -90 to +90 (for 0..100)
    angle = -90 + (value/100.0)*180
    # dashoffset to reveal color arc: we map value to dashoffset (approx)
    full = 250
    dashoffset = full - (value/100.0)*full
    svg = SVG_TEMPLATE.format(angle=angle, value=value, dashoffset=dashoffset, label=label, width=width, height=height)
    components.html(svg, height=height, scrolling=False)

# ---------------------------
# Transcribe wrapper (Whisper if available)
# ---------------------------
def transcribe_with_whisper(tmp_wav_path):
    if not USE_WHISPER or whisper_model is None:
        return "[whisper-not-available]"
    try:
        res = whisper_model.transcribe(tmp_wav_path)
        return res.get("text","").strip()
    except Exception as e:
        return f"[transcription_error: {e}]"

# ---------------------------
# Main UI Styling (glass + dark)
# ---------------------------
st.markdown("""
<style>
body { background: linear-gradient(135deg,#071028 0%, #0b1220 100%); color: #e6eef8; }
.card { background: rgba(255,255,255,0.03); padding:18px; border-radius:14px; border:1px solid rgba(255,255,255,0.04); box-shadow: 0 6px 30px rgba(2,6,23,0.6); }
.top-nav { display:flex; gap:18px; align-items:center; padding:6px 0; }
.top-nav .link { color:#a5b4fc; font-weight:600; cursor:pointer; padding:6px 10px; border-radius:8px; text-decoration:none; }
.top-title { font-size:28px; font-weight:800; background: linear-gradient(90deg,#06b6d4,#3b82f6,#8b5cf6); -webkit-background-clip:text; color:transparent; }
.small-muted { color:#9aa7bf; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Top navigation links (heading style)
# ---------------------------
if "nav" not in st.session_state:
    st.session_state.nav = "Home"
if "user" not in st.session_state:
    st.session_state.user = None

nav_cols = st.columns([1,6,3])
with nav_cols[0]:
    st.markdown(f"<div class='top-title'>Cyber Guard AI</div>", unsafe_allow_html=True)
with nav_cols[1]:
    links_html = """
    <div class='top-nav'>
      <a class='link' href='javascript:window.parent.postMessage({nav:'Home'}, "*")'>Home</a>
      <a class='link' href='javascript:window.parent.postMessage({nav:'Analyze'}, "*")'>Analyze</a>
      <a class='link' href='javascript:window.parent.postMessage({nav:'History'}, "*")'>History</a>
      <a class='link' href='javascript:window.parent.postMessage({nav:'Login'}, "*")'>Login / Create Account</a>
    </div>
    <script>
      window.addEventListener("message", (e) => {
        try {
          const nav = e.data.nav;
          if(nav) {
            const url = new URL(window.location);
            url.searchParams.set("nav", nav);
            window.history.pushState({}, "", url);
          }
        } catch(e) {}
      });
    </script>
    """
    st.markdown(links_html, unsafe_allow_html=True)
with nav_cols[2]:
    if st.session_state.get("user"):
        st.markdown(f"<div class='small-muted'>Signed in as <b>{st.session_state['user']}</b></div>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()
    else:
        st.markdown("<div class='small-muted'>Not signed in</div>", unsafe_allow_html=True)

# Capture nav from query param (so JS links can set it)
query_nav = st.experimental_get_query_params().get("nav", [])
if query_nav:
    st.session_state.nav = query_nav[0]

# ---------------------------
# Pages: Home, Analyze, History, Login/Create
# ---------------------------
nav = st.session_state.nav

# ---------------------------
# HOME
# ---------------------------
if nav == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Welcome to Cyber Guard AI — Premium")
    st.markdown("Enterprise grade scam & fraud detection for voice calls. Upload an audio clip, transcribe with Whisper (if available), classify using Groq (if configured), and get an explainable result with waveform + spectrogram overlays.")
    st.markdown("### Highlights")
    st.markdown("- Animated SVG gauge (fast, elegant)")
    st.markdown("- Waveform + spectrogram overlays for quick audio forensics")
    st.markdown("- Save history & download forensic reports")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# ANALYZE
# ---------------------------
elif nav == "Analyze":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Analyze a Call")
    st.markdown("Upload audio (wav/mp3/m4a) from your laptop, or paste a transcript. The app will transcribe (Whisper) and call Groq for JSON decision-making if `GROQ_API_KEY` is set in the environment.")
    left, right = st.columns([1.2,1])

    with left:
        uploaded = st.file_uploader("Upload audio file (wav / mp3 / m4a)", type=["wav","mp3","m4a"], accept_multiple_files=False)
        use_demo = st.button("Use sample demo audio")
        manual_text = st.text_area("Optional: paste transcript or notes", height=140)
        model_choice = st.selectbox("Groq model (used only if backend key present)", ["mixtral-8x7b","gpt-4o-mini"])
        analyze_btn = st.button("🔎 Start Analysis", key="analyze_btn")

    with right:
        st.markdown("### Live Preview")
        if uploaded:
            try:
                st.audio(uploaded)
            except:
                st.info("Preview not available here.")
        else:
            st.info("No file uploaded yet.")

    # sample audio: generate short tone saved to BytesIO
    if use_demo and not uploaded:
        sr = 16000
        t = np.linspace(0,2.5,int(sr*2.5),False)
        tone = 0.02*np.sin(2*np.pi*440*t)
        buf = BytesIO()
        sf.write(buf, tone, sr, format="WAV")
        buf.seek(0)
        uploaded = buf

    if analyze_btn:
        if uploaded is None and (not manual_text or manual_text.strip()==""):
            st.error("Please upload audio file or paste transcript text.")
            st.stop()

        # prepare audio_data for backend
        audio_data = None
        tmp_wav = None
        if uploaded:
            # ensure bytes
            try:
                uploaded.seek(0)
            except:
                pass
            raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
            # convert to wav file on disk for whisper and librosa
            tmpdir = tempfile.mkdtemp()
            tmp_wav = os.path.join(tmpdir, "input.wav")
            try:
                # try to use soundfile to write
                x, sr = librosa.load(BytesIO(raw), sr=16000, mono=True)
                sf.write(tmp_wav, x, sr, format="WAV")
                audio_data = (x, sr)
            except Exception as e:
                # fallback: try ffmpeg external (not included)
                st.warning(f"Audio decode issue: {e}. Attempting best-effort proceed without audio features.")
                audio_data = None

        # Transcription (try Whisper if available)
        transcription = ""
        if audio_data and tmp_wav and USE_WHISPER:
            transcription = transcribe_with_whisper(tmp_wav)
        else:
            transcription = manual_text or "[no-transcription]"

        # Extract features if audio_data
        features = extract_audio_features_from_array(audio_data[0], audio_data[1]) if audio_data else {}

        emotion = heuristic_emotion_label(features)

        # Groq call if key present
        classification = None
        if GROQ_KEY:
            prompt = build_groq_prompt(transcription or "[no transcription]", emotion, features)
            try:
                api_resp = call_groq_chat(GROQ_KEY, prompt, model=model_choice)
                assistant_text = None
                if "choices" in api_resp and len(api_resp["choices"])>0:
                    ch = api_resp["choices"][0]
                    if "message" in ch and "content" in ch["message"]:
                        assistant_text = ch["message"]["content"]
                    elif "text" in ch:
                        assistant_text = ch["text"]
                if assistant_text:
                    m = re.search(r"\{.*\}", assistant_text, flags=re.DOTALL)
                    if m:
                        classification = json.loads(m.group(0))
                if classification is None:
                    classification = local_heuristic_classifier(transcription, emotion, features)
            except Exception as e:
                st.warning(f"Groq API error: {e}. Using local heuristic.")
                classification = local_heuristic_classifier(transcription, emotion, features)
        else:
            classification = local_heuristic_classifier(transcription, emotion, features)

        # Build result
        result = {
            "transcription": transcription,
            "emotion": emotion,
            "features": features,
            "classification": classification
        }

        # Save to history
        hist = load_history()
        entry = {"id": len(hist)+1, "ts": datetime.utcnow().isoformat()+"Z", "result": result}
        hist.insert(0, entry)
        save_history(hist)

        # Show results elegantly
        st.markdown("<hr style='border:0;height:1px;background:#1f2937'>", unsafe_allow_html=True)
        st.markdown("## Analysis Result")
        main, side = st.columns([1.4,1])
        with main:
            st.markdown("### 🔎 Transcription")
            st.write(result["transcription"])
            st.markdown("### 🧭 Emotion (estimated)")
            st.info(result["emotion"])
            st.markdown("### 🔍 Reasons (explainability)")
            for r in result["classification"].get("reasons",[]):
                st.write(f"- {r}")
            st.markdown("### 🗣 Safe response suggestion")
            st.success(result["classification"].get("safe_response","—"))
            st.markdown("### Downloads")
            # downloads
            make_download_bytes(result, filename=f"analysis_{entry['id']}.json")
            report_txt = "Cyber Guard AI — Analysis Report\n\n"
            report_txt += f"Timestamp: {entry['ts']}\n\nTranscription:\n{result['transcription']}\n\n"
            report_txt += f"Emotion: {result['emotion']}\n\nClassification:\n{json.dumps(result['classification'], indent=2)}\n"
            make_download_txt(report_txt, filename=f"analysis_{entry['id']}_report.txt")
        with side:
            st.markdown("### 🚨 Scam Gauge")
            render_svg_gauge(result["classification"].get("confidence",0), width="320px", height="200px")
            st.markdown("### 🔊 Waveform + Spectrogram")
            if audio_data:
                y,sr = audio_data
                # waveform
                fig1, ax1 = plt.subplots(figsize=(6,1.6))
                times = np.linspace(0, len(y)/sr, num=len(y))
                ax1.plot(times, y, linewidth=0.6)
                ax1.set_yticks([])
                ax1.set_xlabel("Time (s)")
                ax1.set_title("Waveform")
                st.pyplot(fig1)

                # spectrogram overlay
                D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
                DB = librosa.amplitude_to_db(D, ref=np.max)
                fig2, ax2 = plt.subplots(figsize=(6,2.2))
                img = librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='log', ax=ax2)
                ax2.set_title("Spectrogram (dB, log scale)")
                st.pyplot(fig2)
            else:
                st.info("No decoded audio to display waveform/spectrogram.")

# ---------------------------
# HISTORY (no raw JSON printed)
# ---------------------------
elif nav == "History":
    st.markdown("## Analysis History")
    history = load_history()
    if not history:
        st.info("No history yet — analyze a call to create entries.")
    else:
        import pandas as pd
        table = []
        for item in history:
            r = item["result"]["classification"]
            table.append({"id": item["id"], "ts": item["ts"], "label": r.get("label",""), "confidence": r.get("confidence",0)})
        df = pd.DataFrame(table)
        st.table(df)  # simple, clean table

        st.markdown("### View entry details")
        sel = st.number_input("Enter ID to view details", min_value=1, max_value=len(history), value=history[0]["id"])
        chosen = next((h for h in history if h["id"]==sel), None)
        if chosen:
            st.markdown(f"**Timestamp:** {chosen['ts']}")
            st.markdown(f"**Label:** {chosen['result']['classification'].get('label','')}")
            st.markdown(f"**Confidence:** {chosen['result']['classification'].get('confidence',0)}%")
            st.markdown("**Transcription:**")
            st.write(chosen["result"].get("transcription",""))
            st.markdown("**Reasons:**")
            for r in chosen["result"]["classification"].get("reasons",[]):
                st.write(f"- {r}")
            # downloads only
            make_download_bytes(chosen, filename=f"history_{sel}.json")
            make_download_txt("Cyber Guard AI — Saved Report\n\n"+json.dumps(chosen,indent=2), filename=f"history_{sel}.txt")

# ---------------------------
# LOGIN / CREATE ACCOUNT (simple demo)
# ---------------------------
elif nav == "Login":
    st.markdown("## Login / Create Account")
    users = load_users()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Login")
        uname = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login"):
            if uname in users and users[uname]["pw"] == hash_pw(pw):
                st.success("Logged in")
                st.session_state.user = uname
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        st.markdown("### Create account")
        new_user = st.text_input("Choose username", key="new_user")
        new_pw = st.text_input("Choose password", type="password", key="new_pw")
        if st.button("Create account"):
            if not new_user or not new_pw:
                st.error("Username and password required")
            elif new_user in users:
                st.error("Username already exists")
            else:
                users[new_user] = {"pw": hash_pw(new_pw), "created": datetime.utcnow().isoformat()+"Z"}
                save_users(users)
                st.success("Account created — you can now login")

# ---------------------------
# End of file
# ---------------------------

