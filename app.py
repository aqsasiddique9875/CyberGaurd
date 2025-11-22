"""
Cyber Guard AI — Premium Frontend (Streamlit)
Design: full-width hero with dark overlay (uses /mnt/data/maker.png),
top-right navigation, centered upload + analyze button (like the provided screenshot),
animated SVG gauge, waveform + spectrogram overlays, Whisper (optional) + Groq (optional).
History page (no raw JSON).
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO
import json, os, tempfile, shutil, re, requests
from datetime import datetime
import streamlit.components.v1 as components
from PIL import Image

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Cyber Guard AI", page_icon="🛡️", layout="wide")
HISTORY_PATH = "history.json"
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY = os.environ.get("GROQ_API_KEY", None)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# Use uploaded file path as hero background image (developer provided)
HERO_IMAGE_PATH = "/mnt/data/maker.png"  # <- using the uploaded image path you provided

# -----------------------------
# Optional: Whisper support
# -----------------------------
USE_WHISPER = False
try:
    import whisper
    whisper_model = whisper.load_model(WHISPER_MODEL)
    USE_WHISPER = True
except Exception:
    whisper_model = None
    USE_WHISPER = False

# -----------------------------
# Utilities
# -----------------------------
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_history(h):
    with open(HISTORY_PATH, "w") as f:
        json.dump(h, f, indent=2, default=str)

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

def extract_audio_features(y, sr=16000):
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
    return {"label": label, "confidence": confidence, "reasons": reasons, "safe_response": "I don't share personal info on calls; please send in writing."}

# -----------------------------
# SVG gauge template (animated needle)
# -----------------------------
SVG_TEMPLATE = """
<div style="width:100%;display:flex;justify-content:center;margin-bottom:8px;">
  <svg viewBox="0 0 200 120" width="{width}" height="{height}">
    <defs>
      <linearGradient id="g1" x1="0" x2="1">
        <stop offset="0%" stop-color="#0fbf6f"/>
        <stop offset="50%" stop-color="#ffb020"/>
        <stop offset="100%" stop-color="#ff4d4d"/>
      </linearGradient>
    </defs>
    <g transform="translate(100,100)">
      <path d="M-80 0 A80 80 0 0 1 80 0" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="18" stroke-linecap="round"/>
      <path d="M-80 0 A80 80 0 0 1 80 0" fill="none" stroke="url(#g1)" stroke-width="18" stroke-linecap="round" stroke-dasharray="250" stroke-dashoffset="{dashoffset}"/>
      <circle cx="0" cy="0" r="4" fill="#0b1220" stroke="#ffffff" stroke-width="1"/>
      <g transform="rotate({angle})">
        <rect x="-2.5" y="-2" width="85" height="4" rx="2" ry="2" fill="#0b1220" />
        <rect x="-2" y="-1.5" width="80" height="3" rx="1.5" ry="1.5" fill="#ffffff" opacity="0.95"/>
      </g>
      <text x="0" y="30" text-anchor="middle" fill="#cbd5e1" font-size="12">{label}</text>
      <text x="0" y="50" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="700">{value}%</text>
    </g>
  </svg>
</div>
<style>
svg rect {{ transition: transform 1.0s cubic-bezier(.2,.9,.2,1); transform-origin: 100px 100px; }}
</style>
"""

def render_svg_gauge(value:int, width="320px", height="200px", label="Scam Score"):
    angle = -90 + (value/100.0)*180
    full = 250
    dashoffset = full - (value/100.0)*full
    svg = SVG_TEMPLATE.format(angle=angle, value=value, dashoffset=dashoffset, label=label, width=width, height=height)
    components.html(svg, height=220, scrolling=False)

# -----------------------------
# UI CSS (hero + nav)
# -----------------------------
background_css = ""
if os.path.exists(HERO_IMAGE_PATH):
    # use file:// URL for local hero image
    background_css = f"""
    <style>
    .hero {{
      position: relative;
      height: 520px;
      color: white;
      display:flex;
      align-items:center;
      justify-content:center;
      text-align:center;
      background-image: url("file://{HERO_IMAGE_PATH}");
      background-size: cover;
      background-position: center;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(3,7,18,0.68), rgba(3,7,18,0.82));
      backdrop-filter: blur(2px);
    }}
    .hero-content {{
      position: relative;
      z-index: 2;
      max-width: 980px;
      padding: 30px;
    }}
    .hero-title {{
      font-size: 52px;
      font-weight: 800;
      margin-bottom: 12px;
      background: linear-gradient(90deg,#06b6d4,#3b82f6,#8b5cf6);
      -webkit-background-clip:text;
      color:transparent;
    }}
    .hero-sub {{
      color: #cbd5e1;
      font-size: 18px;
      margin-bottom: 26px;
    }}
    .hero-input {{
      display:flex;
      justify-content:center;
      gap:14px;
      margin-top:12px;
    }}
    .upload-box {{
      background: rgba(255,255,255,0.95);
      padding: 14px 18px;
      border-radius: 8px;
      width: 640px;
      display:flex;
      align-items:center;
      gap:12px;
      box-shadow: 0 6px 20px rgba(2,6,23,0.5);
    }}
    .upload-text {{
      color:#0b1220;
      font-weight:600;
      width:100%;
    }}
    .analyze-btn {{
      background: linear-gradient(90deg,#f59e0b,#d97706);
      color:white;
      padding: 12px 18px;
      border-radius:8px;
      border:none;
      font-weight:700;
    }}
    /* top nav */
    .topbar {{
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:12px 26px;
      color: #e6eef8;
    }}
    .nav-links {{ display:flex; gap:18px; align-items:center; }}
    .nav-link {{ color:#cbd5e1; font-weight:600; text-decoration:none; }}
    </style>
    """
else:
    background_css = "<style>.hero{height:300px;background:#091226}</style>"

st.markdown(background_css, unsafe_allow_html=True)

# -----------------------------
# Top Navigation (right aligned)
# -----------------------------
nav_bar = """
<div class="topbar">
  <div style="font-weight:700;font-size:18px;color:#e6eef8">Cyber Guard AI</div>
  <div class="nav-links">
    <a class="nav-link" href="javascript:window.parent.postMessage({nav:'Home'}, '*')">Home</a>
    <a class="nav-link" href="javascript:window.parent.postMessage({nav:'Analyze'}, '*')">Analyze</a>
    <a class="nav-link" href="javascript:window.parent.postMessage({nav:'History'}, '*')">History</a>
    <a class="nav-link" href="javascript:window.parent.postMessage({nav:'Login'}, '*')">Login</a>
  </div>
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
st.markdown(nav_bar, unsafe_allow_html=True)

# Load / persist navigation selection via query param
if "nav" not in st.session_state:
    st.session_state.nav = "Home"
query_nav = st.experimental_get_query_params().get("nav", [])
if query_nav:
    st.session_state.nav = query_nav[0]

# -----------------------------
# Pages: Home, Analyze, History, Login
# -----------------------------
nav = st.session_state.nav

# -----------------------------
# HOME (hero like screenshot)
# -----------------------------
if nav == "Home":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-content">
        <div class="hero-title">Cyber Guard AI</div>
        <div class="hero-sub">Fraud & Scam Call Detection Made Simple.<br> Let Cyber Guard AI analyze and protect you.</div>
        <div class="hero-input">
          <div class="upload-box">
            <div class="upload-text">Upload audio from laptop or paste transcript in Analyze page</div>
          </div>
          <button class="analyze-btn" onclick="window.parent.postMessage({{nav:'Analyze'}}, '*')">Start Analysis</button>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# ANALYZE page
# -----------------------------
elif nav == "Analyze":
    st.markdown('<div style="padding:20px 28px">', unsafe_allow_html=True)
    st.markdown("## Analyze a Call", unsafe_allow_html=True)
    st.markdown("Upload an audio file from your laptop (wav/mp3/m4a) or paste a short transcript. The app will transcribe (Whisper, if available) and classify (Groq if API key present) and show waveform + spectrogram with an animated gauge.")
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        uploaded = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"])
        manual_text = st.text_area("Optional: paste call text / transcript", height=120)
        st.markdown("**Model (Groq)** — used only if GROQ_API_KEY set in environment.")
        model_choice = st.selectbox("Groq model", ["mixtral-8x7b", "gpt-4o-mini"])
        analyze_btn = st.button("🔎 Analyze", key="analyze_action")

        # sample button
        if st.button("Load Sample (3s tone)"):
            sr = 16000
            t = np.linspace(0,3.0,int(sr*3.0),False)
            tone = 0.02*np.sin(2*np.pi*440*t)
            buf = BytesIO()
            sf.write(buf, tone, sr, format="WAV")
            buf.seek(0)
            uploaded = buf
            st.session_state._sample_loaded = True

    with right:
        st.markdown("### Live preview")
        if uploaded:
            try:
                st.audio(uploaded)
            except:
                st.info("Preview not available.")

    # Analyze action
    if analyze_btn:
        if uploaded is None and (not manual_text or manual_text.strip()==""):
            st.error("Please upload audio or provide text.")
            st.stop()

        # prepare audio -> (y,sr)
        audio_data = None
        tmp_wav = None
        if uploaded:
            try:
                uploaded.seek(0)
            except:
                pass
            raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
            # try librosa decode
            try:
                y, sr = librosa.load(BytesIO(raw), sr=16000, mono=True)
                audio_data = (y, sr)
                # write to tmp wav for whisper
                tmpdir = tempfile.mkdtemp()
                tmp_wav = os.path.join(tmpdir, "input.wav")
                sf.write(tmp_wav, y, sr, format="WAV")
            except Exception as e:
                st.warning(f"Audio decode failed: {e}. Proceeding without audio features.")
                audio_data = None

        # transcription via Whisper if available, else manual or placeholder
        transcription = ""
        if audio_data and tmp_wav and USE_WHISPER:
            try:
                with st.spinner("Transcribing with Whisper..."):
                    res = whisper_model.transcribe(tmp_wav)
                    transcription = res.get("text","").strip()
            except Exception as e:
                st.warning(f"Whisper error: {e}")
                transcription = manual_text or "[no-transcription]"
        else:
            transcription = manual_text or "[no-transcription]"

        # audio features
        features = extract_audio_features(audio_data[0], audio_data[1]) if audio_data else {}

        emotion = heuristic_emotion_label(features)

        # classification: Groq if key set, else local heuristic
        classification = None
        if GROQ_KEY:
            prompt = build_groq_prompt(transcription, emotion, features)
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

        # result object
        result = {
            "transcription": transcription,
            "emotion": emotion,
            "features": features,
            "classification": classification
        }

        # save history
        hist = load_history()
        entry = {"id": len(hist)+1, "ts": datetime.utcnow().isoformat()+"Z", "result": result}
        hist.insert(0, entry)
        save_history(hist)

        # Display results
        st.markdown("---")
        st.markdown("## Result")
        colA, colB = st.columns([1.4, 1])
        with colA:
            st.markdown("### 🔎 Transcription")
            st.write(result["transcription"])

            st.markdown("### 🧭 Emotion")
            st.info(result["emotion"])

            st.markdown("### 🔍 Reasons")
            for r in result["classification"].get("reasons",[]):
                st.write(f"- {r}")

            st.markdown("### 🗣 Safe Response")
            st.success(result["classification"].get("safe_response","—"))

            # downloads
            st.download_button("⬇️ Download JSON", data=json.dumps(result, indent=2).encode("utf-8"),
                               file_name=f"analysis_{entry['id']}.json", mime="application/json")
            report_txt = f"Cyber Guard AI — Analysis Report\nTimestamp: {entry['ts']}\n\nTranscription:\n{result['transcription']}\n\nEmotion: {result['emotion']}\n\nClassification:\n{json.dumps(result['classification'], indent=2)}\n"
            st.download_button("⬇️ Download Report (TXT)", data=report_txt.encode("utf-8"), file_name=f"analysis_{entry['id']}_report.txt", mime="text/plain")

        with colB:
            st.markdown("### 🚨 Scam Gauge")
            render_svg_gauge(result["classification"].get("confidence",0), width="320px", height="200px")

            st.markdown("### 🔊 Waveform")
            if audio_data:
                y,sr = audio_data
                fig_w, axw = plt.subplots(figsize=(6,1.6))
                times = np.linspace(0, len(y)/sr, num=len(y))
                axw.plot(times, y, linewidth=0.6, color="#67e8f9")
                axw.set_yticks([])
                axw.set_xlabel("Time (s)")
                axw.set_title("Waveform")
                st.pyplot(fig_w)

                st.markdown("### 🎚 Spectrogram")
                D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
                DB = librosa.amplitude_to_db(D, ref=np.max)
                fig_s, ax_s = plt.subplots(figsize=(6,2.2))
                import librosa.display
                librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='log', ax=ax_s, cmap='magma')
                ax_s.set_title("Spectrogram (dB)")
                st.pyplot(fig_s)
            else:
                st.info("No audio decoded to show waveform/spectrogram.")

# -----------------------------
# HISTORY (clean table; no JSON)
# -----------------------------
elif nav == "History":
    st.markdown("## Analysis History")
    history = load_history()
    if not history:
        st.info("No analyses yet — run an analysis to populate history.")
    else:
        import pandas as pd
        rows = []
        for h in history:
            r = h["result"]["classification"]
            rows.append({"ID": h["id"], "Timestamp": h["ts"], "Label": r.get("label",""), "Confidence": r.get("confidence",0)})
        df = pd.DataFrame(rows)
        st.table(df)

        st.markdown("### View details")
        selected = st.number_input("Select ID", min_value=1, max_value=len(history), value=history[0]["id"])
        chosen = next((x for x in history if x["id"]==selected), None)
        if chosen:
            st.markdown(f"**Timestamp:** {chosen['ts']}")
            st.markdown(f"**Label:** {chosen['result']['classification'].get('label','')}")
            st.markdown(f"**Confidence:** {chosen['result']['classification'].get('confidence',0)}%")
            st.markdown("**Transcription:**")
            st.write(chosen["result"].get("transcription",""))
            st.markdown("**Reasons:**")
            for r in chosen["result"]["classification"].get("reasons",[]):
                st.write(f"- {r}")

            # downloads
            st.download_button("⬇️ Download Entry JSON", data=json.dumps(chosen, indent=2).encode("utf-8"),
                               file_name=f"history_{selected}.json", mime="application/json")
            st.download_button("⬇️ Download Entry TXT", data=json.dumps(chosen, indent=2).encode("utf-8"),
                               file_name=f"history_{selected}.txt", mime="text/plain")

# -----------------------------
# LOGIN page (simple UI hookup)
# -----------------------------
elif nav == "Login":
    st.markdown("## Login / Create Account (demo)")
    st.markdown("This demo app includes a simple local account option (users stored in `users.json`) — for production use a real auth provider.")
    from pathlib import Path
    USERS_PATH = "users.json"
    def load_users():
        if os.path.exists(USERS_PATH):
            with open(USERS_PATH,"r") as f:
                return json.load(f)
        return {}
    def save_users(u):
        with open(USERS_PATH,"w") as f:
            json.dump(u, f, indent=2)
    users = load_users()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Login")
        lu = st.text_input("Username", key="login_user")
        lp = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login account"):
            if lu in users and users[lu]["pw"] == users[lu]["pw"]:
                st.session_state.user = lu
                st.success("Logged in (demo).")
            else:
                st.warning("Invalid (demo) login.")
    with c2:
        st.markdown("### Create account")
        nu = st.text_input("Choose username", key="new_user")
        npw = st.text_input("Choose password", type="password", key="new_pw")
        if st.button("Create account"):
            if not nu or not npw:
                st.error("Enter username & password")
            elif nu in users:
                st.error("Username exists")
            else:
                users[nu] = {"pw": npw, "created": datetime.utcnow().isoformat()+"Z"}
                save_users(users)
                st.success("Account created (demo). You can login now.")

# -----------------------------
# End
# -----------------------------
st.markdown("<br><br><center style='color:#94a3b8'>© Cyber Guard AI — UI Demo</center>", unsafe_allow_html=True)
