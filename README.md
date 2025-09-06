# Summarizer (Text • OCR • Audio)

A Python + Streamlit project that performs **faithful, concise, and context-aware summarization** across **text**, **images (OCR)**, and **audio transcripts**.  
It is designed to **preserve meaning**, avoid hallucinations, and adapt to different input sources (meeting transcripts, slides, or short voice notes).  

The repository contains:
- `summarizer.py` → core summarization logic (text, OCR, audio)  
- `app.py` → Streamlit web UI for interactive use  

---

## 📖 Project Description

This summarizer was built to support **meeting notes and action item extraction**, where accuracy and brevity are critical.  

Key features:
- **Faithful summaries**: fixes run-ons, polishes grammar, uses Hugging Face [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) for abstractive summarization.  
- **Concise summaries**: salience-driven sentence selection to keep outputs short and actionable.  
- **Slide-aware OCR summarization**: extracts and condenses slides into compact sentences.  
- **Audio summarization**: transcribes recordings, removes fillers, and summarizes automatically.  
- **Streamlit UI**: no CLI required — run a local app in your browser.

---

## 📦 Dependencies

Install only what you need (text, OCR, audio, UI).  

### Core
```bash
pip install transformers torch huggingface_hub streamlit
```

### OCR (optional)
```bash
pip install pillow pytesseract
# Also install Tesseract system-wide:
# - Windows: https://github.com/UB-Mannheim/tesseract/wiki
# - macOS: brew install tesseract
# - Linux: sudo apt-get install tesseract-ocr
```

### Audio (optional)
```bash
pip install SpeechRecognition pyaudio
# macOS: brew install portaudio
# Linux: sudo apt-get install portaudio19-dev
```

---

## 🚀 Usage Guide
### 1. Run the Streamlit UI
The `app.py` provides full browser-based interface.

Start it with:
```
streamlit run app.py
```
This will:
- Launch a local web server,
- Open the app in your browser at [http://localhost:8501](http://localhost:8501),
- Let you paste text, upload an image, or upload an audio file for summarization.

### 2. Library usage (for developers/scripts)
Text Summarization
``` python
from summarizer import summarize_text

text = "Emma will deliver the draft by Sept 20 and finalize by Oct 1."
summary = summarize_text(text, style="faithful", max_len=120)
print(summary)
```
OCR(slides/images)
``` python
from summarizer import summarize_ocr

raw_text, summary = summarize_ocr("slide.png", style="concise")
print(raw_text)
print(summary)
```
Audio
``` python
from summarizer import summarize_audio

transcript, summary = summarize_audio("meeting.wav", style="faithful")
print(transcript)
print(summary)
```

---

## 🔍 How it Works
- Faithful: grammar fixes, abstractive via DistilBART CNN, fallback extractive.
- Concise: salience-driven sentence scoring with boosts for deadlines, action verbs, and names.
- OCR: collapses slides into short compact summaries.
- Audio: transcribes speech, removes filler words, adds soft sentence breaks.
- UI: wraps all modes in a Streamlit app for easy interactive use.

---

## ⚙️ Configuration
Environment variables (optional)
- `SUMMARIZER_MODEL`→ override Hugging Face model (default:`sshleifer/distilbart-cnn-12-6`)
- `SUMMARIZER_BEAMS`, `SUMMARIZER_REP_PENALTY`, etc.→ tune abstractive generation

---

## 📂 Repository Structure
``` bash
summarizer.py   # Core summarization logic
app.py          # Streamlit UI
README.md       # User guide
```

---

## ⚠️ Troubleshooting
- Model download slow/fails? → Hugging Face caches models in `~/.cache/huggingface`.
- Tesseract not found? → Install system-wide and set `TESSERACT_CMD` env var if needed.
- PyAudio install errors? → Install PortAudio (`brew install portaudio` or `sudo apt-get install portaudio19-dev`).
- Streamlit not recognized? → Install with `pip install streamlit`.

---

## 📝 Notes
- The default Hugging Face model (`sshleifer/distilbart-cnn-12-6`) is public and will auto-download.
- Streamlit makes the app easy to use for non-developers — no CLI knowledge required.
- Text, OCR, and audio summarization are modular — you can install only what you need.

---

## ⚡ Quick Start (for non-developers)
1. clone the repo
```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
2. Install requirements:
```
pip install transformers torch huggingface_hub streamlit pillow pytesseract SpeechRecognition pyaudio
```
3. Start the app:
```
streamlit run app.py
```
Now open http://localhost:8501 in your browser and start summarizing!
