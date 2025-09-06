# Summarizer (Text • OCR • Audio)

A Python + Streamlit project that performs **faithful, concise, and context-aware summarization** across **text**, **images (OCR)**, and **audio transcripts**.  
It is designed to **preserve meaning**, avoid hallucinations, and adapt to different input sources (meeting transcripts, slides, or short voice notes).  

The repository contains:
- `summarizer.py` → core summarization logic (text, OCR, audio)  
- `app.py` → Streamlit web UI for interactive use  

---

## 📖 Project Description

The Summarizer project is a Python-based application designed to generate accurate, concise, and context-aware summaries from multiple input modalities: plain text, images (via OCR), and audio recordings. It combines **rule-based extractive methods** with **transformer-based abstractive summarization** to provide outputs that are both faithful to the source and easy to consume.  

### Motivation
Modern meetings, presentations, and recordings often produce large amounts of unstructured information that can be difficult to review and share. Traditional summarization tools tend to either hallucinate content (abstractive methods) or miss key details (extractive methods). This project was developed to balance both approaches, producing summaries that are grounded, factual, and tailored to practical use cases such as **meeting minutes, slide decks, or voice memos**.  

### Design
The system is built around a modular core (`summarizer.py`) with three distinct pipelines:
- **Text summarization**  
  - Supports two styles:  
    - *Faithful*: prioritizes accuracy, light rewriting, and readability. Uses Hugging Face’s [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) for abstractive summarization of longer texts, with extractive fallback when transformers are unavailable.  
    - *Concise*: selects the most salient sentences using heuristics that prioritize deadlines, names, and action items. Produces short, actionable summaries.  
- **Slide-aware OCR summarization**  
  - Uses Tesseract OCR to extract text from images.  
  - Detects slide-like structures (titles, bullet points) and condenses them into compact sentences.  
- **Audio summarization**  
  - Transcribes speech with the `SpeechRecognition` library.  
  - Cleans filler words, adds sentence boundaries, and adapts style automatically for short transcripts.  

The `app.py` module provides a **Streamlit-based web application** that exposes these capabilities in an interactive browser UI. Users can paste text, upload images of slides, or upload audio recordings, and receive faithful, concise summaries without using the command line.

### Features
- **Multi-modal support**: text, image (OCR), and audio inputs.  
- **Configurable styles**: choose between faithful (abstractive + extractive) or concise (salience-driven) summaries.  
- **No hallucinations**: summaries never invent facts; outputs are grounded in the source text.  
- **Streamlit UI**: provides a simple interface for non-technical users.  
- **Extensible**: designed with modular pipelines for easy integration of new models or input types.  

### Technology Stack
- **Python 3.10+**  
- **Hugging Face Transformers** (for abstractive summarization with DistilBART CNN)  
- **Regex + scoring heuristics** (for extractive and concise summarization)  
- **Pillow + pytesseract + Tesseract OCR** (for image/slide text extraction)  
- **SpeechRecognition + PyAudio/PortAudio** (for speech-to-text transcription)  
- **Streamlit** (for interactive web UI)  

### Intended Use Cases
- **Meeting notes**: convert long transcripts into crisp summaries with deadlines and action items highlighted.  
- **Slide decks**: automatically condense presentations into 1–2 sentences per slide.  
- **Voice memos**: transcribe and summarize audio recordings for quick review.  
- **General summarization**: process any long-form text into digestible summaries.  

This project emphasizes **practicality, accuracy, and accessibility**, ensuring that both technical and non-technical users can benefit from automated summarization in everyday workflows.


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
