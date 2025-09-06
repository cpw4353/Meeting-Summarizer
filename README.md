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

