# Streamlit UI for Summarizer (Text • OCR • Audio) with Faithful/Concise styles + Slide-aware OCR + Audio compaction
import os
import tempfile
from pathlib import Path
import streamlit as st

from summarizer import (
    summarize_text,
    summarize_audio,
    summarize_ocr,
)

st.set_page_config(page_title="Summarizer (Text • OCR • Audio)", page_icon="📝", layout="wide")
st.title("📝 Summarizer (Text • OCR • Audio)")
st.caption("Faithful & Concise styles. Slide-aware OCR. Audio short-text compaction to avoid verbatim output.")

# Sidebar options (global)
st.sidebar.header("Settings")
style = st.sidebar.radio("Summary style (Text/OCR default)", ["Faithful", "Concise"], index=1, key="style")
mode = st.sidebar.selectbox(
    "Summarization engine",
    options=["auto", "extractive"],
    help="auto = use transformers if available; extractive = deterministic, no generation.",
    key="mode",
)
max_len = st.sidebar.slider("Max summary length (faithful; approx. tokens)", 60, 400, 160, 10, key="max_len")
max_sentences = st.sidebar.slider("Max sentences (concise & faithful fallback)", 3, 12, 7, 1, key="max_sentences")

st.sidebar.divider()
st.sidebar.markdown(
    "- **Concise**: selects up to *Max sentences*; no invented facts.\n"
    "- **Faithful**: lightly polishes; if it falls back to extractive, it uses *Max sentences*.\n"
    "- **Slide-aware OCR** collapses bullets into 1–2 sentences.\n"
    "- **Audio** can auto-switch to Concise for short transcripts."
)

tabs = st.tabs(["Text", "Audio (WAV/AIFF/FLAC)", "Image (OCR)"])

# ---------- Text tab (FORM) ----------
with tabs[0]:
    st.subheader("Summarize Text")
    with st.form("text_form", clear_on_submit=False):
        txt = st.text_area("Paste text:", height=300, placeholder="Paste your text here…", key="text_area")
        submitted = st.form_submit_button("Summarize Text", use_container_width=True)
    if submitted:
        if not txt.strip():
            st.error("Please paste some text.")
        else:
            with st.spinner("Summarizing…"):
                summary = summarize_text(
                    txt,
                    mode=st.session_state["mode"],
                    style=st.session_state["style"].lower(),
                    max_len=st.session_state["max_len"],
                    max_sentences=st.session_state["max_sentences"],
                )
            st.subheader("Summary")
            st.write(summary if summary else "(No summary generated)")
            st.download_button("⬇️ Download Summary", data=summary, file_name="summary.txt", mime="text/plain", use_container_width=True)

# ---------- Audio tab (FORM) ----------
with tabs[1]:
    st.subheader("Summarize Audio")
    with st.form("audio_form", clear_on_submit=False):
        use_offline = st.checkbox("Offline transcription (PocketSphinx)", value=False, key="audio_offline")
        force_concise_if_short = st.checkbox("Force concise for short transcripts", value=True, key="audio_force_concise")
        short_threshold_chars = st.slider("Short transcript threshold (chars)", 120, 500, 220, 10, key="audio_short_thresh")
        audio_sent_cap = st.slider("Audio max sentences (when concise)", 2, 8, 3, 1, key="audio_sent_cap")
        audio = st.file_uploader("Upload audio file", type=["wav", "aiff", "flac"], key="audio_uploader")
        submitted_audio = st.form_submit_button("Transcribe + Summarize", use_container_width=True)

    if submitted_audio:
        if not audio:
            st.error("Please upload an audio file.")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.name).suffix or ".wav")
            tmp.write(audio.getvalue()); tmp.flush(); tmp.close()
            try:
                with st.spinner("Transcribing and summarizing…"):
                    transcript, summary = summarize_audio(
                        tmp.name,
                        use_offline=st.session_state["audio_offline"],
                        mode=st.session_state["mode"],
                        style=st.session_state["style"].lower(),         # default style; may be overridden for short
                        max_len=st.session_state["max_len"],
                        max_sentences=st.session_state["max_sentences"], # used for faithful fallback/long
                        force_concise_if_short=st.session_state["audio_force_concise"],
                        short_threshold_chars=st.session_state["audio_short_thresh"],
                        audio_sentences_cap=st.session_state["audio_sent_cap"],
                    )
                st.subheader("Transcript")
                st.write(transcript if transcript else "(empty)")
                st.subheader("Summary")
                st.write(summary if summary else "(No summary generated)")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇️ Download Transcript", data=transcript, file_name="transcript.txt", mime="text/plain", use_container_width=True)
                with col2:
                    st.download_button("⬇️ Download Summary", data=summary, file_name="summary.txt", mime="text/plain", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp.name)

# ---------- OCR tab (FORM) ----------
with tabs[2]:
    st.subheader("Summarize Image (OCR)")
    slide_aware = st.checkbox("Slide-aware OCR summarization (collapse bullets)", value=True, key="slide_aware")
    t_path = st.text_input("Optional: Tesseract path (Windows)", value=os.environ.get("TESSERACT_CMD", ""), key="tesseract_path")
    if t_path:
        os.environ["TESSERACT_CMD"] = t_path

    with st.form("ocr_form", clear_on_submit=False):
        image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="image_uploader")
        submitted_ocr = st.form_submit_button("OCR + Summarize", use_container_width=True)
    if submitted_ocr:
        if not image:
            st.error("Please upload an image.")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.name).suffix or ".png")
            tmp.write(image.getvalue()); tmp.flush(); tmp.close()
            try:
                with st.spinner("Running OCR and summarizing…"):
                    ocr_text, summary = summarize_ocr(
                        tmp.name,
                        mode=st.session_state["mode"],
                        style=st.session_state["style"].lower(),
                        max_len=st.session_state["max_len"],
                        max_sentences=st.session_state["max_sentences"],
                        slide_aware=st.session_state["slide_aware"],
                    )
                st.subheader("OCR Text")
                st.write(ocr_text if ocr_text else "(empty)")
                st.subheader("Summary")
                st.write(summary if summary else "(No summary generated)")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇️ Download OCR Text", data=ocr_text, file_name="ocr.txt", mime="text/plain", use_container_width=True)
                with col2:
                    st.download_button("⬇️ Download Summary", data=summary, file_name="summary.txt", mime="text/plain", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp.name)
