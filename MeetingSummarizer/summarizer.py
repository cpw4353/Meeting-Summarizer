"""
Summarizer (Text • OCR • Audio) — Faithful & Concise + Slide-aware OCR + Audio short-text compaction
===================================================================================================

- Faithful: run-on fix + micro rewrite (no hallucinations). For long inputs, uses transformers if
  available; otherwise extractive fallback that respects `max_sentences`.
- Concise: salience-scored selection of up to N sentences (user-controlled), then light polishing.
  (No hallucinations; only selects & lightly rewrites original sentences.)
- OCR: optional *slide-aware* mode that collapses title + bullets into 1–2 compact sentences.
- AUDIO: short transcripts are auto-compacted:
    • remove fillers (uh, um, you know…)
    • add light punctuation boundaries
    • optionally force Concise style for short inputs

Python: 3.10+
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple, Dict, List

# ---------------------------- Optional Imports ---------------------------- #
# Flags to record availability of optional dependencies. Each try/except below sets these.
TRANSFORMERS_AVAILABLE = False
SR_AVAILABLE = False
PIL_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    # Hugging Face transformers pipeline (used for abstractive summarization)
    from transformers import pipeline  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    # SpeechRecognition (microphone/file transcription wrapper for Google/Sphinx)
    import speech_recognition as sr  # type: ignore
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

try:
    # Pillow for image loading in OCR flow
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    # pytesseract Python bindings (require native Tesseract installed)
    import pytesseract  # type: ignore
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


# ----------------------------- Text utilities ----------------------------- #
# Small, English-centric stopword list for term-frequency scoring in extractive/concise modes.
_STOPWORDS = {
    "a","an","the","and","or","but","if","while","with","to","of","in","on","for","by","as","at",
    "is","am","are","was","were","be","been","being","do","does","did","doing","have","has","had",
    "i","you","he","she","it","we","they","me","him","her","them","my","your","his","their","our",
    "this","that","these","those","from","up","down","out","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","any","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too","very","can","will",
    "just","don","should","now"
}
# Token regex for words (keeps simple contractions/hyphens).
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")

# Heuristics to detect temporal/actionable sentences for boosting salience.
_DATEISH = re.compile(
    r"\b("
    r"today|tomorrow|yesterday|tonight|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"next\s+(week|month|quarter|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(week|month|quarter|year)|"
    r"by\s+(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|end\s+of\s+next\s+week)|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b",
    re.IGNORECASE,
)
_ACTIONISH = re.compile(
    r"\b("
    r"will|must|need(?:s)?\s+to|should|plan(?:s)?\s+to|hope(?:s)?\s+to|intend(?:s)?\s+to|"
    r"review|complete|deliver|ship|migrate|optimi[sz]e|present|send|draft|prepare|rehearse|finali[sz]e"
    r")\b",
    re.IGNORECASE,
)

# Domain-ish keywords to nudge scoring toward action/ops content and away from small talk.
BOOST_WORDS = {
    "priority","deadline","deliverable","due","risk","downtime","blocker","migration",
    "database","indexes","optimize","optimise","presentation","slides","contract","review",
    "invoice","payment","servers","maintenance","bug","login","testing","code review",
}
PENALTY_WORDS = {
    "coffee","breakroom","pizza","snacks","jokes","small talk","chit","chat","irrelevant",
    "side chat","off-topic","holiday plans","weather","nothing concrete","printer",
}

# Used to avoid injecting a period right after a reporting phrase (prevents "Alice said. we will…").
# NOTE: `$` anchors the pattern at end of a small sliding window (see call site).
REPORTING_VERBS = re.compile(
    r"(says?|said|states?|notes?|adds?|reports?|mentions?|tells?|told)\s+(?:that\s+)?$",
    re.IGNORECASE
)

def clean_text(text: str) -> str:
    """Normalize whitespace and spacing around punctuation/quotes."""
    s = text.strip()
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)     # remove space before punctuation
    s = re.sub(r"\s+([’”\"'\)])", r"\1", s)    # remove space before right quotes/parens
    s = re.sub(r"([“\"(])\s+", r"\1", s)       # remove space after left quotes/parens
    s = re.sub(r"[ \t]{2,}", " ", s)           # collapse runs of spaces/tabs
    return s


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter:
      1) split on punctuation + whitespace or newlines
      2) if that fails (e.g., ASR output with no caps), split on lower→Upper transitions
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    if len(parts) <= 1:
        # Fallback for ASR-like streams: "we did X And Then Y"
        parts = re.split(r"(?<=[a-z])\s+(?=[A-Z])", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------- Run-on fixer (don’t split compound subjects or after reporting verbs) ----------
def _insert_periods_between_true_clauses(text: str) -> str:
    """
    Inserts periods between multiple future/intent clauses that share a subject,
    while trying not to split compound subjects or reported speech.
    """
    s = text.strip()
    if not s:
        return s

    # Cue: subject-like tokens followed by future/intent verbs.
    subjcue = re.compile(
        r"\b("
        r"[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)*"   # Proper names, possibly "Alice and Bob"
        r"|team|sales|marketing|finance|ops|qa|dev|engineering|support|we|they|he|she|i"
        r")\s+(will|need(?:s)?\s+to|must|plan(?:s)?\s+to|hope(?:s)?\s+to|expect(?:s)?\s+to|intend(?:s)?\s+to|aim(?:s)?\s+to)",
        re.IGNORECASE,
    )

    matches = list(subjcue.finditer(s))
    if not matches:
        return s

    insert_positions: List[int] = []
    for i, m in enumerate(matches):
        # Skip first subject; we only consider breaks before subsequent subject-intent cues.
        if i == 0:
            continue
        start = m.start()

        # If there's already terminal punctuation just before, no need to insert.
        if start > 0 and s[start - 1] in ".!?":
            continue

        # Avoid splitting immediately after reporting verbs like "Alice said (that) …"
        window = s[max(0, start - 40):start]
        if REPORTING_VERBS.search(window):
            continue

        # Avoid breaking simple compound subjects: "... and Bob will ..."
        if s[max(0, start - 5):start].lower().endswith(" and "):
            continue

        insert_positions.append(start)

    # Insert ". " at computed positions (single pass using index set).
    out, inset = [], set(insert_positions)
    for i, ch in enumerate(s):
        if i in inset:
            out.append(". ")
        out.append(ch)
    result = "".join(out).strip()

    # Capitalize following sentences after our inserted periods.
    parts = re.split(r"(?<=[.!?])\s+", result)
    parts = [p[:1].upper() + p[1:] if p else p for p in parts]
    return " ".join(parts)


# ---------- Micro rewrite for short inputs (meaning-preserving) ----------
def _micro_rewrite_short(text: str) -> str:
    """
    Small, safe, local rewrites to clean style without changing meaning:
      - collapse 'X says he/she/they' → 'X'
      - drop 'says/said that'
      - tidy 'work together to complete' phrasing
      - remove weak openers like 'During the discussion,'
    """
    s = text
    s = re.sub(r"\b([A-Z][a-z]+)\s+says?\s+(he|she|they)\s+", r"\1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([A-Z][a-z]+)\s+said\s+(he|she|they)\s+", r"\1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(says?|said)\s+that\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwill\s+work\s+together\s+to\s+complete\b", "will complete", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwork\s+together\s+to\s+complete\b", "complete", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*(during the discussion|in general|overall|generally),?\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


# ---------------------------- Extractive baseline ------------------------- #
def extractive_summary(text: str, target_sentences: int = 3) -> str:
    """
    Simple frequency-based extractive summarizer:
      - sentence tokenization
      - term frequency scoring (stopwords removed, normalized by max freq)
      - select top-k sentences, preserving original order
    """
    sentences = split_sentences(text)
    if len(sentences) <= target_sentences:
        return " ".join(sentences)

    # Word frequency accumulation (lowercased, stopwords removed)
    freqs: Dict[str, int] = {}
    for s in sentences:
        for w in _WORD.findall(s.lower()):
            if w in _STOPWORDS:
                continue
            freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        # Degenerate case: no tokens → return leading sentences
        return " ".join(sentences[:target_sentences])

    maxf = max(freqs.values())
    norm = {w: f / maxf for w, f in freqs.items()}

    # Score each sentence by sum of normalized token frequencies
    scored: List[tuple[int, float]] = []
    for i, s in enumerate(sentences):
        score = sum(norm.get(w, 0.0) for w in _WORD.findall(s.lower()) if w not in _STOPWORDS)
        scored.append((i, score))

    # Take indices of top-k by score and re-order by input order
    top_idx = set(i for i, _ in sorted(scored, key=lambda t: t[1], reverse=True)[:target_sentences])
    ordered = [sentences[i] for i in range(len(sentences)) if i in top_idx]
    return " ".join(ordered)


def _chunk_text(text: str, max_chunk_chars: int = 1500) -> List[str]:
    """
    Break long text into sentence-aligned chunks (≤ max_chunk_chars) for
    repeated abstractive summarization. Preserves sentence boundaries.
    """
    text = text.strip()
    if len(text) <= max_chunk_chars:
        return [text]
    sentences = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s_len = len(s) + 1
        if cur_len + s_len > max_chunk_chars and cur:
            chunks.append(" ".join(cur)); cur, cur_len = [s], s_len
        else:
            cur.append(s); cur_len += s_len
    if cur: chunks.append(" ".join(cur))
    return chunks


# --------------------------- Concise summarizer --------------------------- #
def _proper_noun_count(s: str) -> int:
    """Count Proper-like tokens (naive TitleCase heuristic) to boost named-entity sentences."""
    return len(re.findall(r"\b[A-Z][a-z]+\b", s))


def _length_penalty(word_count: int) -> float:
    """
    Soft preference curve for sentence lengths:
      - penalize very short/very long;
      - prefer ~8–32 words.
    """
    if word_count <= 5: return 0.6
    if word_count <= 8: return 0.9
    if word_count <= 32: return 1.0
    if word_count <= 48: return 0.9
    return 0.8


def _polish_selected_sentence(s: str) -> str:
    """Apply safe micro-edits to a selected original sentence."""
    s = _insert_periods_between_true_clauses(s)
    s = _micro_rewrite_short(s)
    return clean_text(s)


def concise_summary(text: str, max_sentences: int = 6) -> str:
    """
    Salience-driven selection of up to `max_sentences` original sentences, then light polishing.
    Strategy:
      - frequency score (with stopwords)
      - boosts for dates, actiony verbs, proper nouns, domain words
      - penalties for chit-chat cues
      - length preference
      - keep original order for readability
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    # Word frequency model
    freqs: Dict[str, int] = {}
    for s in sentences:
        for w in _WORD.findall(s.lower()):
            if w in _STOPWORDS: continue
            freqs[w] = freqs.get(w, 0) + 1
    maxf = max(freqs.values()) if freqs else 1
    norm = {w: f / maxf for w, f in freqs.items()}

    scored: List[tuple[int, float]] = []
    for i, s in enumerate(sentences):
        words = _WORD.findall(s.lower())
        base = sum(norm.get(w, 0.0) for w in words if w not in _STOPWORDS)

        # Heuristics to bias toward actionable/status info
        boost = 0.0
        if _DATEISH.search(s): boost += 0.8
        if _ACTIONISH.search(s): boost += 0.6
        pn = _proper_noun_count(s)
        boost += min(0.45, pn * 0.08)  # cap proper-noun boost
        if any(w in s.lower() for w in BOOST_WORDS): boost += 0.6
        if any(w in s.lower() for w in PENALTY_WORDS): boost -= 1.0
        if re.search(r"\b(nothing concrete|not a direct action item|irrelevant|side chats?)\b", s, re.IGNORECASE):
            boost -= 0.8

        wc = max(1, len(s.split()))
        base *= _length_penalty(wc)

        scored.append((i, base + boost))

    # Choose top-N by score, but output in original order
    top = sorted(scored, key=lambda t: t[1], reverse=True)[:max_sentences]
    chosen_idx = sorted(i for i, _ in top)
    chosen = [sentences[i] for i in chosen_idx]

    # Micro polish only (no new facts)
    polished = [_polish_selected_sentence(c) for c in chosen if c.strip()]

    out = " ".join(polished)
    return clean_text(out)


# --------------------- Slide-aware OCR summarization ---------------------- #
def _is_title_line(line: str) -> bool:
    """
    Heuristic: short line, no terminal punctuation, mostly TitleCase tokens,
    1–8 words → likely a slide title.
    """
    if not line: return False
    if len(line) > 60: return False
    if line.strip().endswith(('.', '!', '?')): return False
    words = line.strip().split()
    if not (1 <= len(words) <= 8): return False
    tc = sum(1 for w in words if re.match(r"^[A-Z][a-zA-Z0-9\-]*$", w))
    return tc >= max(1, int(0.6 * len(words)))


def _looks_like_slide(text: str) -> bool:
    """
    Decide if OCR text is slide-like:
      - short average line length and overall short text, or
      - sentence count close to number of lines (bullets)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2: return False
    sentences = split_sentences(text)
    avg_line_len = sum(len(l) for l in lines) / max(1, len(lines))
    return (avg_line_len < 65 and len(text) <= 500) or (len(sentences) <= len(lines) + 1)


def _collapse_phrases(phrases: List[str]) -> List[str]:
    """
    Normalize bullet-like fragments:
      - drop leading articles
      - small phrasing cleanups
      - strip punctuation/markers
      - de-duplicate while preserving order
    """
    out: List[str] = []
    for p in phrases:
        s = p.strip()
        s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\bcan\s+depend\s+on\b", "depend on", s, flags=re.IGNORECASE)
        s = re.sub(r"\bto date\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s{2,}", " ", s).strip(" -•–—:;,.")
        out.append(s)
    # De-dup
    seen = set(); uniq = []
    for s in out:
        low = s.lower()
        if low in seen: continue
        uniq.append(s); seen.add(low)
    return uniq


def ocr_slide_summary(text: str, prefer_two_sentences: bool = True) -> str:
    """
    Convert slide OCR (title + bullets) to 1–2 compact sentences.
    If a title is detected, include it as a lead-in.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines: return ""
    title = ""
    if _is_title_line(lines[0]):
        title = lines[0]; lines = lines[1:]
    if not lines: return title
    phrases = _collapse_phrases(lines)
    if prefer_two_sentences and len(phrases) >= 2:
        first = phrases[0]; rest = phrases[1:]
        joiner = "; ".join(rest)
        summary = f"{title}: {first}; {joiner}." if title else f"{first}; {joiner}."
    else:
        body = "; ".join(phrases)
        summary = f"{title}: {body}." if title else f"{body}."
    summary = re.sub(r"\s+([.,;:])", r"\1", summary)
    summary = re.sub(r"[ \t]{2,}", " ", summary).strip()
    if summary: summary = summary[0].upper() + summary[1:]
    return summary


# ----------------------------- AUDIO helpers ------------------------------ #
# Regex for common fillers to strip from ASR transcripts.
_FILLERS_RE = re.compile(
    r"\b(uh|um|er|ah|like|you know|i mean|sort of|kind of|basically|literally|right|okay|ok|alright)\b",
    re.IGNORECASE,
)

def _remove_fillers(text: str) -> str:
    """Remove speech fillers and collapse extra spaces."""
    s = _FILLERS_RE.sub("", text)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _add_soft_sentence_breaks(text: str, approx_every_words: int = 18) -> str:
    """
    For long run-on ASR outputs, insert soft periods ~every N words,
    preferably at conjunctions; helps downstream sentence selection.
    """
    tokens = text.strip().split()
    if len(tokens) <= approx_every_words:
        return text.strip()
    out, count = [], 0
    for i, tok in enumerate(tokens, 1):
        out.append(tok)
        count += 1
        if count >= approx_every_words and i < len(tokens):
            # Prefer breaking right after a conjunction cue if present.
            if re.fullmatch(r"(and|but|so|because|then|however)", tokens[i-1].lower()):
                out.append(".")
                count = 0
            elif count >= approx_every_words + 4:
                out.append("."); count = 0
    s = " ".join(out)
    s = re.sub(r"\s+\.", ".", s)
    return s.strip()


# ------------------------------ OCR I/O ---------------------------------- #
def extract_text_from_image(image_path: str) -> str:
    """
    Basic OCR wrapper using Pillow + pytesseract.
    Honors optional env var TESSERACT_CMD to point to the tesseract binary (e.g., Windows).
    """
    if not (PIL_AVAILABLE and TESSERACT_AVAILABLE):
        raise RuntimeError(
            "OCR requires Pillow and pytesseract. Install with `pip install pillow pytesseract` and "
            "ensure Tesseract is installed (e.g., Windows installer or `brew install tesseract`)."
        )
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        # Override pytesseract's tesseract binary path when provided.
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd  # type: ignore
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)  # type: ignore


# ------------------------ Summarization controller ------------------------ #
class Summarizer:
    """
    Unified summarizer facade:
      - style='faithful': for short inputs, fix run-ons + micro rewrite;
                         for long inputs, try abstractive (Transformers) with safe defaults,
                         else deterministic extractive fallback (max_sentences respected).
      - style='concise' : salience selection of up to N sentences + tiny polishing (no new facts).
    """
    def __init__(self, model_name: Optional[str] = None, use_abstractive: bool = True) -> None:
        # Default model can be overridden via env var
        if model_name is None:
            model_name = os.environ.get("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
        self._abstractive = None
        # Boundary between "short" and "long" inputs for faithful mode (env-tunable)
        self._small_limit = int(os.environ.get("FAITHFUL_SMALL_INPUT_CHARS", "240"))

        # Conservative generation knobs to reduce repetition/rambling.
        self._num_beams = int(os.environ.get("SUMMARIZER_BEAMS", "4"))
        self._repetition_penalty = float(os.environ.get("SUMMARIZER_REP_PENALTY", "1.8"))
        self._length_penalty = float(os.environ.get("SUMMARIZER_LEN_PENALTY", "1.0"))
        self._no_repeat_ngram = int(os.environ.get("SUMMARIZER_NO_REPEAT_NGRAM", "4"))

        # Initialize transformers pipeline if requested and available.
        if TRANSFORMERS_AVAILABLE and use_abstractive:
            try:
                self._abstractive = pipeline("summarization", model=model_name)
            except Exception:
                # If model can't load, silently fall back to extractive later.
                self._abstractive = None

    def _summarize_abstractive(self, text: str, max_length: int, min_length: int) -> Optional[str]:
        """Helper that wraps the transformers pipeline and catches runtime errors."""
        if self._abstractive is None:
            return None
        try:
            out = self._abstractive(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=self._num_beams,
                repetition_penalty=self._repetition_penalty,
                length_penalty=self._length_penalty,
                no_repeat_ngram_size=self._no_repeat_ngram,
            )
            return out[0]["summary_text"]
        except Exception:
            return None

    def summarize(
        self,
        text: str,
        *,
        style: str = "faithful",
        max_length: int = 160,
        min_length: int = 40,
        max_sentences: int = 6,   # used by 'concise' and by 'faithful' extractive fallback
    ) -> str:
        """
        Main summarization method for text:
          - cleans input
          - selects style and path (short faithful vs long abstractive vs extractive)
          - returns cleaned summary
        """
        text = clean_text(text)
        if not text:
            return ""

        if style.lower() == "concise":
            return concise_summary(text, max_sentences=max_sentences)

        # Faithful style:
        sentences = split_sentences(text)
        if len(text) <= self._small_limit or len(sentences) <= 2:
            # Very short → do a light, meaning-preserving cleanup.
            s = _insert_periods_between_true_clauses(text)
            s = _micro_rewrite_short(s)
            return clean_text(s)

        # Long input: try abstractive if available, else fall back.
        if self._abstractive is not None:
            dynamic_min = max(8, min(30, int(0.10 * max_length)))  # keep some variability
            chunks = _chunk_text(text, max_chunk_chars=1500)
            partial: List[str] = []
            for ck in chunks:
                s = self._summarize_abstractive(ck[:12000], max_length=max_length, min_length=dynamic_min)
                if s:
                    partial.append(s)
            if partial:
                if len(partial) == 1:
                    return clean_text(partial[0])
                # Summarize concatenated partials once more for cohesion.
                joined = " ".join(clean_text(p) for p in partial)
                final = self._summarize_abstractive(
                    joined[:12000],
                    max_length=max_length,
                    min_length=max(8, dynamic_min // 2)
                )
                if final:
                    return clean_text(final)

        # Deterministic extractive fallback (respects caller's `max_sentences`).
        return clean_text(extractive_summary(text, target_sentences=max_sentences))


# ------------------------------- Facade ----------------------------------- #
def summarize_text(
    text: str,
    *,
    mode: str = "auto",
    style: str = "faithful",
    max_len: int = 160,
    max_sentences: int = 6,
) -> str:
    """
    Convenience function to summarize plain text.
      - mode='auto' uses abstractive if available; 'extractive' forces fallback.
      - style chooses 'faithful' vs 'concise' behavior.
    """
    use_abstractive = (mode.lower() != "extractive")
    return Summarizer(use_abstractive=use_abstractive).summarize(
        text, style=style, max_length=max_len, max_sentences=max_sentences
    )


def summarize_ocr(
    image_path: str,
    *,
    mode: str = "auto",
    style: str = "faithful",
    max_len: int = 160,
    max_sentences: int = 6,
    slide_aware: bool = True,
) -> Tuple[str, str]:
    """
    OCR an image (via Tesseract) then summarize the extracted text.
    If slide-like content is detected and slide_aware=True, compress title + bullets.
    Returns (raw_text, summary).
    """
    text = extract_text_from_image(image_path)
    if slide_aware and _looks_like_slide(text):
        summary = ocr_slide_summary(text, prefer_two_sentences=True)
    else:
        summary = summarize_text(text, mode=mode, style=style, max_len=max_len, max_sentences=max_sentences)
    return text, summary


def summarize_audio(
    audio_path: str,
    *,
    use_offline: bool = False,
    mode: str = "auto",
    style: str = "faithful",
    max_len: int = 160,
    max_sentences: int = 6,
    force_concise_if_short: bool = True,
    short_threshold_chars: int = 220,
    audio_sentences_cap: Optional[int] = None,  # if set, overrides max_sentences for audio
) -> Tuple[str, str]:
    """
    Transcribe audio then summarize. Short transcripts can force concise compaction.
    Returns (transcript, summary).
    """
    transcript = transcribe_audio(audio_path, use_google=not use_offline)

    # Clean up ASR artifacts and add soft sentence boundaries to aid selection.
    cleaned = _remove_fillers(transcript)
    cleaned = _add_soft_sentence_breaks(cleaned, approx_every_words=18)

    # Choose style for audio: optionally force concise for very short content.
    use_style = style
    sentences = split_sentences(cleaned)
    eff_cap = audio_sentences_cap if audio_sentences_cap is not None else max_sentences
    if force_concise_if_short and (len(cleaned) <= short_threshold_chars or len(sentences) <= 2):
        use_style = "concise"
        eff_cap = max(2, min(eff_cap, 3))  # keep mini-summaries tight

    summary = summarize_text(
        cleaned, mode=mode, style=use_style, max_len=max_len, max_sentences=eff_cap
    )
    return transcript, summary


# ------------------------------ Audio I/O -------------------------------- #
def transcribe_audio(audio_path: Optional[str] = None, use_google: bool = True) -> str:
    """
    Transcribe audio using SpeechRecognition:
      - If audio_path provided: transcribe from file
      - Else: capture from microphone
      - use_google=True uses Google Web Speech API (online), else PocketSphinx (offline)
    """
    if not SR_AVAILABLE:
        raise RuntimeError("speech_recognition is not installed. `pip install SpeechRecognition`.\n"
                           "On macOS: `brew install portaudio` then `pip install pyaudio`.")
    recognizer = sr.Recognizer()  # type: ignore

    if audio_path:
        with sr.AudioFile(audio_path) as source:  # type: ignore
            audio = recognizer.record(source)
    else:
        with sr.Microphone() as source:  # type: ignore
            print("Speak now… (Ctrl+C to stop)")
            audio = recognizer.listen(source)
    try:
        if use_google:
            return recognizer.recognize_google(audio)  # type: ignore
        else:
            return recognizer.recognize_sphinx(audio)  # type: ignore
    except sr.UnknownValueError:  # type: ignore
        # No transcript decoded; return empty string to keep pipeline robust.
        return ""
    except sr.RequestError as e:  # type: ignore
        # e.g., network/API errors
        raise RuntimeError(f"Speech API error: {e}")
