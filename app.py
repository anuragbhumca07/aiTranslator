"""
app.py
Streamlit app — Gemini translation + gTTS TTS + file upload support (txt, pdf, csv, xlsx),
audio upload support (wav/mp3) and an in-browser recorder helper (download+upload workflow).
"""

import os
import io
import base64
from typing import Tuple

import streamlit as st
import pandas as pd
from gtts import gTTS
import PyPDF2
from dotenv import load_dotenv

# ---- Load .env (if present) ----
# load_dotenv()

# ---- Try import google generative SDK ----
try:
    import google.generativeai as genai  # pip install google-generativeai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---- Configuration ----
# Default Gemini model name (string). We'll create GenerativeModel instances when needed.
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Language mapping for gTTS (extendable)
LANGUAGE_OPTIONS = {
    "Hindi": "hi",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Bengali": "bn",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Korean": "ko",
}

# ---- Helpers ----

def configure_genai():
    api_key = st.secrets["gcp"]["api_key"]
    genai.configure(api_key=api_key)
    return genai

# def configure_genai_from_env():
#     """
#     Configure genai client if available and API key present.
#     Prefer st.secrets (Streamlit Cloud) or environment variables.
#     """
#     import google.generativeai as genai
#     api_key = None

#     # Try Streamlit secrets first
#     if "GEMINI_API_KEY" in st.secrets:
#         api_key = st.secrets["GEMINI_API_KEY"]
#     else:
#         # Fall back to environment variables
#         api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

#     if not api_key:
#         st.error("❌ No Gemini API key found. Please set it in .streamlit/secrets.toml or as an environment variable.")
#         return None

#     try:
#         genai.configure(api_key=api_key)
#         print("✅ GenAI client configured successfully")
#         return genai
#     except ImportError:
#         st.error("❌ google-generativeai SDK not installed. Run: pip install google-generativeai")
#         return None


def translate_with_gemini(genai_module, model_name: str, text: str, target_language_name: str) -> str:
    """
    Translate `text` into `target_language_name` using the given genai module and model_name.
    Returns translated text or raises RuntimeError on failure.
    """
    if genai_module is None:
        raise RuntimeError("GenAI client not configured. Ensure GEMINI_API_KEY env var is set and SDK installed.")

    # Create model instance
    try:
        model = genai_module.GenerativeModel(model_name)
    except Exception as e:
        # If GenerativeModel isn't present, raise a helpful error
        raise RuntimeError(f"Could not create GenerativeModel('{model_name}'): {e}") from e

    # Craft translation prompt (explicit)
    prompt_text = (
        f"Translate the following text into {target_language_name}.\n"
        f"Return only the translated text with no commentary or additional labels.\n\n"
        f"Text:\n\"\"\"\n{text}\n\"\"\"\n"
    )

    try:
        # The generate_content call varies between SDK versions; the pattern below works with python-genai.
        response = model.generate_content(prompt_text)
    except Exception as e:
        raise RuntimeError(f"Gemini translation call failed: {e}") from e

    # Response objects sometimes expose .text, sometimes .output etc.
    translated = None
    try:
        translated = getattr(response, "text", None)
    except Exception:
        translated = None

    if not translated:
        # Try common alternative structures
        try:
            # some SDKs return .output[0].content[0].text
            translated = response.output[0].content[0].text
        except Exception:
            try:
                translated = str(response)
            except Exception:
                translated = None

    if not translated:
        raise RuntimeError("No text returned by Gemini model.")

    return translated.strip()


def text_to_speech_bytes(text: str, lang_code: str = "en") -> bytes:
    """
    Convert `text` to MP3 bytes using gTTS.
    """
    tts = gTTS(text=text, lang=lang_code)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    return mp3_buf.read()


def extract_text_from_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """
    Extract text from uploaded files.
    Returns (text, detected_type).
    Supported types: .txt, .pdf (text PDFs), .csv, .xls/.xlsx
    """
    if uploaded_file is None:
        return "", ""

    filename = uploaded_file.name.lower()
    mime = (uploaded_file.type or "").lower()

    # TXT
    if filename.endswith(".txt") or mime.startswith("text"):
        raw = uploaded_file.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
        return text, "text"

    # PDF
    if filename.endswith(".pdf") or "pdf" in mime:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = []
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n".join(pages), "pdf"
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")

    # CSV
    if filename.endswith(".csv") or "csv" in mime:
        try:
            df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
            text = "\n".join(df.apply(lambda row: " | ".join(row.values.astype(str)), axis=1))
            return text, "csv"
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")

    # Excel
    if filename.endswith(".xls") or filename.endswith(".xlsx") or "excel" in mime:
        try:
            df = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
            text = "\n".join(df.apply(lambda row: " | ".join(row.values.astype(str)), axis=1))
            return text, "excel"
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file: {e}")

    # fallback: try decode
    try:
        content = uploaded_file.read()
        text = content.decode("utf-8", errors="ignore")
        if text.strip():
            return text, "unknown_text"
    except Exception:
        pass

    return "", "unsupported"


def make_download_link_bytes(content: bytes, filename: str, label: str = "Download") -> str:
    """Return an HTML link to download raw bytes."""
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href


# ---- Streamlit app ----

def main():
    st.set_page_config(page_title="AI Translator + TTS", layout="wide")

    st.title("🌍 AI Translator + Text-to-Speech (Gemini + gTTS)")
    st.markdown(
        "Translate text or files using Google Gemini and convert translations to MP3 with gTTS. "
        "**Default target language is Hindi**."
    )

    # Quick debug info (helpful when debugging env/api)
    with st.expander("Debug / Environment info (click to expand)"):
        python_exec = st.text(sys_executable := os.sys.executable if hasattr(os, "sys") else "unknown")
        st.write("Python executable:", os.sys.executable)
        st.write("GEMINI_API_KEY present:", bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))
        st.write("google-generativeai installed:", GENAI_AVAILABLE)

    # Configure genai client (if available)
    # client = configure_genai_from_env()
    client = configure_genai()
    if client is None and GENAI_AVAILABLE:
        # SDK present but key missing
        st.warning("Gemini SDK is installed but no API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment or a .env file.")
    elif not GENAI_AVAILABLE:
        st.warning("google-generativeai SDK not installed. Install it with: pip install google-generativeai")

    # UI layout
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Input")
        input_mode = st.radio("Input mode", ["Type / Paste", "Upload file", "Upload audio (or record)"], index=0)

        input_text = ""
        uploaded_file = None
        uploaded_audio = None

        if input_mode == "Type / Paste":
            input_text = st.text_area("Enter text to translate", height=240)
        elif input_mode == "Upload file":
            uploaded_file = st.file_uploader("Upload a file (txt, pdf, csv, xlsx)", type=["txt", "pdf", "csv", "xlsx"])
            if uploaded_file:
                try:
                    extracted, ftype = extract_text_from_uploaded_file(uploaded_file)
                    if extracted:
                        st.success(f"Detected file type: {ftype}. Extracted {len(extracted)} characters.")
                        st.text_area("Preview (first 2000 chars)", extracted[:2000], height=240)
                        input_text = extracted
                    else:
                        st.warning("No text extracted from file.")
                except Exception as e:
                    st.error(f"File read error: {e}")
        else:
            # audio mode: allow upload or use embedded recorder helper
            st.markdown("You can either upload an audio file (wav/mp3) or use the recorder helper below to create a short clip and then upload it for processing.")
            uploaded_audio = st.file_uploader("Upload audio (wav, mp3)", type=["wav", "mp3", "m4a"])
            if uploaded_audio:
                st.success(f"Uploaded audio file: {uploaded_audio.name}")
                st.audio(uploaded_audio.read())

            st.markdown("---")
            st.markdown("**In-browser recorder helper** (click record, stop, then download; upload back using 'Upload audio' above to use it in this app).")
            # Embedded JS/HTML recorder that lets the user record and download locally.
            recorder_html = """
            <style>
            .recorder { background: #f8f9fa; padding: 10px; border-radius:8px; }
            </style>
            <div class="recorder">
            <p><b>Recorder</b>: Allow microphone, then Press Record → Stop → Download. Then upload the file back to the app.</p>
            <button id="record">Record</button>
            <button id="stop" disabled>Stop</button>
            <a id="download" style="display:none">Download audio</a>
            <script>
            let mediaRecorder;
            let audioChunks = [];
            const recordBtn = document.getElementById("record");
            const stopBtn = document.getElementById("stop");
            const downloadLink = document.getElementById("download");
            recordBtn.onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                audioChunks = [];
                mediaRecorder.addEventListener("dataavailable", event => {
                    audioChunks.push(event.data);
                });
                mediaRecorder.addEventListener("stop", () => {
                    const blob = new Blob(audioChunks);
                    const url = URL.createObjectURL(blob);
                    downloadLink.href = url;
                    downloadLink.download = "recording.wav";
                    downloadLink.style.display = "inline-block";
                    downloadLink.textContent = "Download recorded audio (recording.wav)";
                });
                recordBtn.disabled = true;
                stopBtn.disabled = false;
            };
            stopBtn.onclick = () => {
                mediaRecorder.stop();
                recordBtn.disabled = false;
                stopBtn.disabled = true;
            };
            </script>
            </div>
            """
            st.components.v1.html(recorder_html, height=160)

        # Translation settings
        st.subheader("Translation settings")
        # default index for Hindi
        target_lang_display = st.selectbox("Target language", list(LANGUAGE_OPTIONS.keys()), index=0)
        tts_lang_code = LANGUAGE_OPTIONS[target_lang_display]

        st.markdown("Optional: choose Gemini model name (leave default for typical use).")
        model_choice = st.text_input("Gemini model (optional)", value=DEFAULT_GEMINI_MODEL)

        # actions
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1,1,1])
        with col_btn_1:
            translate_btn = st.button("Translate")
        with col_btn_2:
            tts_btn = st.button("Generate & Play Audio (MP3)")
        with col_btn_3:
            download_translated_btn = st.button("Download translation (.txt)")

    with col_r:
        st.subheader("Output / Logs")
        # Empty placeholders
        translated_text = None

        if translate_btn or tts_btn or download_translated_btn:
            # Collect the effective input_text
            # If audio uploaded and user wants transcription -> we do not provide auto-transcription here (outside scope)
            if input_mode == "Upload audio (or record)" and uploaded_audio:
                st.info("Audio upload detected. This app doesn't auto-transcribe audio by default. Please provide a text transcription in the text box or upload a text file. (You can use external tools or add Google Speech-to-Text integration later.)")
                # Stop here unless there's text to translate
                if not input_text:
                    st.error("No text available for translation. Provide text or upload a text file.")
                    return

            if not input_text or not input_text.strip():
                st.error("No input text found. Please enter text or upload a text file.")
            else:
                # Prepare genai client
                # client = configure_genai_from_env()
                client = configure_genai()
                if client is None:
                    st.error("Gemini client not configured. Ensure GEMINI_API_KEY or GOOGLE_API_KEY environment variable is set and the google-generativeai package is installed.")
                else:
                    model_to_use = model_choice.strip() or DEFAULT_GEMINI_MODEL
                    with st.spinner("Translating with Gemini..."):
                        try:
                            translated_text = translate_with_gemini(client, model_to_use, input_text, target_lang_display)
                            st.success("Translation complete.")
                        except Exception as e:
                            st.exception(f"Translation failed: {e}")
                            translated_text = None

        # Show translated text if present
        if translated_text:
            st.subheader(f"Translated text ({target_lang_display})")
            st.text_area("Translation", translated_text, height=250)

            if download_translated_btn:
                b = translated_text.encode("utf-8")
                st.download_button("Download translated .txt", b, file_name="translation.txt", mime="text/plain")

            if tts_btn:
                with st.spinner("Generating audio via gTTS..."):
                    try:
                        mp3_bytes = text_to_speech_bytes(translated_text, lang_code=tts_lang_code)
                        st.audio(mp3_bytes, format="audio/mp3")
                        # Provide a download link
                        st.markdown(make_download_link_bytes(mp3_bytes, "translation.mp3", "Download MP3"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Text-to-speech failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown("**Notes:**\n"
                "- Gemini API requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable (or set in a .env file).\n"
                "- The embedded recorder helper allows you to record and download a clip locally; upload it back if you want to use it with other tools. "
                "- Automatic speech-to-text (audio -> text) is not included by default; you can integrate Google Speech-to-Text later if needed.\n")

if __name__ == "__main__":
    main()
