"""
AI-Powered YouTube Video Chatbot (RAG)
--------------------------------------
Streamlit UI with:
  1. YouTube URL input + automatic Video ID extraction
  2. Automatic Hindi / English language detection & responses
  3. A clean, premium pastel UI (lavender / blue / pink / white)

The underlying RAG pipeline (LangChain + Ollama + FAISS) is kept intact.
"""

import re

import requests
import streamlit as st

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Optional language detector (fallback only; primary detection uses caption metadata)
try:
    from langdetect import detect as _lang_detect
except Exception:  # pragma: no cover - langdetect is optional
    _lang_detect = None


# ==================================================================
# PAGE CONFIGURATION
# ==================================================================

st.set_page_config(
    page_title="AI Video Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================
# PASTEL PREMIUM THEME  (lavender / blue / pink / white)
# ==================================================================

st.markdown(
    """
    <style>
    /* ---------- Fonts & base ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- App background: soft pastel gradient ---------- */
    .stApp {
        background: linear-gradient(135deg, #f5f3ff 0%, #eef2ff 35%, #fdf2f8 100%);
        background-attachment: fixed;
        color: #3f3d56;
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent;}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(196, 181, 253, 0.35);
    }
    section[data-testid="stSidebar"] * {
        color: #4b4b6b;
    }

    /* ---------- Hero header ---------- */
    .hero {
        text-align: center;
        padding: 28px 20px 8px 20px;
    }
    .hero-title {
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(90deg, #a78bfa 0%, #818cf8 45%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 17px;
        color: #7c7a99;
        font-weight: 500;
    }

    /* ---------- Cards ---------- */
    .soft-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(196, 181, 253, 0.35);
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 10px 30px rgba(129, 140, 248, 0.12);
        margin-bottom: 18px;
    }
    .video-title {
        font-size: 20px;
        font-weight: 700;
        color: #4c4a6a;
        margin-bottom: 4px;
    }
    .video-meta {
        font-size: 14px;
        color: #8a88a6;
    }

    /* ---------- Language / status pills ---------- */
    .pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 8px;
    }
    .pill-lang {
        background: linear-gradient(90deg, #ede9fe, #dbeafe);
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }
    .pill-chunks {
        background: linear-gradient(90deg, #fce7f3, #ede9fe);
        color: #be185d;
        border: 1px solid #fbcfe8;
    }

    /* ---------- Inputs ---------- */
    .stTextInput > div > div > input {
        border-radius: 14px !important;
        border: 1.5px solid #ddd6fe !important;
        padding: 12px 14px !important;
        background: #ffffff !important;
        color: #3f3d56 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 14px;
        border: none;
        padding: 12px 18px;
        font-weight: 700;
        font-size: 15px;
        color: white;
        background: linear-gradient(90deg, #a78bfa 0%, #818cf8 50%, #f472b6 100%);
        box-shadow: 0 8px 20px rgba(129, 140, 248, 0.35);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 26px rgba(129, 140, 248, 0.45);
        color: white;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ---------- Chat bubbles ---------- */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(196, 181, 253, 0.3);
        border-radius: 18px;
        padding: 6px 14px;
        box-shadow: 0 6px 18px rgba(129, 140, 248, 0.08);
        margin-bottom: 6px;
    }

    /* ---------- Chat input ---------- */
    div[data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
        border: 1.5px solid #ddd6fe !important;
    }

    /* ---------- Alerts (rounded) ---------- */
    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    /* ---------- Steps list ---------- */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #5b5977;
    }
    .step-num {
        min-width: 24px;
        height: 24px;
        border-radius: 999px;
        background: linear-gradient(90deg, #ede9fe, #fce7f3);
        color: #7c3aed;
        font-weight: 700;
        font-size: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# HERO HEADER
# ==================================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎬 AI Video Chatbot</div>
        <div class="hero-subtitle">
            
            &nbsp;·&nbsp; Auto Hindi / English
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# SESSION STATE
# ==================================================================

for key, default in {
    "retriever": None,
    "video_id": None,
    "video_title": None,
    "video_author": None,
    "video_thumbnail": None,
    "transcript_lang": "English",
    "num_chunks": 0,
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==================================================================
# MODELS (cached)
# ==================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="mistral:7b")
    return embeddings, llm


embeddings, llm = load_models()


# ==================================================================
# LANGUAGE-AWARE PROMPT
# ==================================================================

prompt = PromptTemplate(
    template="""
You are an AI assistant that answers questions about a YouTube video, using its
transcript as the only source of truth.

Rules:
- Answer ONLY using the information available in the transcript context below.
- If the answer is not present in the context, say you don't know based on the
  provided transcript (say this in {language}).
- Do NOT use outside knowledge.
- Keep the answer clear, accurate and concise.
- Respond in {language}. Keep the entire answer in {language} unless the user
  explicitly asks for a different language, in which case reply in that language.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question", "language"],
)


# ==================================================================
# HELPERS
# ==================================================================

# 11-char YouTube video id
_VIDEO_ID_RE = r"[0-9A-Za-z_-]{11}"


def extract_video_id(url: str):
    """Extract a YouTube video ID from a variety of URL formats.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/live/VIDEO_ID
      - a raw 11-character video ID

    Returns the video ID string, or ``None`` if nothing valid is found.
    """
    if not url:
        return None

    url = url.strip()

    # A bare video ID was pasted.
    if re.fullmatch(_VIDEO_ID_RE, url):
        return url

    patterns = [
        rf"(?:v=)({_VIDEO_ID_RE})",
        rf"(?:youtu\.be/)({_VIDEO_ID_RE})",
        rf"(?:embed/)({_VIDEO_ID_RE})",
        rf"(?:shorts/)({_VIDEO_ID_RE})",
        rf"(?:live/)({_VIDEO_ID_RE})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_video_info(video_id: str) -> dict:
    """Fetch lightweight video metadata (title/author/thumbnail) via YouTube oEmbed.

    Returns a dict with keys title, author, thumbnail. Values may be ``None`` if
    the request fails (metadata is best-effort and never blocks the pipeline).
    """
    info = {"title": None, "author": None, "thumbnail": None}
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            info["title"] = data.get("title")
            info["author"] = data.get("author_name")
            info["thumbnail"] = data.get("thumbnail_url")
    except Exception:
        pass
    return info


def detect_language(text: str, language_code: str) -> str:
    """Detect whether responses should be in Hindi or English.

    Primary signal is the caption's own language code. Falls back to a
    Devanagari character check and finally to ``langdetect`` if available.
    """
    code = (language_code or "").lower()

    if code.startswith("hi"):
        return "Hindi"
    if code.startswith("en"):
        return "English"

    # Fallback 1: Devanagari script strongly implies Hindi.
    if text and re.search(r"[\u0900-\u097F]", text):
        return "Hindi"

    # Fallback 2: statistical detection.
    if _lang_detect and text:
        try:
            return "Hindi" if _lang_detect(text) == "hi" else "English"
        except Exception:
            pass

    return "English"


def fetch_transcript(video_id: str):
    """Fetch the best available transcript for a video.

    Prefers a native English or Hindi caption track (manually created first),
    otherwise falls back to whatever caption track is available.

    Returns a tuple ``(transcript_text, language_label)``.
    Raises the underlying youtube_transcript_api exceptions on failure so the
    caller can present a precise error message.
    """
    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)
    transcripts = list(transcript_list)

    if not transcripts:
        raise NoTranscriptFound(video_id, ["en", "hi"], transcript_list)

    # __iter__ yields manually-created transcripts before generated ones, so the
    # first en/hi match is the highest quality track in a language we support.
    preferred = [
        t for t in transcripts
        if t.language_code.split("-")[0] in ("en", "hi")
    ]
    chosen = preferred[0] if preferred else transcripts[0]

    fetched = chosen.fetch()
    transcript_text = " ".join(snippet.text for snippet in fetched).strip()

    language_label = detect_language(
        transcript_text,
        getattr(fetched, "language_code", chosen.language_code),
    )

    return transcript_text, language_label


def build_retriever(transcript_text: str):
    """Split the transcript, embed it, and return a FAISS retriever + chunk count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents([transcript_text])

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    return retriever, len(chunks)


def format_docs(retrieved_docs) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def process_video(video_input: str):
    """Full pipeline: extract ID -> metadata -> transcript -> detect language -> RAG.

    Uses a staged status indicator so the user can watch progress. Updates
    ``st.session_state`` on success and surfaces friendly errors on failure.
    """
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error(
            "That doesn't look like a valid YouTube URL. Try a link like "
            "`https://www.youtube.com/watch?v=VIDEO_ID` or "
            "`https://youtu.be/VIDEO_ID`."
        )
        return

    with st.status("Processing video...", expanded=True) as status:
        # --- Step 1: video info ---
        st.write("🔎 Fetching video information...")
        info = get_video_info(video_id)

        # --- Step 2: transcript ---
        st.write("📝 Fetching transcript...")
        try:
            transcript_text, language_label = fetch_transcript(video_id)
        except TranscriptsDisabled:
            status.update(label="Failed", state="error")
            st.error("Captions are disabled for this video, so I can't read it.")
            return
        except NoTranscriptFound:
            status.update(label="Failed", state="error")
            st.error("No transcript is available for this video.")
            return
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Couldn't fetch the transcript: {exc}")
            return

        if not transcript_text:
            status.update(label="Failed", state="error")
            st.error("The transcript came back empty. Try another video.")
            return

        # --- Step 3: language ---
        st.write(f"🌐 Detected transcript language: **{language_label}**")

        # --- Step 4: embeddings + vector store ---
        st.write("🧠 Creating embeddings and building the vector store...")
        retriever, num_chunks = build_retriever(transcript_text)
        # st.write(f"📦 Transcript split into **{num_chunks}** chunks.")

        # --- Persist state ---
        st.session_state.retriever = retriever
        st.session_state.video_id = video_id
        st.session_state.video_title = info["title"]
        st.session_state.video_author = info["author"]
        st.session_state.video_thumbnail = info["thumbnail"]
        st.session_state.transcript_lang = language_label
        st.session_state.num_chunks = num_chunks
        st.session_state.messages = []

        status.update(label="Video ready to chat!", state="complete", expanded=False)

    st.success("Video processed successfully. Ask anything about it below! 🎉")


# ==================================================================
# SIDEBAR  ·  load video + how it works
# ==================================================================

with st.sidebar:
    st.markdown("### 🎥 Load a YouTube Video")

    video_input = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    load_clicked = st.button("🚀 Process Video", use_container_width=True)

    st.divider()

    st.markdown("#### How it works")
    steps = [
        "Paste a YouTube URL",
        "Video ID is auto-extracted",
        "Transcript is fetched",
        "Language auto-detected (HI / EN)",
        "Embeddings stored in FAISS",
        "Ask questions in chat",
        "Answers stay in the video's language",
    ]
    steps_html = "".join(
        f'<div class="step-item"><div class="step-num">{i}</div>'
        f"<div>{text}</div></div>"
        for i, text in enumerate(steps, start=1)
    )
    st.markdown(steps_html, unsafe_allow_html=True)

    st.divider()
    st.caption("Models · mistral:7b (LLM) · nomic-embed-text (embeddings)")


# ==================================================================
# HANDLE "PROCESS VIDEO"
# ==================================================================

if load_clicked:
    if not video_input:
        st.warning("Please enter a YouTube URL or video ID first.")
    else:
        process_video(video_input)


# ==================================================================
# CURRENT VIDEO CARD
# ==================================================================

if st.session_state.video_id:
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)

        title = st.session_state.video_title or "YouTube Video"
        author = st.session_state.video_author

        st.markdown(
            f'<div class="video-title">📺 {title}</div>',
            unsafe_allow_html=True,
        )
        if author:
            st.markdown(
                f'<div class="video-meta">by {author}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<span class="pill pill-lang">🌐 {st.session_state.transcript_lang}</span>'
            f'<span class="pill pill-chunks">📦 {st.session_state.num_chunks} chunks</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="video-meta" style="margin-top:8px;">Video ID: '
            f"{st.session_state.video_id}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.video(
            f"https://www.youtube.com/watch?v={st.session_state.video_id}"
        )
else:
    st.markdown(
        '<div class="soft-card" style="text-align:center;">'
        "👋 Paste a YouTube link in the sidebar and hit "
        "<b>Process Video</b> to start chatting."
        "</div>",
        unsafe_allow_html=True,
    )


# ==================================================================
# CHAT HISTORY
# ==================================================================

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ==================================================================
# CHAT INPUT
# ==================================================================

question = st.chat_input("Ask something about the video...")

if question:
    if st.session_state.retriever is None:
        st.warning("Please process a YouTube video first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching the transcript and thinking..."):
                retrieved_docs = st.session_state.retriever.invoke(question)
                context_text = format_docs(retrieved_docs)

                final_prompt = prompt.invoke(
                    {
                        "context": context_text,
                        "question": question,
                        "language": st.session_state.transcript_lang,
                    }
                )

                answer = llm.invoke(final_prompt)
                response = answer.content

            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
