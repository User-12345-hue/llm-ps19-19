
import os
import io
import json
import time
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS


# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chat_history.json"
)

st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL UI DESIGN
# =====================================================

st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background: #F3F7FD;
    color: #183153;
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* TOP HEADER */
header[data-testid="stHeader"] {
    background: #F3F7FD;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2EAF4;
}

section[data-testid="stSidebar"] > div {
    background: #FFFFFF;
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #15365B !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption {
    color: #64748B !important;
}

/* MAIN HEADINGS */
h1, h2, h3 {
    color: #15365B !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px;
}

p, label {
    color: #526780;
}

/* HERO BANNER */
.hero {
    background: linear-gradient(
        120deg,
        #102F56 0%,
        #1769AA 65%,
        #368ED0 100%
    );
    padding: 42px 38px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(23, 105, 170, 0.16);
    position: relative;
    overflow: hidden;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 38px;
    font-weight: 750 !important;
    margin: 12px 0;
}

.hero p {
    color: #E4F1FF !important;
    font-size: 16px;
    line-height: 1.7;
    max-width: 650px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 30px;
    padding: 7px 15px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
}

/* FEATURE CARDS */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E2EAF4;
    border-radius: 17px;
    padding: 24px;
    min-height: 145px;
    box-shadow: 0 5px 18px rgba(25, 55, 90, 0.05);
    transition: all 0.25s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(25, 55, 90, 0.10);
    border-color: #A8CDEF;
}

.feature-icon {
    font-size: 29px;
    margin-bottom: 12px;
}

.feature-card h3 {
    color: #1769AA !important;
    font-size: 17px;
    margin-bottom: 8px;
}

.feature-card p {
    color: #64748B !important;
    font-size: 13px;
    line-height: 1.6;
}

/* CHAT MESSAGES */
div[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E2EAF4;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 13px;
    box-shadow: 0 3px 12px rgba(20, 50, 80, 0.04);
}


/* BUTTONS */
.stButton > button,
.stFormSubmitButton > button {
    background: #1769AA !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #1769AA !important;
    border-radius: 10px !important;
    min-height: 44px;
    font-size: 14px;
    font-weight: 600;
}

.stButton > button p,
.stFormSubmitButton > button p,
.stButton > button span,
.stFormSubmitButton > button span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #10558D !important;
    border-color: #10558D !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}


/* INPUT FIELDS */
.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #183153 !important;
    border: 1px solid #C9D8E9 !important;
    border-radius: 10px !important;
    min-height: 45px;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #8495A9 !important;
    opacity: 1 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #368ED0 !important;
    box-shadow: 0 0 0 2px rgba(54, 142, 208, 0.13) !important;
}

/* INPUT LABELS */
.stTextInput label,
.stTextArea label {
    color: #294967 !important;
    font-weight: 600;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #183153 !important;
    border: 1px solid #D5E0ED !important;
    border-radius: 10px !important;
}

/* SIDEBAR INFORMATION CARDS */
.sidebar-info {
    background: #F2F7FD;
    border: 1px solid #DFEAF6;
    border-radius: 12px;
    padding: 13px 15px;
    margin: 8px 0 15px 0;
}

.sidebar-info-title {
    color: #1769AA;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 5px;
}

.sidebar-info-value {
    color: #183153;
    font-size: 14px;
    font-weight: 600;
}

/* API STATUS */
.status-card {
    background: #EAF7F0;
    border: 1px solid #C8EBD6;
    border-radius: 11px;
    padding: 12px 14px;
    color: #187344;
    font-size: 13px;
    font-weight: 600;
    margin: 12px 0;
}

/* AUDIO PLAYER */
audio {
    width: 100%;
    border-radius: 12px;
}

/* AUDIO RECORDER */
div[data-testid="stAudioInput"] {
    background: #FFFFFF;
    border: 1px solid #DCE7F3;
    border-radius: 13px;
    padding: 12px;
}

/* ALERTS */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* DIVIDERS */
hr {
    border-color: #E2EAF4 !important;
}

/* SPINNER */
div[data-testid="stSpinner"] {
    color: #1769AA;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #71849B;
    font-size: 13px;
    padding: 25px 10px;
    border-top: 1px solid #E2EAF4;
    margin-top: 30px;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .hero {
        padding: 28px 22px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. GEMINI API CONNECTION
# =====================================================

if not API_KEY:
    st.error(
        "Gemini API key not found. "
        "Please add GEMINI_API_KEY to your .env file."
    )
    st.stop()

try:
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=8.0,
                exp_base=2.0,
                jitter=1.0
            )
        )
    )

except Exception as error:
    st.error(f"Could not connect to Gemini: {error}")
    st.stop()


# =====================================================
# 4. MODEL DISCOVERY AND FALLBACK
# =====================================================

@st.cache_data(ttl=3600, show_spinner=False)
def discover_available_models():
    """Find available Gemini Flash models."""

    available = []

    try:
        model_list = client.models.list()

        for model_info in model_list:
            name = getattr(model_info, "name", None)

            if not name:
                continue

            name = name.replace("models/", "")

            actions = getattr(
                model_info,
                "supported_actions",
                None
            ) or []

            if actions and "generateContent" not in actions:
                continue

            if "flash" in name.lower():
                available.append(name)

    except Exception:
        pass

    available = list(dict.fromkeys(available))

    if PREFERRED_MODEL in available:
        available.remove(PREFERRED_MODEL)
        available.insert(0, PREFERRED_MODEL)

    return available


def get_model_candidates():
    """Return available models with fallback options."""

    available = discover_available_models()

    if available:
        return available

    return [
        PREFERRED_MODEL,
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]


def generate_with_fallback(contents, config):
    """Try Gemini models one by one."""

    models_to_try = get_model_candidates()
    last_error = None

    for model_name in models_to_try:

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            if response.text:
                return response.text.strip(), model_name

            last_error = ValueError(
                f"{model_name} returned an empty response."
            )

        except Exception as error:

            last_error = error
            error_text = str(error).lower()

            if "401" in error_text or "403" in error_text:
                raise error

            continue

    raise RuntimeError(
        "All available Gemini models failed. "
        "Google's service may be temporarily busy, or your "
        "API quota may have been reached. "
        f"Last error: {last_error}"
    )


# =====================================================
# 5. CHAT HISTORY STORAGE
# =====================================================

def load_chat_history():
    """Load saved conversations from the JSON file."""

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        return []


def save_chat_history():
    """Save all conversations to the JSON file."""

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(
                st.session_state.chat_history,
                file,
                indent=4,
                ensure_ascii=False
            )

    except OSError as error:
        st.error(f"Could not save chat history: {error}")



def create_new_chat():
    """Create and save a new conversation."""

    chat_id = str(uuid.uuid4())

    new_chat = {
        "id": chat_id,
        "title": "New Conversation",
        "created": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "updated": datetime.now().isoformat(),
        "messages": []
    }

    st.session_state.chat_history.insert(0, new_chat)
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.voice_result = None
    st.session_state.voice_audio = None

    # Update the Chat History dropdown selection
    st.session_state.history_picker = chat_id

    # Save immediately to JSON
    save_chat_history()



def save_current_chat():
    """Update and save the active conversation."""

    current_id = st.session_state.current_chat_id

    for chat in st.session_state.chat_history:

        if chat["id"] == current_id:

            chat["messages"] = st.session_state.messages
            chat["updated"] = datetime.now().isoformat()

            for message in st.session_state.messages:
                if message["role"] == "user":

                    title = message["content"].strip()

                    chat["title"] = (
                        title[:35] + "..."
                        if len(title) > 35
                        else title
                    )

                    break

            break

    save_chat_history()



def load_selected_chat(chat_id):
    """Open a previous conversation."""

    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:

            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat.get("messages", [])
            st.session_state.voice_result = None
            st.session_state.voice_audio = None

            # Keep dropdown synchronized
            st.session_state.history_picker = chat_id

            return



# =====================================================
# 6. INITIALIZE SESSION STATE
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice_result" not in st.session_state:
    st.session_state.voice_result = None

if "voice_audio" not in st.session_state:
    st.session_state.voice_audio = None

if "current_chat_id" not in st.session_state:

    if st.session_state.chat_history:

        latest_chat = st.session_state.chat_history[0]

        st.session_state.current_chat_id = latest_chat["id"]
        st.session_state.messages = latest_chat.get("messages", [])

    else:
        create_new_chat()


# =====================================================
# 7. AI RESPONSE FUNCTION
# =====================================================

def generate_response(question):

    prompt = (
        "You are a friendly and intelligent AI voice assistant. "
        "Answer accurately in simple conversational language. "
        "Use a maximum of about 100 words.\n\n"
        f"User question: {question}"
    )

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=200
    )

    answer, used_model = generate_with_fallback(
        prompt,
        config
    )


    return answer, used_model


# =====================================================
# 8. TEXT TO SPEECH FUNCTION
# =====================================================

def convert_to_speech(text):

    audio_buffer = io.BytesIO()

    speech = gTTS(
        text=text,
        lang="en",
        slow=False
    )

    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer


# =====================================================
# 9. PROCESS TEXT QUESTION
# =====================================================

def process_question(question):

    question = question.strip()

    if not question:
        return

    answer, used_model = generate_response(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.last_used_model = used_model

    save_current_chat()


# =====================================================
# 10. SIDEBAR AND CHAT HISTORY
# =====================================================

with st.sidebar:

    st.markdown("## 🎙️ AI Assistant")
    st.caption("Your intelligent voice companion")

    st.divider()

    st.markdown("### 💬 Chat History")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        create_new_chat()
        st.rerun()

    if st.session_state.chat_history:

        chat_ids = [
            chat["id"]
            for chat in st.session_state.chat_history
        ]

        def format_chat(chat_id):

            for chat in st.session_state.chat_history:
                if chat["id"] == chat_id:
                    return f"💬 {chat['title']}"

            return "Conversation"

        current_index = 0

        if st.session_state.current_chat_id in chat_ids:
            current_index = chat_ids.index(
                st.session_state.current_chat_id
            )

        selected_chat = st.selectbox(
            "Open previous conversation",
            options=chat_ids,
            index=current_index,
            format_func=format_chat,
            key="chat_history_selector"
        )

        if selected_chat != st.session_state.current_chat_id:
            load_selected_chat(selected_chat)
            st.rerun()

    else:
        st.caption("No previous conversations yet.")

    st.divider()

    st.markdown("### ⚙️ Settings")

    # Modern settings information cards
    st.markdown("""
    <div class="sidebar-info">
        <div class="sidebar-info-title">AI Model</div>
        <div class="sidebar-info-value">Google Gemini AI</div>
    </div>

    <div class="sidebar-info">
        <div class="sidebar-info-title">Speech Engine</div>
        <div class="sidebar-info-value">Google Text-to-Speech</div>
    </div>

    <div class="sidebar-info">
        <div class="sidebar-info-title">Application</div>
        <div class="sidebar-info-value">Streamlit + Python</div>
    </div>

    <div class="status-card">
        ● API key detected
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button(
        "🗑️ Clear Current Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.voice_result = None
        st.session_state.voice_audio = None

        save_current_chat()
        st.rerun()

    if st.button(
        "❌ Delete Current Chat",
        use_container_width=True
    ):

        current_id = st.session_state.current_chat_id

        st.session_state.chat_history = [
            chat for chat in st.session_state.chat_history
            if chat["id"] != current_id
        ]

        if st.session_state.chat_history:

            latest_chat = st.session_state.chat_history[0]
            load_selected_chat(latest_chat["id"])

        else:
            st.session_state.current_chat_id = None
            create_new_chat()

        save_chat_history()
        st.rerun()

    st.divider()

    st.caption("Problem Statement 19")
    st.caption("AI Voice Assistant with LLM")


# =====================================================
# 11. MAIN HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ INTELLIGENT CONVERSATIONAL AI</div>

    <h1>AI Voice Assistant</h1>

    <p>
        Your intelligent companion for conversations,
        instant answers and voice interactions.
        Ask questions through text or speak naturally
        using your microphone.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 12. FEATURE CARDS
# =====================================================

c1, c2, c3 = st.columns(3, gap="medium")

with c1:

    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎙️</div>
        <h3>Voice Interaction</h3>
        <p>
            Speak naturally and convert your speech
            into questions for the AI assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">✦</div>
        <h3>Intelligent Responses</h3>
        <p>
            Get helpful and relevant answers
            powered by Google Gemini AI.
        </p>
    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔊</div>
        <h3>Voice Output</h3>
        <p>
            Listen to AI-generated answers
            with text-to-speech functionality.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# =====================================================
# 13. CHAT CONVERSATION
# =====================================================

st.subheader("💬 Conversation")

if not st.session_state.messages:

    st.info(
        "Welcome! Type a question below or "
        "use the microphone to begin."
    )

for index, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            if st.button(
                "🔊 Listen",
                key=f"chat_listen_{index}"
            ):

                try:

                    with st.spinner("Preparing speech..."):

                        audio = convert_to_speech(
                            message["content"]
                        )

                    st.audio(
                        audio.getvalue(),
                        format="audio/mp3"
                    )

                except Exception as error:
                    st.error(f"Speech error: {error}")


# =====================================================
# 14. TEXT CHAT INPUT
# =====================================================

st.subheader("⌨️ Ask a Question")

with st.form("text_question_form", clear_on_submit=True):

    typed_question = st.text_input(
        "Enter your question",
        placeholder="Ask anything..."
    )

    send_button = st.form_submit_button(
        "🚀 Send Question",
        use_container_width=True
    )


if send_button:

    if typed_question.strip():

        try:

            with st.spinner(
                "🤖 Gemini is thinking... Please wait."
            ):
                process_question(typed_question)

            st.rerun()

        except Exception as error:

            st.error(f"AI Error: {error}")

            st.info(
                "Please wait for a short time and try again. "
                "If the problem continues, check your Gemini "
                "API quota and available models."
            )

    else:
        st.warning("Please type a question first.")


# =====================================================
# 15. VOICE INPUT AND PROCESSING
# =====================================================

st.divider()

st.subheader("🎤 Talk to Your Assistant")

st.caption(
    "Record your question and let Gemini understand your speech."
)

audio_file = st.audio_input(
    "🎙️ Record your question",
    sample_rate=16000
)

if audio_file is not None:

    audio_bytes = audio_file.getvalue()

    st.audio(
        audio_bytes,
        format=audio_file.type
    )

    st.caption(
        f"Recording size: {len(audio_bytes) / 1024:.1f} KB"
    )

    if st.button(
        "🚀 Process Voice",
        key="process_voice"
    ):

        start_time = time.perf_counter()

        try:

            with st.spinner(
                "🎧 Gemini is listening and preparing your answer..."
            ):

                audio_prompt = [

                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=audio_file.type or "audio/wav"
                    ),

                    """
                    You are a helpful AI voice assistant.

                    Listen carefully to the user's audio.

                    First identify the words spoken by the user.
                    Then answer the user's question.

                    Return your response in this exact format:

                    TRANSCRIPT:
                    [The words spoken by the user]

                    ANSWER:
                    [A clear and concise answer in simple language]

                    Keep the answer under 100 words.

                    If the audio is unclear, explain that
                    the speech could not be understood.
                    """
                ]

                config = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=350
                )

                output, used_model = generate_with_fallback(
                    audio_prompt,
                    config
                )

            if "ANSWER:" not in output:

                st.error(
                    "Could not understand the response. "
                    "Please record again."
                )

            else:

                transcript_part, answer = output.split(
                    "ANSWER:",
                    1
                )

                spoken_text = transcript_part.replace(
                    "TRANSCRIPT:",
                    ""
                ).strip()

                answer = answer.strip()

                if not answer:

                    st.error("Gemini returned an empty answer.")

                else:

                    elapsed = time.perf_counter() - start_time

                    st.session_state.voice_result = {
                        "transcript": spoken_text,
                        "answer": answer,
                        "elapsed": elapsed,
                        "model": used_model
                    }

                    st.session_state.voice_audio = None

                    # Save voice conversation in chat history.
                    st.session_state.messages.append({
                        "role": "user",
                        "content": spoken_text or "Voice question"
                    })

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    save_current_chat()

                    st.success("Voice processing completed!")

                    st.rerun()

        except Exception as error:

            st.error(
                f"Voice processing failed: {error}"
            )

            st.info(
                "Please try recording again after a short wait. "
                "If all models are unavailable, the Gemini service "
                "may be experiencing temporary high demand."
            )


# =====================================================
# 16. DISPLAY VOICE RESULT AND AUDIO
# =====================================================

if st.session_state.voice_result is not None:

    voice_result = st.session_state.voice_result

    st.divider()

    st.markdown("### 📝 Your Speech")

    st.info(
        voice_result["transcript"] or "Speech received"
    )

    st.markdown("### 🤖 AI Response")

    st.write(voice_result["answer"])

    st.caption(
        f"Processing time: {voice_result['elapsed']:.1f} seconds"
    )

    st.caption(
        f"Gemini model used: {voice_result['model']}"
    )

    st.markdown("### 🔊 Listen to AI Answer")

    if st.button(
        "🔊 Generate Voice Output",
        key="voice_listen_button"
    ):

        try:

            with st.spinner("Preparing your AI voice..."):

                speech_audio = convert_to_speech(
                    voice_result["answer"]
                )

                st.session_state.voice_audio = (
                    speech_audio.getvalue()
                )

        except Exception as error:

            st.error(f"Audio error: {error}")

    if st.session_state.voice_audio is not None:

        st.audio(
            st.session_state.voice_audio,
            format="audio/mp3"
        )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    AI Voice Assistant | Problem Statement 19<br>
    Powered by Python, Google Gemini and gTTS
</div>
""", unsafe_allow_html=True)
