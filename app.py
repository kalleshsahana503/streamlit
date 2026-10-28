import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import os
import base64
import random
import markdown as md
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="Spark AI", page_icon="✨", layout="wide")

# =====================================================================
# CONFIG
# =====================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # set this before launching (env var / notebook cell)

SYSTEM_PROMPT = "You are a helpful, friendly AI assistant, similar to ChatGPT. Answer clearly and concisely."
DB_PATH = "chat_history.db"

TEXT_MODEL = "openai/gpt-oss-120b"      # current Groq flagship text model
VISION_MODEL = "qwen/qwen3.6-27b"       # current Groq vision-capable model

# Login credentials — a single admin account, plain username/password.
# No Google/OAuth involved. Replace with a real user store (DB table
# with hashed passwords) before putting this online.
USER_CREDENTIALS = {
    "admin": "admin123",
}


# =====================================================================
# DATABASE (background conversation storage)
# =====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_message(conn, username, role, content):
    try:
        conn.execute(
            "INSERT INTO conversations (username, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (username, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        st.warning(f"Couldn't save this message to history: {e}")


def load_history(conn, username):
    cur = conn.execute(
        "SELECT role, content FROM conversations WHERE username = ? ORDER BY id ASC",
        (username,),
    )
    return cur.fetchall()


def load_sessions(conn, username, limit=100):
    """Distinct day-buckets of past activity, used for the 'Recent Chats' list."""
    cur = conn.execute(
        """
        SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS n
        FROM conversations WHERE username = ?
        GROUP BY day ORDER BY day DESC LIMIT ?
        """,
        (username, limit),
    )
    return cur.fetchall()


conn = init_db()

# =====================================================================
# STYLING
# =====================================================================
CUSTOM_CSS = """
<style>
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

html, body,
[data-testid="stAppViewContainer"],
.stApp {
    background: linear-gradient(-45deg, #fbcfe8, #ede9fe, #dbeafe, #fdf2f8) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 18s ease infinite;
}

/* Freeze the whole page — nothing here scrolls except the dedicated
   chat-log box below. This is what keeps the sidebar, header, and
   input bar completely still no matter how long the conversation gets. */
html, body {
    height: 100vh !important;
    overflow: hidden !important;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden;
    height: 0;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    background: transparent !important;
    max-width: 1300px;
    padding: 1rem 2rem 2rem 2rem !important;
    position: relative;
    z-index: 1;
    height: 100vh !important;
    overflow: hidden !important;
}

.sidebar-logo {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.6rem;
}

/* Left-side panel. The page itself no longer scrolls at all (see the
   html/body rule above), so this just needs to sit in normal flow —
   it will naturally stay put since nothing around it moves. */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.55) !important;
    border-radius: 18px !important;
    max-height: calc(100vh - 8rem) !important;
    overflow-y: auto !important;
}
[data-testid="stVerticalBlockBorderWrapper"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #4c1d95 !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.7rem !important;
    border-radius: 10px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] button:hover {
    background: rgba(236, 72, 153, 0.14) !important;
    background-color: rgba(236, 72, 153, 0.14) !important;
}

/* The chat log itself is the ONLY thing on the page that scrolls. */
.chat-scroll-area {
    height: calc(100vh - 14rem);
    overflow-y: auto;
    padding-right: 10px;
}

/* Cap the video preview's height so it can't grow past the (frozen)
   page. */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    max-height: 320px !important;
    object-fit: cover !important;
}

/* ---- Merged ChatGPT-style input pill: the "+", camera, and text box
   all sit inside one seamless rounded bar instead of separate rows. ---- */
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] {
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 999px;
    padding: 6px 10px;
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.08);
    align-items: center !important;
}
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] button,
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] [data-testid="baseButton-secondary"],
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] [data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-width: 0 !important;
    box-shadow: none !important;
    color: #6d28d9 !important;
    font-size: 1.1rem !important;
}
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] button:hover {
    background: rgba(109, 40, 217, 0.08) !important;
    background-color: rgba(109, 40, 217, 0.08) !important;
}
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] [data-testid="stChatInput"],
div:has(> .input-bar-marker) + div[data-testid="stHorizontalBlock"] [data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Vertically nudge the login form toward the middle of the page */
.login-vcenter { margin-top: 7vh; }

.spark-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #a78bfa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s linear infinite;
    margin-bottom: 0;
}
.spark-subtitle {
    text-align: center;
    color: #6d28d9;
    opacity: 0.75;
    margin-top: 0.2rem;
    margin-bottom: 1.6rem;
}

/* Compact, left-aligned header for the chat page's top-left corner
   (unlike the big centered title used on the login page). */
.spark-title-corner {
    text-align: left;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #a78bfa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s linear infinite;
    margin-bottom: 0;
}
.spark-subtitle-corner {
    text-align: left;
    color: #6d28d9;
    opacity: 0.75;
    font-size: 0.85rem;
    margin-top: 0.1rem;
}

.bubble-row { display: flex; align-items: flex-start; gap: 10px; margin: 12px 0; }
.bubble-row.user { justify-content: flex-end; }
.bubble-row.bot { justify-content: flex-start; }

.avatar {
    width: 38px; height: 38px; min-width: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.avatar.bot { background: #ddd6fe; }
.avatar.user { background: #f9a8d4; }

.bubble { max-width: 75%; padding: 12px 18px; font-size: 0.95rem; line-height: 1.4; }
.bubble.bot { background: #ede9fe; color: #4c1d95; border-radius: 18px 18px 18px 4px; }
.bubble.user { background: #fce7f3; color: #831843; border-radius: 18px 18px 4px 18px; }
.bubble img.chat-img { max-width: 220px; border-radius: 12px; margin-bottom: 6px; display: block; }

div[data-testid="stHorizontalBlock"] button { border-radius: 12px !important; }

/* ---- Pink-bordered login inputs ---- */
div[data-testid="stTextInput"] > div {
    border: 2px solid #ec4899 !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.6) !important;
    box-shadow: 0 0 0 0 rgba(236, 72, 153, 0);
    transition: box-shadow 0.2s ease-in-out;
}
div[data-testid="stTextInput"] > div:focus-within {
    box-shadow: 0 0 10px 2px rgba(236, 72, 153, 0.55);
}
div[data-testid="stTextInput"] input {
    color: #831843 !important;
}

/* ---- Twinkling star field (login + chat pages) ---- */
.star-field {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    overflow: hidden;
    z-index: 0;
}
.star {
    position: absolute;
    background: #ffffff;
    border-radius: 50%;
    will-change: opacity, transform;
    box-shadow: 0 0 6px 1px rgba(255, 255, 255, 0.85);
}
@keyframes twinkle {
    0%   { opacity: 0.1; transform: scale(0.6); }
    50%  { opacity: 0.9; transform: scale(1.3); }
    100% { opacity: 0.1; transform: scale(0.6); }
}

/* ---- ChatGPT-style markdown content inside bubbles ---- */
.bubble p { margin: 0 0 0.5em 0; }
.bubble p:last-child { margin-bottom: 0; }
.bubble table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 0.85rem;
}
.bubble th, .bubble td {
    border: 1px solid rgba(76, 29, 149, 0.25);
    padding: 6px 10px;
    text-align: left;
}
.bubble th { background: rgba(167, 139, 250, 0.25); }
.bubble code {
    background: rgba(76, 29, 149, 0.08);
    padding: 2px 5px;
    border-radius: 4px;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.85em;
}
.bubble pre {
    background: #1e1b2e;
    color: #f4f0ff;
    padding: 12px 14px;
    border-radius: 10px;
    overflow-x: auto;
    margin: 0.5em 0;
}
.bubble pre code {
    background: transparent;
    padding: 0;
    color: inherit;
}
.bubble ul, .bubble ol { margin: 0.3em 0 0.3em 1.2em; }
.bubble strong { font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_stars(count=40, size_range=(1, 2.5), duration_range=(1.0, 2.2), delay_range=(0, 2.0)):
    """Fixed-position twinkling star overlay.
    Call with the defaults for a calm background field.
    Call again with a bigger count/size/faster duration right after a
    message is sent for a bright 'burst' flicker on top of it."""
    stars_html = []
    for _ in range(count):
        top = random.uniform(0, 100)
        left = random.uniform(0, 100)
        size = random.uniform(*size_range)
        delay = random.uniform(*delay_range)
        duration = random.uniform(*duration_range)
        stars_html.append(
            f'<div class="star" style="top:{top:.2f}%; left:{left:.2f}%; '
            f'width:{size:.1f}px; height:{size:.1f}px; '
            f'animation: twinkle {duration:.2f}s ease-in-out {delay:.2f}s infinite;"></div>'
        )
    st.markdown(f'<div class="star-field">{"".join(stars_html)}</div>', unsafe_allow_html=True)


def render_star_burst():
    """A brighter, faster, denser flicker — fired once, right after a message send."""
    render_stars(count=90, size_range=(2, 4.5), duration_range=(0.35, 0.9), delay_range=(0, 0.4))


def scroll_to_bottom():
    """Scrolls the dedicated chat-log box down so the newest message is
    in view and older ones read as scrolling upward — the page itself
    no longer scrolls at all, only this box does."""
    components.html(
        "<script>"
        "var el = window.parent.document.getElementById('chat-scroll-area');"
        "if (el) { el.scrollTop = el.scrollHeight; }"
        "</script>",
        height=0,
    )


# =====================================================================
# SESSION STATE
# =====================================================================
defaults = {
    "logged_in": False,
    "username": None,
    "messages": [],
    "show_menu": True,   # sidebar panel open/closed
    "show_recent": False,
    "show_settings": False,
    "show_uploader": False,
    "show_camera_input": False,
    "attached_image": None,   # {"bytes": ..., "mime": ...}
    "runtime_api_key": "",
    "just_sent": False,       # triggers one star burst on the next render
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def login_page():
    render_stars()
    if st.session_state.pop("just_sent", False):
        render_star_burst()
    st.markdown('<div class="spark-title login-vcenter">✨ Spark AI ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="spark-subtitle">Sign in to start chatting</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    rows = load_history(conn, username)
                    st.session_state.messages = [{"role": r, "content": c, "image_b64": None, "mime": None} for r, c in rows]
                    st.session_state.just_sent = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.caption("Login")


def bubble_html(role, content, image_b64=None, mime=None):
    """Returns the HTML for one chat bubble (doesn't render it directly) —
    all bubbles get joined into a single HTML blob so they can share one
    scrollable container."""
    rendered = md.markdown(content, extensions=["tables", "fenced_code", "nl2br"]) if content else ""
    img_tag = f'<img class="chat-img" src="data:{mime};base64,{image_b64}" />' if image_b64 else ""
    if role == "user":
        return (
            f'<div class="bubble-row user"><div class="bubble user">{img_tag}{rendered}</div>'
            f'<div class="avatar user">🧑</div></div>'
        )
    else:
        return (
            f'<div class="bubble-row bot"><div class="avatar bot">🤖</div>'
            f'<div class="bubble bot">{img_tag}{rendered}</div></div>'
        )


def render_sidebar_panel():
    """Left-side panel built from a bordered container inside a column —
    this is the approach that actually rendered correctly for you
    earlier; native st.sidebar was not showing up in your setup."""
    with st.container(border=True):
        st.markdown('<div class="sidebar-logo">✨ Spark AI</div>', unsafe_allow_html=True)

        active_api_key = GROQ_API_KEY or st.session_state.runtime_api_key
        if not active_api_key:
            st.warning("No Groq API key detected.")
            key_input = st.text_input("Paste your Groq API key", type="password", key="api_key_input")
            if key_input:
                st.session_state.runtime_api_key = key_input
                st.rerun()
            st.divider()

        if st.button("➕  New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.show_recent = False
            st.rerun()

        if st.button("🕐  Recent Chats", use_container_width=True):
            st.session_state.show_recent = not st.session_state.show_recent

        if st.session_state.show_recent:
            sessions = load_sessions(conn, st.session_state.username)
            if sessions:
                for day, n in sessions:
                    st.caption(f"📅 {day} — {n} messages")
            else:
                st.caption("No past conversations yet.")

        if st.button("🖼️  Images", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader

        if st.button("⚙️  Settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings

        if st.session_state.show_settings:
            st.divider()
            st.caption(f"👤 Logged in as **{st.session_state.username}**")

        st.divider()
        if st.button("🚪  Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()


def chat_page():
    render_stars()
    if st.session_state.pop("just_sent", False):
        render_star_burst()

    active_api_key = GROQ_API_KEY or st.session_state.runtime_api_key

    side_col, main_col = st.columns([1, 3.3])

    with side_col:
        render_sidebar_panel()

    with main_col:
        top_l, top_r1, top_r2 = st.columns([6, 1, 1])
        with top_l:
            st.markdown('<div class="spark-title-corner">✨ Spark AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="spark-subtitle-corner">My Spark AI Chatbot</div>', unsafe_allow_html=True)
        with top_r1:
            st.button("🚀 Deploy", use_container_width=True)
        with top_r2:
            st.button("⋮", use_container_width=True)

        if not active_api_key:
            st.warning("GROQ_API_KEY isn't set — open the sidebar and paste your key to continue.")

        # ---- Attach panels: file upload and camera ----
        if st.session_state.show_uploader:
            up_col1, up_col2 = st.columns([5, 1])
            with up_col1:
                uploaded_image = st.file_uploader(
                    "Upload an image from your computer", type=["png", "jpg", "jpeg", "webp"], key="uploader"
                )
            with up_col2:
                st.write("")
                if st.button("✖ Close", key="close_uploader", use_container_width=True):
                    st.session_state.show_uploader = False
                    st.rerun()
            if uploaded_image is not None:
                st.session_state.attached_image = {
                    "bytes": uploaded_image.getvalue(),
                    "mime": getattr(uploaded_image, "type", None) or "image/png",
                }

        if st.session_state.show_camera_input:
            cam_label_col, cam_close_col = st.columns([6, 1])
            with cam_label_col:
                st.caption("Camera preview")
            with cam_close_col:
                if st.button("✖ Close", key="close_camera", use_container_width=True):
                    st.session_state.show_camera_input = False
                    st.rerun()
            camera_image = st.camera_input("Take a photo", key="camera", label_visibility="collapsed")
            if camera_image is not None:
                st.session_state.attached_image = {
                    "bytes": camera_image.getvalue(),
                    "mime": getattr(camera_image, "type", None) or "image/png",
                }

        # ---- Preview of whatever is currently attached, with Send / Remove ----
        send_image_clicked = False
        if st.session_state.attached_image:
            pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
            with pcol1:
                st.image(st.session_state.attached_image["bytes"], width=160, caption="Ready to send")
            with pcol2:
                send_image_clicked = st.button("📤 Send image", use_container_width=True, key="send_image_btn")
            with pcol3:
                if st.button("✖ Remove", use_container_width=True, key="remove_image_btn"):
                    st.session_state.attached_image = None
                    st.session_state.show_uploader = False
                    st.session_state.show_camera_input = False
                    st.rerun()

        if not st.session_state.messages:
            bubbles_html = bubble_html("bot", "Hello 👋 I'm Spark AI. How can I help you today?")
        else:
            bubbles_html = "".join(
                bubble_html(msg["role"], msg["content"], msg.get("image_b64"), msg.get("mime"))
                for msg in st.session_state.messages
            )
        st.markdown(f'<div class="chat-scroll-area" id="chat-scroll-area">{bubbles_html}</div>', unsafe_allow_html=True)
        if st.session_state.messages:
            scroll_to_bottom()

        st.markdown('<div class="input-bar-marker"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 10])
        with c1:
            if st.button("➕", use_container_width=True, help="Attach a file"):
                st.session_state.show_uploader = not st.session_state.show_uploader
                st.rerun()
        with c2:
            if st.button("📷", use_container_width=True, help="Camera"):
                st.session_state.show_camera_input = not st.session_state.show_camera_input
                st.rerun()
        with c3:
            prompt = st.chat_input("Ask anything")

        # A message is sent when the user hits Enter with text, OR clicks
        # "Send image" (in which case a default caption is used if none typed).
        if prompt or send_image_clicked:
            user_text = prompt if prompt else "Please describe this image."
            attached = st.session_state.attached_image

            user_image_b64, user_mime = None, None
            if attached:
                user_image_b64 = base64.b64encode(attached["bytes"]).decode("utf-8")
                user_mime = attached["mime"]

            st.session_state.messages.append(
                {"role": "user", "content": user_text, "image_b64": user_image_b64, "mime": user_mime}
            )
            save_message(conn, st.session_state.username, "user", user_text)

            lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    lc_messages.append(AIMessage(content=m["content"]))

            model_name = TEXT_MODEL
            human_content = user_text

            if attached:
                model_name = VISION_MODEL
                human_content = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:{user_mime};base64,{user_image_b64}"}},
                ]
                # clear the attachment now that it's been sent
                st.session_state.attached_image = None
                st.session_state.show_uploader = False
                st.session_state.show_camera_input = False

            lc_messages.append(HumanMessage(content=human_content))

            try:
                llm = ChatGroq(model=model_name, api_key=active_api_key, temperature=0.7)
                response = llm.invoke(lc_messages)
                reply = response.content
            except Exception as e:
                reply = f"⚠️ Error calling Groq API: {type(e).__name__}: {e}"

            st.session_state.messages.append({"role": "assistant", "content": reply, "image_b64": None, "mime": None})
            save_message(conn, st.session_state.username, "assistant", reply)
            st.session_state.just_sent = True  # trigger a bright star burst on the next render
            st.rerun()


if st.session_state.logged_in:
    chat_page()
else:
    login_page()