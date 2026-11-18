"""Streamlit interface for the StudyPilot study assistant."""

import csv
import hmac
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from modules.analysis import extract_keywords, summarize
from modules.answer import answer_question
from modules.config import REASONING_EFFORT as DEFAULT_REASONING_EFFORT
from modules.loader import clean_text, read_pdf, split_into_chunks
from modules.search import _get_model, build_index
from modules.studykit import make_flashcards, make_quiz
from modules import storage


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PASTED_CHARS = 50_000
MODEL_CALL_LIMIT = 20
DAILY_MODEL_CALL_LIMIT = 20


def _get_secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        return ""
    return value if isinstance(value, str) else ""


def _resolve_openai_key(session_key=""):
    if session_key.strip():
        return session_key.strip()
    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return environment_key or _get_secret("OPENAI_API_KEY").strip()


def _reserve_model_call():
    calls_used = st.session_state["model_calls_used"]
    if calls_used >= MODEL_CALL_LIMIT:
        st.session_state["model_call_block_reason"] = "session"
        return False
    user_id = st.session_state.get("persistence_user_id", "")
    if user_id:
        try:
            if not storage.reserve_daily_model_call(
                user_id, daily_limit=DAILY_MODEL_CALL_LIMIT
            ):
                st.session_state["model_call_block_reason"] = "daily"
                return False
        except storage.StorageError:
            st.session_state["model_call_block_reason"] = "database"
            return False
    st.session_state["model_calls_used"] = calls_used + 1
    st.session_state["model_call_block_reason"] = ""
    return True


def _model_call_block_message():
    reason = st.session_state.get("model_call_block_reason")
    if reason == "daily":
        return "This private data ID has reached its 20-call daily limit (UTC)."
    if reason == "database":
        return "The daily limit could not be checked, so no model call was made."
    return "This session has reached its 20-call model limit."


def _clear_workspace_for_user_change():
    for key in (
        "study_indexes",
        "flashcards_by_text",
        "flashcards_generated",
        "flashcard_progress",
        "quizzes_by_text",
        "quiz_generated",
        "quiz_progress",
        "ask_chats",
    ):
        st.session_state[key] = {}
    st.session_state["active_saved_document_id"] = ""
    st.session_state["_reset_study_inputs_pending"] = True


def _activate_persistence_user(user_id, preserve_local=False):
    """Load a user's saved workspace and activate its private recovery ID."""
    saved_documents = storage.load_user_data(user_id)
    if preserve_local:
        for document_id, document in st.session_state["study_indexes"].items():
            storage.save_document(
                user_id,
                document_id,
                document.get("name", "Pasted notes"),
                document["text"],
                document["chunks"],
                document["index"]["embeddings"],
            )
            storage.save_study_aids(
                user_id,
                document_id,
                st.session_state["flashcards_by_text"].get(document_id),
                st.session_state["flashcard_progress"].get(document_id),
                st.session_state["quizzes_by_text"].get(document_id),
                st.session_state["quiz_progress"].get(document_id),
            )
        saved_documents = storage.load_user_data(user_id)
    restored = {}
    for document_id, document in saved_documents.items():
        try:
            index = build_index(
                document["chunks"], embeddings=document["embeddings"]
            )
        except Exception:
            raise storage.StorageError(
                "Some saved data could not be loaded."
            ) from None
        restored[document_id] = {
            "text": document["document_text"],
            "chunks": document["chunks"],
            "index": index,
            "name": document["document_name"],
        }

    _clear_workspace_for_user_change()
    st.session_state["study_indexes"] = restored
    for document_id, document in saved_documents.items():
        if document["flashcards"] is not None:
            st.session_state["flashcards_by_text"][document_id] = document[
                "flashcards"
            ]
            st.session_state["flashcards_generated"][document_id] = True
            progress = document["flashcard_progress"] or {
                "index": 0,
                "flipped": False,
                "ratings": [None] * len(document["flashcards"]),
                "complete": False,
            }
            st.session_state["flashcard_progress"][document_id] = progress
        if document["quiz"] is not None:
            st.session_state["quizzes_by_text"][document_id] = document["quiz"]
            st.session_state["quiz_generated"][document_id] = True
            progress = document["quiz_results"] or {
                "index": 0,
                "answers": [None] * len(document["quiz"]),
                "checked": [False] * len(document["quiz"]),
                "complete": False,
                "attempt": 0,
            }
            st.session_state["quiz_progress"][document_id] = progress

    st.session_state["persistence_user_id"] = user_id.strip()
    st.session_state["_clear_delete_confirmation_pending"] = True
    st.session_state["persistence_message"] = (
        f"Connected. Loaded {len(saved_documents)} saved document(s)."
    )
    st.session_state["persistence_error"] = ""


def _persist_study_aids(user_id, document_id):
    if not user_id or not document_id:
        return
    try:
        storage.save_study_aids(
            user_id,
            document_id,
            st.session_state["flashcards_by_text"].get(document_id),
            st.session_state["flashcard_progress"].get(document_id),
            st.session_state["quizzes_by_text"].get(document_id),
            st.session_state["quiz_progress"].get(document_id),
        )
        st.session_state["persistence_error"] = ""
    except storage.StorageError:
        st.session_state["persistence_error"] = (
            "Study progress could not be saved. It remains available in this session."
        )


@st.cache_resource(show_spinner="Loading the embedding model...")
def _cached_embedding_model():
    # Reuse the model cached by modules.search across Streamlit reruns.
    return _get_model()


def _read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="replace"), None

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", delete=False
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        return read_pdf(temp_path, return_pages=True)
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink()
            except FileNotFoundError:
                pass


def _load_sample_notes():
    st.session_state["study_pasted_text"] = (
        PROJECT_ROOT / "sample_notes.txt"
    ).read_text(encoding="utf-8")


def _example_questions(keywords):
    if not keywords:
        return [
            "What is the main idea of these notes?",
            "Which details are most important?",
            "What should I remember from this material?",
        ]
    question_templates = (
        "What do the notes say about {keyword}?",
        "How does {keyword} work?",
        "Why is {keyword} important?",
    )
    return [
        template.format(keyword=keywords[index % len(keywords)])
        for index, template in enumerate(question_templates)
    ]


def _render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for number, source in enumerate(sources, start=1):
            score = float(source.get("score", 0.0))
            st.markdown(f"**[{number}] Score: {score:.3f}**")
            page_start = source.get("page_start")
            page_end = source.get("page_end")
            if page_start is not None and page_end is not None:
                page_label = (
                    f"page {page_start}"
                    if page_start == page_end
                    else f"pages {page_start}–{page_end}"
                )
                st.caption(page_label)
            st.write(source.get("text", ""))


def _render_chat_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if message["role"] == "assistant" and sources:
            _render_sources(sources)


def _initialize_state():
    defaults = {
        "study_indexes": {},
        "flashcards_by_text": {},
        "flashcards_generated": {},
        "flashcard_progress": {},
        "quizzes_by_text": {},
        "quiz_generated": {},
        "quiz_progress": {},
        "ask_chats": {},
        "model_calls_used": 0,
        "model_call_block_reason": "",
        "persistence_user_id": "",
        "persistence_message": "",
        "persistence_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.set_page_config(page_title="StudyPilot", layout="wide")
_initialize_state()
if st.session_state.pop("_clear_storage_id_input_pending", False):
    st.session_state["persistence_user_id_input"] = ""
if st.session_state.pop("_reset_study_inputs_pending", False):
    st.session_state.pop("study_upload", None)
    st.session_state["study_pasted_text"] = ""
    st.session_state.pop("saved_document_select", None)
    st.session_state["active_saved_document_id"] = ""
if st.session_state.pop("_clear_delete_confirmation_pending", False):
    st.session_state["confirm_delete_persisted_data"] = False
database_configured = bool(storage.get_database_url())
st.title("StudyPilot")
st.write("StudyPilot turns your notes into summaries, answers, flashcards, and quizzes.")

app_access_code = _get_secret("APP_ACCESS_CODE")
access_code_required = bool(app_access_code)

with st.sidebar:
    st.header("Study material")
    access_granted = True
    if access_code_required:
        entered_access_code = st.text_input(
            "App access code", type="password", key="app_access_code_input"
        )
        access_granted = hmac.compare_digest(entered_access_code, app_access_code)
        if not access_granted:
            st.session_state.pop("session_openai_key", None)
            st.caption("Enter the access code to enable Ask, Flashcards, and Quiz.")

    session_openai_key = ""
    if access_granted:
        session_openai_key = st.text_input(
            "Use your own OpenAI key (optional)",
            type="password",
            key="session_openai_key",
            help="Used only in this session; it is not displayed, logged, or written to disk.",
        )
    api_key = _resolve_openai_key(session_openai_key) if access_granted else ""
    api_key_present = bool(api_key)

    effort_labels = {
        "low": "Fast (low)",
        "medium": "Balanced (medium)",
        "high": "Deep (high)",
        "max": "Max",
    }
    default_effort = (
        DEFAULT_REASONING_EFFORT
        if DEFAULT_REASONING_EFFORT in effort_labels
        else "max"
    )
    reasoning_effort = st.selectbox(
        "Reasoning effort",
        options=tuple(effort_labels),
        format_func=effort_labels.get,
        index=tuple(effort_labels).index(default_effort),
        key="luna_reasoning_effort",
    )
    st.caption("Max is slower, but allows more time for reasoning.")

    if database_configured:
        st.subheader("Optional database storage")
        if not access_granted:
            st.info("Enter the app access code to manage saved data.")
        else:
            if st.button("Create a private data ID", key="create_private_data_id"):
                new_id = storage.new_user_id()
                st.session_state["persistence_user_id_input"] = new_id
                try:
                    with st.spinner("Creating your private storage..."):
                        _activate_persistence_user(
                            new_id,
                            preserve_local=not bool(
                                st.session_state["persistence_user_id"]
                            ),
                        )
                except storage.StorageError:
                    st.session_state["persistence_error"] = (
                        "Could not connect to the configured database. "
                        "Your notes remain in this session."
                    )

            st.caption(
                "Connect before loading notes when possible. Connecting to another "
                "ID replaces this session's in-memory workspace."
            )
            with st.form("persistence_identity_form"):
                entered_data_id = st.text_input(
                    "Private data ID",
                    type="password",
                    key="persistence_user_id_input",
                    help="Enter a previously downloaded ID to restore its saved data.",
                )
                connect_clicked = st.form_submit_button("Connect")
            if connect_clicked:
                if not entered_data_id.strip():
                    st.session_state["persistence_error"] = (
                        "Enter a private data ID or create a new one."
                    )
                else:
                    try:
                        with st.spinner("Loading your saved data..."):
                            _activate_persistence_user(entered_data_id)
                    except storage.StorageError:
                        st.session_state["persistence_error"] = (
                            "Could not load saved data. Check the data ID and "
                            "database connection."
                        )

            if st.session_state.pop("_clear_delete_confirmation_pending", False):
                st.session_state["confirm_delete_persisted_data"] = False

            active_data_id = st.session_state["persistence_user_id"]
            if active_data_id:
                st.caption("Connected to saved data.")
                try:
                    daily_calls = storage.daily_model_calls(active_data_id)
                except storage.StorageError:
                    st.session_state["persistence_error"] = (
                        "The daily model-call count is unavailable; model calls "
                        "will be blocked until it can be checked."
                    )
                else:
                    st.caption(
                        f"Daily model calls: {daily_calls}/{DAILY_MODEL_CALL_LIMIT} (UTC)"
                    )
                st.download_button(
                    "Download private data ID",
                    data=active_data_id,
                    file_name="studypilot-private-data-id.txt",
                    mime="text/plain",
                    key="download_private_data_id_"
                    + hashlib.sha256(active_data_id.encode("utf-8")).hexdigest()[:12],
                )
                st.caption(
                    "Treat this ID like a password. Anyone with it can read or "
                    "delete the saved data."
                )
                confirm_delete = st.checkbox(
                    "Confirm deletion of my saved data",
                    key="confirm_delete_persisted_data",
                )
                if st.button(
                    "Delete my data",
                    key="delete_my_persisted_data",
                    disabled=not confirm_delete,
                ):
                    try:
                        storage.delete_user_data(active_data_id)
                    except storage.StorageError:
                        st.session_state["persistence_error"] = (
                            "Could not delete saved data from the database."
                        )
                    else:
                        _clear_workspace_for_user_change()
                        st.session_state["persistence_user_id"] = ""
                        st.session_state["persistence_message"] = "Saved data deleted."
                        st.session_state["persistence_error"] = ""
                        st.session_state["_clear_storage_id_input_pending"] = True
                        st.session_state["_clear_delete_confirmation_pending"] = True
            else:
                st.caption(
                    "Create or connect a private ID to save notes and enable the "
                    "database-backed daily call limit. Without an ID, this session "
                    "stays in memory."
                )

            if st.session_state["persistence_message"]:
                st.success(st.session_state["persistence_message"])
            if st.session_state["persistence_error"]:
                st.warning(st.session_state["persistence_error"])

    if st.session_state.pop("_reset_study_inputs_pending", False):
        st.session_state.pop("study_upload", None)
        st.session_state["study_pasted_text"] = ""
        st.session_state.pop("saved_document_select", None)
    active_data_id = st.session_state["persistence_user_id"]
    if active_data_id and st.session_state["study_indexes"]:
        saved_ids = list(st.session_state["study_indexes"])
        selected_saved_id = st.selectbox(
            "Saved notes",
            options=saved_ids,
            format_func=lambda item_id: st.session_state["study_indexes"][item_id].get(
                "name", "Saved notes"
            ),
            key="saved_document_select",
        )
        if st.button("Load saved notes", key="load_saved_document"):
            st.session_state.pop("study_upload", None)
            st.session_state["active_saved_document_id"] = selected_saved_id

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        key="study_upload",
        max_upload_size=5,
        help="Maximum file size: 5 MB.",
    )
    pasted_text = st.text_area(
        "Or paste your notes",
        height=180,
        key="study_pasted_text",
        max_chars=MAX_PASTED_CHARS + 1,
        help="Maximum accepted length: 50,000 characters.",
    )
    privacy_note = (
        "Uploaded or pasted notes are sent to OpenAI when you use Ask, Flashcards, "
        "or Quiz. Ask sends retrieved chunks; Flashcards and Quiz send up to "
        "6,000 characters. Summary and keywords run locally."
    )
    if database_configured and active_data_id:
        privacy_note += (
            " Your documents, chunks, embeddings, flashcards, quiz progress, and "
            "daily call count are saved in the configured database."
        )
    elif database_configured:
        privacy_note += (
            " Database persistence is configured, but notes remain in memory "
            "until you connect or create a private data ID."
        )
    st.warning(privacy_note)
    if uploaded_file is not None and pasted_text.strip():
        st.caption("The uploaded file is used while it is selected.")
    model_calls_used = st.session_state["model_calls_used"]
    model_call_status = st.empty()
    model_call_status.caption(
        f"Model calls this session: {model_calls_used}/{MODEL_CALL_LIMIT}"
    )
    st.caption("This per-session limit is a usage guardrail, not a security control.")
    if model_calls_used >= MODEL_CALL_LIMIT:
        st.warning("This session has reached its 20-call model limit.")
    if active_data_id:
        st.caption("A separate 20-call daily limit is tracked for this private ID (UTC).")

if access_granted and not api_key_present:
    st.error("Set OPENAI_API_KEY in the environment or secrets, or enter your own key.")

raw_text = ""
page_texts = None
load_error = None
if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        load_error = "That file is larger than 5 MB. Please choose a smaller PDF or TXT file."
    else:
        try:
            raw_text, page_texts = _read_uploaded_file(uploaded_file)
        except Exception:
            load_error = "The uploaded file could not be read. Try another PDF or TXT file."
elif pasted_text.strip():
    if len(pasted_text) > MAX_PASTED_CHARS:
        load_error = "Pasted notes must be 50,000 characters or fewer. Please shorten them."
    else:
        raw_text = pasted_text
else:
    active_saved_id = st.session_state.get("active_saved_document_id", "")
    saved_document = st.session_state["study_indexes"].get(active_saved_id)
    if saved_document:
        raw_text = saved_document["text"]

if load_error:
    st.error(load_error)

cleaned_text = clean_text(raw_text) if raw_text and not load_error else ""
if (
    uploaded_file is not None
    and uploaded_file.name.lower().endswith(".pdf")
    and not cleaned_text
    and not load_error
):
    st.info(
        "No readable text was extracted from this PDF. Scanned or image-only "
        "PDFs need OCR before they can be searched."
    )
text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest() if cleaned_text else ""
keywords = []
analysis_error = False
if cleaned_text:
    try:
        keywords = extract_keywords(cleaned_text)
    except (LookupError, OSError):
        analysis_error = True
current_data = None

if cleaned_text:
    index_cache = st.session_state["study_indexes"]
    if text_hash not in index_cache:
        try:
            _cached_embedding_model()
            with st.spinner("Preparing your notes for search..."):
                chunks = split_into_chunks(cleaned_text, page_texts=page_texts)
                index = build_index(chunks)
                document_name = (
                    uploaded_file.name if uploaded_file is not None else "Pasted notes"
                )
                index_cache[text_hash] = {
                    "text": cleaned_text,
                    "chunks": chunks,
                    "index": index,
                    "name": document_name,
                }
                active_data_id = st.session_state["persistence_user_id"]
                if active_data_id:
                    try:
                        storage.save_document(
                            active_data_id,
                            text_hash,
                            document_name,
                            cleaned_text,
                            chunks,
                            index["embeddings"],
                        )
                        st.session_state["persistence_error"] = ""
                    except storage.StorageError:
                        st.session_state["persistence_error"] = (
                            "These notes could not be saved. They remain available "
                            "in this session."
                        )
        except Exception:
            st.error("Could not prepare semantic search for these notes.")
    current_data = index_cache.get(text_hash)

if not cleaned_text:
    st.button(
        "Load sample notes",
        key="load_sample_notes",
        on_click=_load_sample_notes,
    )
    st.subheader("How it works")
    how_it_works = st.columns(3)
    steps = (
        ("1. Load notes", "Upload a PDF or text file, paste notes, or try the sample."),
        ("2. Explore", "Get a concise summary, keywords, and answers grounded in your notes."),
        ("3. Practice", "Create flashcards and a quiz to review the key ideas."),
    )
    for column, (heading, description) in zip(how_it_works, steps):
        with column:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.caption(description)

summary_tab, ask_tab, flashcards_tab, quiz_tab = st.tabs(
    ["Summary", "Ask", "Flashcards", "Quiz"]
)

with summary_tab:
    if not cleaned_text:
        st.info("Upload a PDF or TXT file, or paste notes in the sidebar to get started.")
    else:
        st.subheader("Summary")
        if analysis_error:
            st.warning(
                "Local summary and keyword analysis is unavailable because its "
                "text-processing resources could not be loaded."
            )
        else:
            try:
                summary_text = summarize(cleaned_text)
            except (LookupError, OSError):
                summary_error = True
            else:
                summary_error = False

            if summary_error:
                st.warning(
                    "Local summary and keyword analysis is unavailable because "
                    "its text-processing resources could not be loaded."
                )
            elif summary_text:
                st.write(summary_text)
            else:
                st.info("No summary could be made from this text.")

        if keywords:
            st.markdown("**Keywords:** " + ", ".join(keywords))
        elif not analysis_error:
            st.info("No keywords were found.")

with ask_tab:
    if not cleaned_text:
        st.info("Load some notes before asking a question.")
    elif not access_granted:
        st.info("Enter the app access code in the sidebar to use Ask.")
    elif current_data is None:
        st.info("Search is not ready for these notes yet.")
    else:
        chat_history = st.session_state["ask_chats"].setdefault(text_hash, [])
        if st.button("Clear chat", key=f"clear_chat_{text_hash[:12]}"):
            chat_history.clear()

        for message in chat_history:
            _render_chat_message(message)

        selected_example = None
        if not chat_history:
            st.caption("Try one of these questions")
            for number, example in enumerate(_example_questions(keywords), start=1):
                if st.button(
                    example,
                    key=f"ask_example_{text_hash[:12]}_{number}",
                    width="stretch",
                ):
                    selected_example = example

        question = st.chat_input(
            "Ask a question about your notes",
            key=f"ask_chat_input_{text_hash[:12]}",
        ) or selected_example
        if question:
            user_message = {"role": "user", "content": question}
            chat_history.append(user_message)
            _render_chat_message(user_message)

            if not api_key_present:
                assistant_message = {
                    "role": "assistant",
                    "content": "Set an OpenAI key in the environment, secrets, or sidebar.",
                    "sources": [],
                }
                _render_chat_message(assistant_message)
            elif not _reserve_model_call():
                assistant_message = {
                    "role": "assistant",
                    "content": _model_call_block_message(),
                    "sources": [],
                }
                _render_chat_message(assistant_message)
            else:
                model_call_status.caption(
                    f"Model calls this session: "
                    f"{st.session_state['model_calls_used']}/{MODEL_CALL_LIMIT}"
                )
                with st.chat_message("assistant"):
                    with st.spinner("Searching your notes and drafting an answer..."):
                        result = answer_question(
                            current_data["index"],
                            question.strip(),
                            reasoning_effort=reasoning_effort,
                            api_key=api_key,
                        )
                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                    st.markdown(assistant_message["content"])
                    _render_sources(assistant_message["sources"])
            chat_history.append(assistant_message)

with flashcards_tab:
    if not cleaned_text:
        st.info("Load some notes before creating flashcards.")
    elif not access_granted:
        st.info("Enter the app access code in the sidebar to use Flashcards.")
    else:
        if st.button("Generate flashcards", key=f"flashcards_button_{text_hash[:12]}"):
            if not api_key_present:
                st.warning("Set an OpenAI key in the environment, secrets, or sidebar.")
            elif not _reserve_model_call():
                st.warning(_model_call_block_message())
            else:
                model_call_status.caption(
                    f"Model calls this session: "
                    f"{st.session_state['model_calls_used']}/{MODEL_CALL_LIMIT}"
                )
                with st.spinner("Creating flashcards..."):
                    cards = make_flashcards(
                        cleaned_text[:6000],
                        n=5,
                        reasoning_effort=reasoning_effort,
                        api_key=api_key,
                    )
                st.session_state["flashcards_by_text"][text_hash] = cards
                st.session_state["flashcards_generated"][text_hash] = True
                st.session_state["flashcard_progress"][text_hash] = {
                    "index": 0,
                    "flipped": False,
                    "ratings": [None] * len(cards),
                    "complete": False,
                }
                _persist_study_aids(active_data_id, text_hash)

        cards = st.session_state["flashcards_by_text"].get(text_hash, [])
        if cards:
            csv_buffer = io.StringIO(newline="")
            writer = csv.writer(csv_buffer)
            writer.writerow(["front", "back"])
            writer.writerows((card["front"], card["back"]) for card in cards)
            st.download_button(
                "Download flashcards as CSV",
                data=csv_buffer.getvalue(),
                file_name="flashcards.csv",
                mime="text/csv",
                key=f"download_flashcards_{text_hash[:12]}",
            )

            progress = st.session_state["flashcard_progress"].setdefault(
                text_hash,
                {
                    "index": 0,
                    "flipped": False,
                    "ratings": [None] * len(cards),
                    "complete": False,
                },
            )
            if progress["complete"]:
                ratings = progress["ratings"]
                knew_count = ratings.count("knew")
                review_count = ratings.count("review")
                unrated_count = len(cards) - knew_count - review_count
                st.subheader("Flashcard review complete")
                st.metric("I knew it", knew_count)
                st.metric("Review again", review_count)
                if unrated_count:
                    st.caption(f"{unrated_count} card(s) were not rated.")
                if st.button("Review cards", key=f"review_cards_{text_hash[:12]}"):
                    progress["index"] = 0
                    progress["flipped"] = False
                    progress["complete"] = False
                    _persist_study_aids(active_data_id, text_hash)
                    st.rerun()
            else:
                card_index = progress["index"]
                card = cards[card_index]
                st.caption(f"Card {card_index + 1} of {len(cards)}")
                st.subheader(card["front"])
                if st.button("Flip", key=f"flip_card_{text_hash[:12]}"):
                    progress["flipped"] = not progress["flipped"]
                    _persist_study_aids(active_data_id, text_hash)
                if progress["flipped"]:
                    st.markdown("**Answer**")
                    st.write(card["back"])
                    rating = progress["ratings"][card_index]
                    if rating:
                        st.caption(
                            "Marked: "
                            + ("I knew it" if rating == "knew" else "Review again")
                        )
                    rating_columns = st.columns(2)
                    if rating_columns[0].button(
                        "I knew it", key=f"knew_card_{text_hash[:12]}"
                    ):
                        progress["ratings"][card_index] = "knew"
                        if card_index == len(cards) - 1:
                            progress["complete"] = True
                        else:
                            progress["index"] += 1
                            progress["flipped"] = False
                        _persist_study_aids(active_data_id, text_hash)
                        st.rerun()
                    if rating_columns[1].button(
                        "Review again", key=f"review_card_{text_hash[:12]}"
                    ):
                        progress["ratings"][card_index] = "review"
                        if card_index == len(cards) - 1:
                            progress["complete"] = True
                        else:
                            progress["index"] += 1
                            progress["flipped"] = False
                        _persist_study_aids(active_data_id, text_hash)
                        st.rerun()

                navigation_columns = st.columns(2)
                if navigation_columns[0].button(
                    "Previous",
                    key=f"previous_card_{text_hash[:12]}",
                    disabled=card_index == 0,
                ):
                    progress["index"] -= 1
                    progress["flipped"] = False
                    _persist_study_aids(active_data_id, text_hash)
                    st.rerun()
                next_label = "Finish review" if card_index == len(cards) - 1 else "Next"
                if navigation_columns[1].button(
                    next_label,
                    key=f"next_card_{text_hash[:12]}",
                ):
                    if card_index == len(cards) - 1:
                        progress["complete"] = True
                    else:
                        progress["index"] += 1
                    progress["flipped"] = False
                    _persist_study_aids(active_data_id, text_hash)
                    st.rerun()
        elif st.session_state["flashcards_generated"].get(text_hash):
            st.info("No flashcards were returned. You can try again.")
        else:
            st.info("Choose Generate flashcards to create cards from your notes.")

with quiz_tab:
    if not cleaned_text:
        st.info("Load some notes before starting a quiz.")
    elif not access_granted:
        st.info("Enter the app access code in the sidebar to use Quiz.")
    else:
        if st.button("Generate quiz", key=f"quiz_button_{text_hash[:12]}"):
            if not api_key_present:
                st.warning("Set an OpenAI key in the environment, secrets, or sidebar.")
            elif not _reserve_model_call():
                st.warning(_model_call_block_message())
            else:
                model_call_status.caption(
                    f"Model calls this session: "
                    f"{st.session_state['model_calls_used']}/{MODEL_CALL_LIMIT}"
                )
                with st.spinner("Creating your quiz..."):
                    quiz = make_quiz(
                        cleaned_text[:6000],
                        n=5,
                        reasoning_effort=reasoning_effort,
                        api_key=api_key,
                    )
                old_progress = st.session_state["quiz_progress"].get(text_hash, {})
                attempt = old_progress.get("attempt", -1) + 1
                st.session_state["quizzes_by_text"][text_hash] = quiz
                st.session_state["quiz_generated"][text_hash] = True
                st.session_state["quiz_progress"][text_hash] = {
                    "index": 0,
                    "answers": [None] * len(quiz),
                    "checked": [False] * len(quiz),
                    "complete": False,
                    "attempt": attempt,
                }
                _persist_study_aids(active_data_id, text_hash)

        quiz = st.session_state["quizzes_by_text"].get(text_hash, [])
        if quiz:
            st.download_button(
                "Download quiz as JSON",
                data=json.dumps(quiz, ensure_ascii=False, indent=2),
                file_name="quiz.json",
                mime="application/json",
                key=f"download_quiz_{text_hash[:12]}",
            )
            progress = st.session_state["quiz_progress"].setdefault(
                text_hash,
                {
                    "index": 0,
                    "answers": [None] * len(quiz),
                    "checked": [False] * len(quiz),
                    "complete": False,
                    "attempt": 0,
                },
            )
            if progress["complete"]:
                score = sum(
                    answer_index == item["answer_index"]
                    for answer_index, item in zip(progress["answers"], quiz)
                )
                st.subheader("Quiz complete")
                st.metric("Final score", f"{score}/{len(quiz)}")
                st.progress(1.0)
                if st.button("Retry", key=f"retry_quiz_{text_hash[:12]}"):
                    progress["index"] = 0
                    progress["answers"] = [None] * len(quiz)
                    progress["checked"] = [False] * len(quiz)
                    progress["complete"] = False
                    progress["attempt"] += 1
                    _persist_study_aids(active_data_id, text_hash)
                    st.rerun()
            else:
                question_index = progress["index"]
                item = quiz[question_index]
                checked_count = sum(progress["checked"])
                st.progress(checked_count / len(quiz))
                st.caption(f"Question {question_index + 1} of {len(quiz)}")
                option_labels = item["options"]
                selected_answer = st.radio(
                    item["question"],
                    options=range(4),
                    format_func=lambda option_index, labels=option_labels: labels[
                        option_index
                    ],
                    index=None,
                    key=(
                        f"quiz_answer_{text_hash[:12]}_"
                        f"{progress['attempt']}_{question_index}"
                    ),
                    disabled=progress["checked"][question_index],
                )
                if st.button(
                    "Check",
                    key=(
                        f"check_quiz_{text_hash[:12]}_"
                        f"{progress['attempt']}_{question_index}"
                    ),
                    disabled=progress["checked"][question_index],
                ):
                    if selected_answer is None:
                        st.warning("Choose an answer before checking.")
                    else:
                        progress["answers"][question_index] = selected_answer
                        progress["checked"][question_index] = True
                        _persist_study_aids(active_data_id, text_hash)
                        st.rerun()

                if progress["checked"][question_index]:
                    if progress["answers"][question_index] == item["answer_index"]:
                        st.success("Correct")
                    else:
                        st.error(
                            "Wrong. The correct answer is "
                            f"{item['options'][item['answer_index']]}"
                        )
                    st.write(item["explanation"])

                navigation_columns = st.columns(2)
                if navigation_columns[0].button(
                    "Previous",
                    key=f"previous_quiz_{text_hash[:12]}_{progress['attempt']}",
                    disabled=question_index == 0,
                ):
                    progress["index"] -= 1
                    _persist_study_aids(active_data_id, text_hash)
                    st.rerun()
                next_label = "Finish quiz" if question_index == len(quiz) - 1 else "Next"
                if navigation_columns[1].button(
                    next_label,
                    key=f"next_quiz_{text_hash[:12]}_{progress['attempt']}",
                    disabled=not progress["checked"][question_index],
                ):
                    if question_index == len(quiz) - 1:
                        progress["complete"] = True
                    else:
                        progress["index"] += 1
                    _persist_study_aids(active_data_id, text_hash)
                    st.rerun()
        elif st.session_state["quiz_generated"].get(text_hash):
            st.info("No quiz questions were returned. You can try again.")
        else:
            st.info("Choose Generate quiz to create questions from your notes.")
