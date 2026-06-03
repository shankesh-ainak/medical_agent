"""Streamlit UI for dscribe.

  Tab 1 — Discharge Summary: upload a patient PDF bundle, run the agent, view the
          structured draft and the step-by-step agent trace.
  Tab 2 — Ask the Chart: grounded Q&A over the same patient's notes, with page
          citations.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from dscribe.chat import answer_question
from dscribe.config import CONFIG
from dscribe.pipeline import run
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="dscribe — Discharge Summary Agent", layout="wide")

UPLOAD_DIR = CONFIG.storage_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- session state ---
st.session_state.setdefault("result", None)
st.session_state.setdefault("pdf_name", None)
st.session_state.setdefault("chat", [])  # list of {role, content}


def _save_upload(uploaded) -> str:
    data = uploaded.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:12]
    path = UPLOAD_DIR / f"{digest}_{uploaded.name}"
    path.write_bytes(data)
    return str(path)


st.title("🩺 dscribe — Agentic Discharge Summary")
if not CONFIG.openai_api_key:
    st.error("OPENAI_API_KEY is not set. Add it to a .env file (see .env.example).")

tab_summary, tab_chat = st.tabs(["📝 Discharge Summary", "💬 Ask the Chart"])

# ============================ TAB 1: SUMMARY ============================
with tab_summary:
    uploaded = st.file_uploader("Upload patient source-note PDF bundle", type=["pdf"])
    col1, col2 = st.columns([1, 3])
    with col1:
        use_cache = st.checkbox("Use cached OCR", value=True,
                                help="Reuse extraction from a previous run of the same file.")
        go = st.button("Generate discharge summary", type="primary",
                       disabled=uploaded is None)

    if go and uploaded is not None:
        pdf_path = _save_upload(uploaded)
        with st.spinner("Reading the chart, reconciling, drafting… (vision OCR can take a while)"):
            try:
                result = run(pdf_path, use_cache=use_cache, echo_trace=False)
                st.session_state.result = result
                st.session_state.pdf_name = uploaded.name
                st.session_state.chat = []  # reset chat for the new patient
            except Exception as exc:  # noqa: BLE001 — surface to the user
                st.exception(exc)

    result = st.session_state.result
    if result is not None:
        st.caption(f"Patient bundle: {st.session_state.pdf_name} · "
                   f"{len(result.pages)} pages · {len(result.trace)} agent steps")

        # Headline counts
        flags = result.summary.review_flags
        n_conflict = sum(1 for f in result.summary.fields.values()
                         if f.status.value == "conflict")
        n_pending = sum(1 for f in result.summary.fields.values()
                        if f.status.value == "pending")
        m1, m2, m3 = st.columns(3)
        m1.metric("Clinician-review flags", len(flags))
        m2.metric("Conflicts", n_conflict)
        m3.metric("Pending results", n_pending)

        st.markdown(result.markdown)
        st.download_button("Download draft (Markdown)", result.markdown,
                           file_name="discharge_summary_draft.md")

        with st.expander("🔍 Agent trace (reasoning → tool → inputs → result)"):
            for s in result.trace:
                st.markdown(
                    f"**Step {s['step']} · `{s['action']}`**  \n"
                    f"_reasoning_: {s['reasoning'] or '—'}  \n"
                    f"_inputs_: `{s['inputs']}`  \n"
                    f"_result_: {s['result']}"
                )
                st.divider()
    else:
        st.info("Upload a PDF and click *Generate* to produce a draft.")

# ============================ TAB 2: CHAT ============================
with tab_chat:
    if st.session_state.result is None:
        st.info("Generate a discharge summary first — the chat answers from that "
                "patient's indexed notes.")
    else:
        st.caption(f"Asking about: {st.session_state.pdf_name}. Answers cite source pages "
                   "and say so when the records don't contain the answer.")
        for turn in st.session_state.chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        prompt = st.chat_input("Ask about this patient (e.g. 'What was the discharge diagnosis?')")
        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching the notes…"):
                    try:
                        res = answer_question(
                            st.session_state.result.index, prompt,
                            history=st.session_state.chat[:-1],
                        )
                        answer = res["answer"]
                        if res["pages"]:
                            answer += "\n\n_sources: " + ", ".join(
                                f"p{p}" for p in res["pages"]) + "_"
                    except Exception as exc:  # noqa: BLE001
                        answer = f"Error: {exc}"
                st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
