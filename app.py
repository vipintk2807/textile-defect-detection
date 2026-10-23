"""Streamlit inspection app (presentation only; all logic lives in textile_defect_detection.workflow).

Start from the project root with:  python -m streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import logging
import sys
from pathlib import Path

import streamlit as st

try:
    import textile_defect_detection  # noqa: F401
except ImportError:  # works without 'pip install -e .'
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from textile_defect_detection import theme
from textile_defect_detection.errors import ImageValidationError, StorageError
from textile_defect_detection.export import records_to_csv_bytes, records_to_json_bytes
from textile_defect_detection.inspection_store import HistoryFilter
from textile_defect_detection.predictions import (STATUS_COMPLETED, STATUS_LABELS,
                                                  STATUS_LOW_CONFIDENCE)
from textile_defect_detection.workflow import (load_model_state, open_store, run_analysis,
                                               validate_upload)

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "inference.yaml"
NOTICE = ("This is a model-generated prediction, not a confirmed inspection result. "
          "Performance on unseen fabrics and factory conditions may differ from the evaluated "
          "dataset.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
st.set_page_config(page_title="Textile defect classification", layout="wide")
st.markdown(theme.app_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_model_state():
    """Load the model once per server process (not on every Streamlit rerun)."""
    return load_model_state(CONFIG_PATH)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def clear_image() -> None:
    """Clear selection and any result belonging to it."""
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state["outcome"] = None
    st.session_state["outcome_signature"] = None


def status_markup(css_class: str, text: str, symbol: str) -> str:
    # Status is always conveyed with text AND a symbol, never colour alone.
    return f'<p class="tdd-status {css_class}">{symbol} {esc(text)}</p>'


state = get_model_state()
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("outcome", None)
st.session_state.setdefault("outcome_signature", None)

# ------------------------------------------------------------------ header + model status
st.markdown('<p class="tdd-title">Textile defect classification</p>'
            '<p class="tdd-subtitle">Image-based classification of textile or garment images '
            'using a locally trained model.</p>', unsafe_allow_html=True)

if state.ready:
    info = state.engine.info
    st.markdown(status_markup("ok", f"Model ready: {info.model_id}", "\u2714"),
                unsafe_allow_html=True)
    if info.trained_on_synthetic_data:
        st.warning("\u26a0 This model was trained on SYNTHETIC demo data. It demonstrates the "
                   "software only and says nothing about real textile defects.")
else:
    st.markdown(status_markup("err", "Model unavailable", "\u2716"), unsafe_allow_html=True)
    st.error(state.message)
    if st.button("Reload model", key="reload_main"):
        get_model_state.clear()
        st.rerun()

st.info("\u2139 Predictions can be wrong and are not a substitute for validated industrial "
        "inspection. Classification labels the whole image; it does not locate defects. "
        "Images are processed locally and are not sent to any external service.")

config = state.config
supported = ", ".join(f.upper() for f in (config.input.supported_formats if config
                                          else ("jpg", "jpeg", "png", "webp")))
limit_mb = config.input.max_file_size_mb if config else 10

# ------------------------------------------------------------------ sidebar: history options
with st.sidebar:
    st.header("Options")
    save_history = False
    store = None
    if state.ready and config is not None:
        save_history = st.checkbox(
            "Save results to local history", value=False,
            help="Stores metadata and model scores (never the image) in a local SQLite file.")
        store, store_problem = open_store(config)
        if store_problem:
            st.warning(f"History unavailable: {store_problem}")
        else:
            st.caption(f"History file: {config.storage.database_path}")
    if st.button("Reload model", key="reload_sidebar"):
        get_model_state.clear()
        st.rerun()

# ------------------------------------------------------------------ upload + preview
st.subheader("1. Select an image")
st.caption(f"Supported formats: {supported}. Maximum file size: {limit_mb:g} MB.")
upload = st.file_uploader(
    "Upload a textile or garment image",
    type=list(config.input.supported_formats) if config else ["jpg", "jpeg", "png", "webp"],
    key=f"uploader_{st.session_state['uploader_key']}",
    disabled=not state.ready,
)

loaded = None
data = b""
signature = None
if upload is not None:
    data = upload.getvalue()
    signature = hashlib.sha256(data).hexdigest() + "|" + upload.name
    if signature != st.session_state["outcome_signature"]:
        st.session_state["outcome"] = None  # a new image never inherits an old result
    try:
        loaded = validate_upload(data, upload.name, config.input)
    except ImageValidationError as exc:
        st.markdown(status_markup("err", "Invalid image", "\u2716"), unsafe_allow_html=True)
        st.error(exc.user_message)
else:
    st.session_state["outcome"] = None
    st.session_state["outcome_signature"] = None
    if state.ready:
        st.caption("No image selected. Upload an image to begin.")

if loaded is not None:
    left, right = st.columns(2)
    with left:
        st.subheader("2. Preview")
        st.image(loaded.image, caption=f"{upload.name} ({loaded.width} x {loaded.height} px)")
        st.markdown(status_markup("info", "Image ready for analysis", "\u2139"),
                    unsafe_allow_html=True)

    with right:
        st.subheader("3. Analyze")
        outcome = st.session_state["outcome"]
        already_done = outcome is not None and outcome.ok  # a failed attempt may be retried
        c1, c2 = st.columns(2)
        analyze = c1.button("Analyze image", type="primary",
                            disabled=(not state.ready) or already_done)
        c2.button("Clear image", on_click=clear_image)

        if analyze and state.ready and not already_done:
            with st.spinner("Processing image..."):
                outcome = run_analysis(state.engine, data, upload.name, store=store,
                                       save_history=save_history, loaded=loaded)
            st.session_state["outcome"] = outcome
            st.session_state["outcome_signature"] = signature

        outcome = st.session_state["outcome"]
        if outcome is not None and st.session_state["outcome_signature"] == signature:
            if not outcome.ok:
                st.markdown(status_markup("err", "Inference failed", "\u2716"),
                            unsafe_allow_html=True)
                st.error(outcome.error_message)
            else:
                r = outcome.result
                p = r.prediction
                css, symbol = (("ok", "\u2714") if r.status == STATUS_COMPLETED
                               else ("warn", "\u26a0"))
                rows = "".join(
                    f'<div class="tdd-score-row"><div class="tdd-score-name">{esc(c.class_name)}'
                    f'</div><div class="tdd-bar"><div style="width:{c.score * 100:.1f}%"></div>'
                    f'</div><div class="tdd-score-value">{c.score:.3f}</div></div>'
                    for c in p.ranked)
                st.markdown(
                    '<div class="tdd-panel"><p class="tdd-label">Predicted class</p>'
                    f'<p class="tdd-prediction">{esc(p.predicted_class)}</p>'
                    + status_markup(css, STATUS_LABELS[r.status], symbol)
                    + f'<p class="tdd-label">Model score (softmax; not a calibrated probability), '
                      f'threshold {r.threshold:g}</p>{rows}'
                    + f'<p class="tdd-label">Model: {esc(r.model_id)} &middot; '
                      f'inference {r.processing_time_ms:.0f} ms</p></div>',
                    unsafe_allow_html=True)
                st.markdown(f'<div class="tdd-notice">{esc(NOTICE)}</div>', unsafe_allow_html=True)
                if outcome.saved:
                    st.success("Saved to local history.")
                elif outcome.saved is False:
                    st.error(f"Not saved to history: {outcome.save_error}")

# ------------------------------------------------------------------ history + export
if state.ready and store is not None:
    with st.expander("Inspection history and export"):
        st.caption("Entries are model predictions, not human-confirmed inspection results.")
        status_choice = st.selectbox(
            "Status", ["(all)", *[STATUS_LABELS[s] for s in (STATUS_COMPLETED, STATUS_LOW_CONFIDENCE)],
                       "Inference failed"])
        label_to_status = {v: k for k, v in STATUS_LABELS.items()}
        class_choice = st.selectbox("Predicted class", ["(all)", *state.engine.info.class_names])
        try:
            records = store.list(HistoryFilter(
                status=None if status_choice == "(all)" else label_to_status[status_choice],
                predicted_class=None if class_choice == "(all)" else class_choice))
        except StorageError as exc:
            records = []
            st.error(exc.user_message)
        if records:
            st.dataframe([{k: v for k, v in rec.items() if k != "predictions_json"}
                          for rec in records])
            d1, d2 = st.columns(2)
            d1.download_button("Download CSV", records_to_csv_bytes(records),
                               file_name="inspections.csv", mime="text/csv")
            d2.download_button("Download JSON", records_to_json_bytes(records),
                               file_name="inspections.json", mime="application/json")
        else:
            st.write("No saved inspections match.")
