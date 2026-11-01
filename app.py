"""
Cat Skin Disease Classifier — Streamlit demo (compares all models)
Bumakil & Omac — Mapúa University, 2026

One page, nothing to select — every model is run in every format automatically:
  - the original Keras model (.keras, all CPU cores), and
  - TensorFlow Lite fp32 and int8, the formats used by Android/iOS apps, limited to
    1 CPU thread (closest to a budget phone or old laptop) and 4 threads.

Sections:
  1. Classify a photo — every model and format predicts the uploaded photo.
  2. Computational cost — parameters, FLOPs, model file size, and inference time,
     all measured live on this computer.

Classes:  Flea Allergy Dermatitis | Healthy Skin | Ringworm | Scabies
Preprocessing matches the thesis notebook (see edge.py).

Run it with:   streamlit run app.py
"""

import os
import time

import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st
import tensorflow as tf

import edge
from edge import CLASS_NAMES, MODELS_DIR

TIMING_RUNS, TIMING_WARMUP = 50, 5  # same as the thesis timing procedure (Equation 3.18)

# Fixed colour per model (colour-blind-checked; same in every chart).
MODEL_COLORS = {"ResNet50": "#2a78d6", "MobileNetV2": "#eb6834", "EfficientNetB0": "#1baf7a"}
COLOR_SCALE = alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))

# Every way each model is run: (label, engine, CPU threads; None = all cores)
SETTINGS = [
    ("Original .keras · all cores", "keras", None),
    ("TFLite fp32 · 1 thread", "fp32", 1),
    ("TFLite fp32 · 4 threads", "fp32", 4),
    ("TFLite int8 · 1 thread", "int8", 1),
    ("TFLite int8 · 4 threads", "int8", 4),
]
SETTING_ORDER = [s[0] for s in SETTINGS]
# Formats (thread count changes speed, not predictions or file size)
FORMATS = [("Original .keras", "keras"), ("TFLite fp32", "fp32"), ("TFLite int8", "int8")]


# ------------------------------------------------------------------
# Cached loaders and measurements
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_keras(path: str):
    model = tf.keras.models.load_model(path)
    return model, edge.keras_predict_fn(model)


@st.cache_resource(show_spinner=False)
def ensure_tflite(keras_file: str, variant: str) -> str:
    """Convert once to models/tflite/ and reuse the file on later runs."""
    src = os.path.join(MODELS_DIR, keras_file)
    out = edge.tflite_path(keras_file, variant)
    if not edge.tflite_is_current(src, out):          # missing, or the .keras was replaced
        model, _ = load_keras(src)
        edge.convert_to_tflite(model, out, variant)
    return out


@st.cache_resource(show_spinner=False)
def load_tflite(path: str, threads: int):
    return edge.TFLiteModel(path, threads)


@st.cache_resource(show_spinner=False)
def gflops(path: str) -> float:
    model, _ = load_keras(path)
    return edge.measure_gflops(model)


@st.cache_data(show_spinner=False)
def system_info():
    return edge.system_info()


def get_runner(m, engine: str, threads):
    if engine == "keras":
        return m["keras_fn"]
    return load_tflite(ensure_tflite(m["file"], engine), threads)


def model_file(m, engine: str) -> str:
    if engine == "keras":
        return os.path.join(MODELS_DIR, m["file"])
    return ensure_tflite(m["file"], engine)


@st.cache_data(show_spinner=False)
def timing(_m, file: str, engine: str, threads, nonce: int) -> dict:
    """Mean ± SD of single-image predictions (cached until 'Re-measure' is pressed)."""
    rng = np.random.default_rng(42)
    x = edge.prepare_input(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8), _m["arch"])
    return edge.time_fn(get_runner(_m, engine, threads), x, runs=TIMING_RUNS, warmup=TIMING_WARMUP)


# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.set_page_config(page_title="Cat Skin Disease Classifier", page_icon="🐱", layout="wide")

st.title("Deep Learning-Based Image Classification of Cat Skin Diseases: A Comparative Study of ResNet50, MobileNetV2, and EfficientNetB0")
st.caption("Research prototype — Bumakil & Omac, Mapúa University")

st.warning(
    "This is a research demo, **not a veterinary diagnosis**. The models only know four "
    "classes — Flea Allergy Dermatitis, Ringworm, Scabies, and Healthy Skin — so a cat with "
    "a different skin problem will still be put into one of these four. Always consult a "
    "veterinarian for real diagnosis and treatment.",
    icon="⚠️",
)

available = edge.list_models(MODELS_DIR)
if not available:
    st.error(f"No .keras files found in the '{MODELS_DIR}/' folder.")
    st.info(
        f"Create a folder named **{MODELS_DIR}** next to app.py with one subfolder per "
        "model, each holding the `model.keras` and `model_info.json` exported by the notebook:\n\n"
        "- `ResNet50/model.keras` + `ResNet50/model_info.json`\n"
        "- `EfficientNetB0/model.keras` + `EfficientNetB0/model_info.json`\n"
        "- `MobileNetV2/model.keras` + `MobileNetV2/model_info.json`\n\n"
        "Use the same fold for all three so the comparison is fair. "
        "See the README for how to download them from Colab."
    )
    st.stop()

st.sidebar.header("Models loaded")
for f in available:
    info = edge.model_info(f)
    line = f"• {f}  ->  {edge.detect_architecture(f)}"
    if "fold" in info:
        line += f" (fold {info['fold']}"
        if "fold_accuracy" in info:
            line += f", test accuracy {info['fold_accuracy'] * 100:.1f}%"
        line += ")"
    st.sidebar.write(line)
st.sidebar.header("This computer")
for k, v in system_info().items():
    st.sidebar.caption(f"**{k}:** {v}")

models = []
for f in available:
    arch = edge.detect_architecture(f)
    with st.spinner(f"Loading {f}..."):
        km, kfn = load_keras(os.path.join(MODELS_DIR, f))
    info_classes = list(edge.model_info(f).get("class_names", CLASS_NAMES))
    if info_classes != CLASS_NAMES or km.output_shape[-1] != len(CLASS_NAMES):
        st.error(f"{f} predicts {km.output_shape[-1]} classes {info_classes}, but the app "
                 f"expects {len(CLASS_NAMES)}: {CLASS_NAMES}. Re-export the model or update "
                 "CLASS_NAMES in edge.py.")
        st.stop()
    models.append({"file": f, "arch": arch, "model": km, "keras_fn": kfn})
order = [m["arch"] for m in models]

with st.spinner("Preparing the TFLite (phone) versions — first run only, about a minute per model..."):
    for m in models:
        for _, eng in FORMATS[1:]:
            ensure_tflite(m["file"], eng)

st.markdown(
    "Every model runs in three formats: the **original .keras** model, and **TFLite fp32** "
    "and **TFLite int8**, the formats used inside Android/iOS apps (int8 stores weights as "
    "8-bit integers, about 4× smaller). TFLite is timed with **1 CPU thread** (closest to a "
    "budget phone or an old laptop) and **4 threads**."
)

# ==================================================================
# 1 · Classify a photo
# ==================================================================
st.header("1 · Classify a photo")
uploaded = st.file_uploader("Upload a cat skin photo (JPG, PNG, or WebP)",
                            type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload an image above to compare all models.")
else:
    image = Image.open(uploaded)
    square_uint8 = np.array(edge.letterbox_square(image))     # shared 224x224 image

    rows, results = [], []
    with st.spinner("Running every model in every format..."):
        for m in models:
            x = edge.prepare_input(square_uint8, m["arch"])
            ref_top = None
            for label, eng, thr in SETTINGS:
                runner = get_runner(m, eng, thr)
                runner(x)                               # warm-up, not timed
                t0 = time.perf_counter()
                probs = np.asarray(runner(x))
                ms = (time.perf_counter() - t0) * 1000
                top = int(np.argmax(probs))
                if eng == "keras":
                    ref_top = top
                    results.append({"arch": m["arch"], "probs": probs,
                                    "pred": CLASS_NAMES[top], "conf": float(probs[top])})
                rows.append({"Model": m["arch"], "Format": label, "Prediction": CLASS_NAMES[top],
                             "Confidence (%)": probs[top] * 100, "Time (ms)": ms,
                             "Same as original": "✅" if top == ref_top else "❗"})

    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Uploaded image", width="stretch")
    with right:
        st.subheader("What each model predicts")
        cols = st.columns(len(results))
        for c, r in zip(cols, results):
            with c:
                st.markdown(f"**{r['arch']}**")
                st.metric("Prediction", r["pred"])
                st.metric("Confidence", f"{r['conf'] * 100:.1f}%")
                if r["conf"] < 0.5:
                    st.caption("low confidence")
        preds = {r["pred"] for r in results}
        if len(preds) == 1:
            st.success(f"All models agree: {results[0]['pred']}")
        else:
            st.info("The models disagree — see the probabilities below to compare them.")

    st.subheader("Every model in every format")
    st.dataframe(
        pd.DataFrame(rows).style.format({"Confidence (%)": "{:.1f}", "Time (ms)": "{:.1f}"}),
        hide_index=True, width="stretch",
    )
    st.caption("Time for this one photo, batch size 1, after one warm-up run. "
               "'Same as original' compares the predicted class with the .keras model.")

    st.subheader("Class probabilities across models (original .keras)")
    prob_df = pd.DataFrame([
        {"Model": r["arch"], "Class": CLASS_NAMES[i], "Probability": float(r["probs"][i])}
        for r in results for i in range(len(CLASS_NAMES))
    ])
    st.altair_chart(
        alt.Chart(prob_df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Model:N", title=None, sort=order),
            x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top")),
            row=alt.Row("Class:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=["Model", "Class", alt.Tooltip("Probability:Q", format=".1%")],
        ).properties(height=80),
        width="stretch",
    )
    st.table(pd.DataFrame(
        {r["arch"]: [f"{p * 100:.2f}%" for p in r["probs"]] for r in results},
        index=CLASS_NAMES,
    ))

# ==================================================================
# 2 · Computational cost (all measured live)
# ==================================================================
st.header("2 · Computational cost on this computer")

nonce = st.session_state.get("timing_nonce", 0)
with st.spinner(f"Measuring inference time ({TIMING_RUNS} predictions per model and format)..."):
    time_rows = []
    for m in models:
        for label, eng, thr in SETTINGS:
            t = timing(m, m["file"], eng, thr, nonce)
            time_rows.append({"Model": m["arch"], "Setting": label, "Format": label.split(" · ")[0],
                              "Mean (ms)": t["mean_ms"], "SD (ms)": t["std_ms"],
                              "Median (ms)": t["median_ms"], "95th pct (ms)": t["p95_ms"],
                              "Images/s": 1000.0 / t["mean_ms"]})
times = pd.DataFrame(time_rows)

# Size and computation (do not depend on thread count)
cost_rows = []
for m in models:
    g = gflops(os.path.join(MODELS_DIR, m["file"]))
    row = {"Model": m["arch"], "Parameters": m["model"].count_params(), "GFLOPs": g}
    for label, eng in FORMATS:
        row[f"{label} (MB)"] = edge.file_mb(model_file(m, eng))
    cost_rows.append(row)
cost = pd.DataFrame(cost_rows)
ref = cost.loc[cost["Model"] == "ResNet50", "GFLOPs"]
if not ref.empty:
    cost["FLOPs vs. ResNet50"] = cost["GFLOPs"].map(
        lambda g: "1×" if g == ref.iloc[0] else f"{ref.iloc[0] / g:.1f}× fewer")

# Headline cards: one per model
cols = st.columns(len(models))
for col, m in zip(cols, models):
    c = cost[cost["Model"] == m["arch"]].iloc[0]
    t = times[times["Model"] == m["arch"]].set_index("Setting")
    with col:
        st.markdown(f"**{m['arch']}**")
        st.metric("Parameters", f"{c['Parameters'] / 1e6:.2f} M")
        st.metric("FLOPs per image", f"{c['GFLOPs']:.2f} G")
        st.metric("Model file", f"{c['TFLite int8 (MB)']:.1f} MB as a phone app",
                  help=f".keras {c['Original .keras (MB)']:.1f} MB · TFLite fp32 "
                       f"{c['TFLite fp32 (MB)']:.1f} MB · TFLite int8 {c['TFLite int8 (MB)']:.1f} MB")
        one = t[t.index.str.contains("1 thread")]
        best = one["Mean (ms)"].idxmin()
        st.metric("Inference, 1 CPU thread", f"{one.loc[best, 'Mean (ms)']:.1f} ms",
                  help=f"Fastest 1-thread setting: {best}")
        st.caption(f"{best.split(' · ')[0]} · ≈{one.loc[best, 'Images/s']:.0f} images/s")

st.subheader("Parameters, FLOPs, and file size")
st.dataframe(
    cost.style.format({"Parameters": "{:,.0f}", "GFLOPs": "{:.3f}",
                       **{f"{l} (MB)": "{:.1f}" for l, _ in FORMATS}}),
    hide_index=True, width="stretch",
)

chart_cols = st.columns(3)
for col, (field, title, fmt) in zip(chart_cols[:2], [
        ("Parameters", "Parameters", ",.0f"), ("GFLOPs", "GFLOPs per image", ".2f")]):
    base = alt.Chart(cost).encode(
        x=alt.X("Model:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{field}:Q", title=None),
    )
    col.markdown(f"**{title}**")
    col.altair_chart(
        base.mark_bar(cornerRadiusEnd=4).encode(
            color=alt.Color("Model:N", scale=COLOR_SCALE, legend=None),
            tooltip=["Model", alt.Tooltip(f"{field}:Q", format=fmt)])
        + base.mark_text(dy=-6).encode(text=alt.Text(f"{field}:Q", format=fmt)),
        width="stretch",
    )
size_long = cost.melt(id_vars="Model", value_vars=[f"{l} (MB)" for l, _ in FORMATS],
                      var_name="Format", value_name="MB")
size_long["Format"] = size_long["Format"].str.replace(" (MB)", "", regex=False)
chart_cols[2].markdown("**Model file (MB)**")
chart_cols[2].altair_chart(
    alt.Chart(size_long).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("MB:Q", title=None),
        y=alt.Y("Format:N", title=None, sort=[l for l, _ in FORMATS]),
        color=alt.Color("Model:N", scale=COLOR_SCALE, legend=None),
        row=alt.Row("Model:N", title=None, sort=order, header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=["Model", "Format", alt.Tooltip("MB:Q", format=".1f")],
    ).properties(height=alt.Step(14)),
    width="stretch",
)

st.subheader("Inference time")
st.altair_chart(
    alt.Chart(times).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Setting:N", title=None, sort=SETTING_ORDER),
        x=alt.X("Mean (ms):Q", title="Mean time per image (ms) — lower is better"),
        color=alt.Color("Model:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top")),
        row=alt.Row("Model:N", title=None, sort=order,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight="bold")),
        tooltip=["Model", "Setting", alt.Tooltip("Mean (ms):Q", format=".1f"),
                 alt.Tooltip("SD (ms):Q", format=".1f"), alt.Tooltip("Images/s:Q", format=".1f")],
    ).properties(height=alt.Step(18)),
    width="stretch",
)
time_table = times.drop(columns="Format")
st.dataframe(time_table.style.format({c: "{:.1f}" for c in time_table.columns[2:]}),
             hide_index=True, width="stretch")

c1, c2 = st.columns([1, 4])
if c1.button("↻ Re-measure"):
    st.session_state["timing_nonce"] = nonce + 1
    st.rerun()
c2.download_button(
    "⬇️ Download all cost results (CSV)",
    time_table.merge(cost, on="Model").to_csv(index=False),
    file_name="computational_cost.csv",
)

st.caption(
    f"All values are measured live. **Parameters** are counted by Keras. **FLOPs** are counted "
    f"by the TensorFlow profiler for one 224×224 image (multiplications and additions counted "
    f"separately, so FLOPs ≈ 2 × MACs); they depend only on the network, so they are the same "
    f"on any device and after TFLite conversion. **Model file**: the .keras file includes the "
    f"optimizer state; the TFLite files hold only what is needed to predict. **Inference time** "
    f"is the mean ± SD of {TIMING_RUNS} single-image predictions (batch size 1) after "
    f"{TIMING_WARMUP} warm-ups on a random 224×224 input, excluding photo decoding and resizing; "
    f"it varies with what else the computer is doing. Dynamic-range int8 mainly shrinks the "
    f"file; on x86 laptop CPUs fp32 TFLite can be faster, and phone (ARM) CPUs may differ. "
    f"A budget phone core is slower than a laptop core. As a usability guide, about 0.1 s feels "
    f"instant and up to 1 s keeps the user's flow (Nielsen, 1993)."
)

with st.expander("⬇️ Download the TFLite files (to test on a real phone)"):
    st.markdown(
        "These files can be run on an Android phone with Google's official "
        "[TFLite benchmark tool](https://ai.google.dev/edge/litert/models/measurement) "
        "to get real on-phone timings."
    )
    for m in models:
        c1, c2 = st.columns(2)
        for col, v in ((c1, "fp32"), (c2, "int8")):
            p = ensure_tflite(m["file"], v)
            with open(p, "rb") as fh:
                col.download_button(f"{m['arch']} {v} ({edge.file_mb(p):.1f} MB)", fh.read(),
                                    file_name=os.path.basename(p), key=f"dl_{m['arch']}_{v}")
