# Cat Skin Disease Classifier — Streamlit Demo (3-Model Comparison)

A small web app that loads **all three of your trained models** and, from a single
uploaded cat photo, shows what **ResNet50, EfficientNetB0, and MobileNetV2** each
predict, side by side. Classes: **Flea Allergy Dermatitis**, **Healthy Skin**, **Ringworm**,
**Scabies**.

You do not need to know Streamlit. Just follow the steps below in order.

---

## What you need

The app files plus one folder per model (each with the `model.keras` and
`model_info.json` exported by the notebook):

```
cat-app/
├── app.py
├── edge.py
├── benchmark.py
├── requirements.txt
├── README.md
└── models/
    ├── ResNet50/
    │   ├── model.keras
    │   └── model_info.json      ← architecture, fold, class names, preprocessing
    ├── EfficientNetB0/
    │   ├── model.keras
    │   └── model_info.json
    └── MobileNetV2/
        ├── model.keras
        └── model_info.json
```

`model_info.json` tells the app which backbone each model is (so it uses the right
preprocessing) and which classes it predicts. The app stops with an error if a model's
classes do not match the four listed above.

Use the **same fold** for all three models (fold 3 is recommended — it is the
representative fold in your thesis figures) so the comparison is fair.

---

## Step 1 — Install Python

If you do not already have Python:

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (TensorFlow 2.20 supports Python 3.9–3.12).
3. Run the installer. **On Windows, tick the box "Add Python to PATH"** before clicking Install.

Check it worked. Open a terminal (Windows: search "PowerShell"; Mac: open "Terminal") and type:

```
python --version
```

You should see something like `Python 3.11.x`. (On Mac you may need `python3` instead of `python`.)

---

## Step 2 — Get your model file out of Colab

Download each model's `model.keras` and `model_info.json` from Colab (same fold for all
three), and put them in **`models/ResNet50/`**, **`models/EfficientNetB0/`**, and
**`models/MobileNetV2/`** next to `app.py`.

When you replace the models with newly trained ones, just overwrite these files. The app
and `benchmark.py` notice that the `.keras` file is newer and rebuild the TFLite versions
in `models/tflite/` automatically.

---

## Step 3 — Open a terminal in the app folder

- **Windows:** open the `cat-app` folder in File Explorer, click the address bar,
  type `powershell`, and press Enter.
- **Mac:** right-click the folder → "New Terminal at Folder" (or `cd` into it).

You should now be "inside" the folder in the terminal.

---

## Step 4 — (Recommended) Make a clean virtual environment

This keeps the app's packages separate from the rest of your computer.

**Windows (PowerShell):**
```
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

After this you should see `(venv)` at the start of the terminal line.
(You only create it once; next time just run the `activate` line again.)

---

## Step 5 — Install the packages

```
pip install -r requirements.txt
```

This downloads Streamlit and TensorFlow. It can take a few minutes the first time.

> **Important:** the `requirements.txt` pins `tensorflow==2.20.0`. The current models were
> saved with TensorFlow 2.21 (see `model_info.json`) and have been checked to load and
> predict correctly in 2.20. A model saved in one version may fail to load in a very
> different one. If `pip` cannot find 2.20 on your machine, first check
> `python --version` (it must be 3.9–3.12), and see Troubleshooting below.

---

## Step 6 — Run the app

```
streamlit run app.py
```

Your web browser opens automatically at `http://localhost:8501`.
If it does not, copy that address from the terminal into your browser.

In the app:
1. The left sidebar lists every model it loaded.
2. Upload one cat photo (JPG, PNG, or WebP).
3. See all three models' predictions side by side, whether they agree, a grouped bar
   chart comparing their class probabilities, and a full probability table.

To stop the app, go back to the terminal and press **Ctrl + C**.

---

## What the page shows (everything is measured live)

The app is a single page with two sections. No numbers are copied from the thesis;
everything is measured from the loaded models on the computer running the app.

1. **Classify a photo** — upload a photo; every model, in every format, predicts it, with
   confidence, the time it took, and whether the phone format gives the same answer.
2. **Computational cost** — parameters, FLOPs (TensorFlow profiler, as in the notebook), model
   file size, and inference time (mean ± SD of 50 single-image predictions after 5 warm-ups).
   Nothing needs to be selected: every model is shown in every format at once — the original
   `.keras` model (all CPU cores) and the phone formats **TFLite fp32** and **TFLite int8** with
   **1 CPU thread** (closest to a budget phone or old laptop) and **4 threads**. The first time,
   each model is converted to TFLite and saved in `models/tflite/` (about a minute per model).
   "Download the TFLite files" lets you test them on a real Android phone with Google's TFLite
   benchmark tool.

The same measurements are available from the command line (saved to `benchmark_results.csv`):

```
python benchmark.py                          # 1 and 4 threads
python benchmark.py --threads 1 --runs 50
python benchmark.py --data test_images       # also accuracy (fold-3 test images only)
```

---

## Step 7 (Optional) — Put it online to share a link

If you want a link others can open (e.g., your panel):

1. Make a free account at https://streamlit.io/cloud
2. Create a free GitHub account and upload this folder (including `models/`) to a repository.
   - Note: GitHub rejects files over 100 MB. `models/ResNet50/model.keras` is about 228 MB,
     so upload it with Git LFS (or leave ResNet50 out of the online version).
3. In Streamlit Community Cloud, click **New app**, choose your repository, set the main
   file to `app.py`, and deploy. You get a public link.

---

## Troubleshooting

**"streamlit is not recognized" / "command not found"**
Your virtual environment is not active, or install failed. Re-run the `activate` line
from Step 4, then Step 5, then Step 6.

**"Could not find a version that satisfies tensorflow==2.20.0"**
Your Python version is probably outside 3.9–3.12, or you are on an unsupported platform.
Check `python --version`. As a fallback you can loosen the pin: open `requirements.txt`
and change the line to `tensorflow>=2.16` — but then re-test that the model loads,
because a version mismatch can cause load errors.

**Error when loading the model / "Unknown layer"**
This is almost always a TensorFlow version mismatch. Use the same TensorFlow version
you trained with (2.20.0). If you must use a different version, re-save the model in
Colab with that version.

**Predictions look wrong / random**
The app picks each model's preprocessing from the `arch` in its `model_info.json` (or, if
that file is missing, from the folder/file name). Make sure each `model_info.json` sits next
to its `model.keras` and names the right architecture. Wrong preprocessing is the most common
cause of random-looking predictions.

**Reminder:** this is a research prototype. It only knows four classes (three diseases and
healthy skin); a cat with any other skin problem will still be put into one of these four.
It is not a substitute for a veterinarian.
