"""
Shared helpers for the demo app and the benchmark script.

- Model discovery: models/<Arch>/model.keras + model_info.json (exported by the notebook)
- Preprocessing that matches the thesis notebook
  (letterbox -> 384 square -> 224 -> backbone preprocessing)
- Conversion of the trained .keras models to TensorFlow Lite (the format phones run)
- CPU latency benchmarks for Keras and TFLite models
- Basic hardware information about the machine running the test
"""

import os
import json
import time
import platform
import threading

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import (
    resnet as _resnet,
    mobilenet_v2 as _mnet,
    efficientnet as _effnet,
)

# Defaults; the values in each model's model_info.json are checked against these.
CLASS_NAMES = ["Flea Allergy Dermatitis", "Healthy Skin", "Ringworm", "Scabies"]
IMG_SIZE = (224, 224)
SQUARE_SIZE = 384                   # notebook letterboxes to 384x384 before resizing to 224
PAD_FILL = (114, 114, 114)          # same neutral gray as the notebook
MODELS_DIR = "models"
TFLITE_DIR = os.path.join(MODELS_DIR, "tflite")

PREPROCESS = {
    "ResNet50": _resnet.preprocess_input,
    "MobileNetV2": _mnet.preprocess_input,
    "EfficientNetB0": _effnet.preprocess_input,
}

# TFLite variants. "int8" is dynamic-range quantization: weights stored as 8-bit
# integers, which shrinks the file about 4x and needs no calibration images.
VARIANTS = ("fp32", "int8")


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def model_info(keras_file: str, folder: str = MODELS_DIR) -> dict:
    """Read the model_info.json saved next to a model ({} if there is none)."""
    p = os.path.join(folder, os.path.dirname(keras_file), "model_info.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def detect_architecture(keras_file: str) -> str:
    """Backbone from model_info.json, else guessed from the path, so we use the right preprocessing."""
    arch = model_info(keras_file).get("arch")
    if arch in PREPROCESS:
        return arch
    name = keras_file.lower()
    if "resnet50" in name:
        return "ResNet50"
    if "efficientnet" in name:
        return "EfficientNetB0"
    if "mobilenet" in name:
        return "MobileNetV2"
    return "EfficientNetB0"  # fallback default


def letterbox_square(img: Image.Image, size=IMG_SIZE[0], square_size=SQUARE_SIZE,
                     fill=PAD_FILL) -> Image.Image:
    """Pad the image to a square (keeping its shape), resize it to square_size x square_size
    as the notebook's dataset was stored, then resize to the model's size x size."""
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), tuple(fill))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    if square_size:
        canvas = canvas.resize((square_size, square_size), Image.BILINEAR)
    return canvas.resize((size, size), Image.BILINEAR)


def prepare_input(square_uint8: np.ndarray, arch: str) -> np.ndarray:
    """Apply a backbone's preprocessing to an already-letterboxed 224x224 array."""
    arr = PREPROCESS[arch](square_uint8.astype("float32").copy())
    return np.expand_dims(arr, axis=0).astype("float32")      # (1, 224, 224, 3)


def load_labelled_images(folder: str):
    """Read <folder>/<class name>/<images>, letterboxed to 224x224 uint8 arrays."""
    items = []
    for label, cls in enumerate(CLASS_NAMES):
        d = os.path.join(folder, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                with Image.open(os.path.join(d, f)) as im:
                    items.append((np.array(letterbox_square(im)), label))
    return items


def list_models(folder: str = MODELS_DIR):
    """Model paths relative to `folder`: <Arch>/model.keras, plus any loose *.keras files."""
    if not os.path.isdir(folder):
        return []
    found = []
    for entry in sorted(os.listdir(folder)):
        sub = os.path.join(folder, entry)
        if entry.endswith(".keras") and os.path.isfile(sub):
            found.append(entry)
        elif os.path.isdir(sub) and entry != os.path.basename(TFLITE_DIR):
            found += [f"{entry}/{f}" for f in sorted(os.listdir(sub)) if f.endswith(".keras")]
    return found


# ------------------------------------------------------------------
# TFLite conversion
# ------------------------------------------------------------------
def tflite_path(keras_file: str, variant: str) -> str:
    info = model_info(keras_file)
    if info.get("arch"):
        stem = info["arch"] + (f"_fold{info['fold']}" if "fold" in info else "")
    else:
        stem = os.path.splitext(keras_file)[0].replace("/", "_").replace("\\", "_")
    return os.path.join(TFLITE_DIR, f"{stem}_{variant}.tflite")


def tflite_is_current(keras_path: str, out_path: str) -> bool:
    """True if the .tflite exists and is newer than the .keras it was made from."""
    return os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(keras_path)


def convert_to_tflite(keras_model, out_path: str, variant: str) -> str:
    """Convert a loaded Keras model to a .tflite file (overwrites any existing file)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if variant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    data = converter.convert()
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)
    return out_path


def make_interpreter(path: str, num_threads: int):
    """Create a TFLite interpreter, preferring the newer LiteRT package if installed."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import warnings
        warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=num_threads)
    it.allocate_tensors()
    return it


class TFLiteModel:
    """Small wrapper so a TFLite model can be called like the Keras one.

    A TFLite interpreter must not be used by two threads at once. Streamlit shares cached
    objects between reruns and browser tabs, so every call is guarded by a lock.
    """

    def __init__(self, path: str, num_threads: int = 1):
        self.path = path
        self.num_threads = num_threads
        self.it = make_interpreter(path, num_threads)
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.lock = threading.Lock()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.lock:
            self.it.set_tensor(self.inp, x)
            self.it.invoke()
            return self.it.get_tensor(self.out)[0].copy()


def keras_predict_fn(keras_model):
    """Graph-compiled single-image prediction (avoids the eager overhead noted in Section 4.5)."""
    fn = tf.function(lambda x: keras_model(x, training=False))
    return lambda x: fn(tf.constant(x)).numpy()[0]


def measure_gflops(keras_model) -> float:
    """FLOPs for one 224x224 image, counted by the TensorFlow profiler as in the notebook
    (multiplications and additions counted separately, so FLOPs ~ 2 x MACs; Section 3.6.3)."""
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    spec = tf.TensorSpec([1, *IMG_SIZE, 3], tf.float32)
    concrete = tf.function(lambda x: keras_model(x, training=False)).get_concrete_function(spec)
    graph = convert_variables_to_constants_v2(concrete).graph
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    info = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(),
                                         cmd="op", options=opts)
    return info.total_float_ops / 1e9


# ------------------------------------------------------------------
# Benchmarking
# ------------------------------------------------------------------
def time_fn(fn, x, runs: int = 30, warmup: int = 5) -> dict:
    """Time single-image predictions (batch size 1), like Equation 3.18 in the thesis."""
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "std_ms": float(t.std()),
        "fps": float(1000.0 / np.median(t)),
    }


# ------------------------------------------------------------------
# Classification metrics (thesis Section 3.6.1, Equations 3.5-3.10)
# ------------------------------------------------------------------
def classification_metrics(y_true, y_pred, k: int = len(CLASS_NAMES)) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    cm = np.zeros((k, k), dtype=int)                 # C_ij: true class i, predicted j
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = n - tp - fp - fn
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall)
    specificity = div(tn, tn + fp)
    c, t_k, p_k = tp.sum(), cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    denom = np.sqrt((n ** 2 - (p_k ** 2).sum()) * (n ** 2 - (t_k ** 2).sum()))
    mcc = float((c * n - (p_k * t_k).sum()) / denom) if denom > 0 else 0.0
    return {
        "n": n, "confusion": cm, "accuracy": float(c / n) if n else 0.0,
        "precision": precision, "recall": recall, "f1": f1, "specificity": specificity,
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "mcc": mcc,
    }


def file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def system_info() -> dict:
    info = {
        "CPU": cpu_name(),
        "Logical cores": os.cpu_count(),
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "TensorFlow": tf.__version__,
        "GPU used": "No (CPU only)" if not tf.config.list_physical_devices("GPU") else "Yes",
    }
    try:
        import psutil
        info["RAM"] = f"{psutil.virtual_memory().total / 1024 ** 3:.1f} GB"
    except ImportError:
        pass
    return info


def process_ram_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        return None
