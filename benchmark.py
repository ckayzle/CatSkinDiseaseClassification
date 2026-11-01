"""
Low-cost device benchmark for the three trained models.

What it does:
  1. Converts every models/<Arch>/model.keras file to TensorFlow Lite (fp32 and int8),
     saved in models/tflite/ (done once; redone only if the .keras file is replaced).
  2. Times single-image CPU inference for the Keras model and both TFLite
     versions, with the TFLite runs limited to a chosen number of CPU threads
     (1 thread approximates one core of a budget phone or old laptop).
  3. Optionally (--data) measures accuracy and Keras/TFLite agreement on a
     folder of labelled images laid out as  <data>/<class name>/<image files>.
     Use ONLY the fold-3 test images there: the fold-3 models were trained on
     the other folds, so any other image would inflate the result.

Results are printed and saved to benchmark_results.csv.

Usage:
  python benchmark.py
  python benchmark.py --threads 1 2 4 --runs 50
  python benchmark.py --data path/to/fold3_test_images
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

import edge


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4],
                    help="CPU thread counts to test for TFLite (default: 1 4)")
    ap.add_argument("--runs", type=int, default=30, help="timed runs per setting (default 30)")
    ap.add_argument("--data", help="folder of fold-3 test images, one subfolder per class")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    files = edge.list_models()
    if not files:
        raise SystemExit("No .keras files in models/ (expected models/<Arch>/model.keras)")

    print("Machine:")
    for k, v in edge.system_info().items():
        print(f"  {k}: {v}")
    print()

    data = edge.load_labelled_images(args.data) if args.data else None
    rng = np.random.default_rng(42)
    sample = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    rows = []
    for f in files:
        arch = edge.detect_architecture(f)
        print(f"== {arch} ({f})")
        km = tf.keras.models.load_model(os.path.join(edge.MODELS_DIR, f))
        params = km.count_params()
        gflops = edge.measure_gflops(km)
        print(f"   parameters: {params:,}   GFLOPs (measured): {gflops:.3f}")
        keras_fn = edge.keras_predict_fn(km)
        x = edge.prepare_input(sample, arch)

        engines = [("Keras (TensorFlow)", "keras", None, keras_fn,
                    edge.file_mb(os.path.join(edge.MODELS_DIR, f)))]
        for v in edge.VARIANTS:
            p = edge.tflite_path(f, v)
            if not edge.tflite_is_current(os.path.join(edge.MODELS_DIR, f), p):
                print(f"   converting to TFLite {v} ...")
                edge.convert_to_tflite(km, p, v)
            for t in args.threads:
                engines.append((f"TFLite {v}", v, t, edge.TFLiteModel(p, t), edge.file_mb(p)))

        for label, variant, threads, fn, size in engines:
            r = edge.time_fn(fn, x, runs=args.runs)
            row = {
                "model": arch, "engine": label,
                "threads": threads if threads else "all",
                "params": params, "gflops": round(gflops, 3),
                "file_mb": round(size, 2),
                **{k: round(v, 2) for k, v in r.items()},
            }
            if data is not None:
                preds = [int(np.argmax(fn(edge.prepare_input(sq, arch)))) for sq, _ in data]
                ref = [int(np.argmax(keras_fn(edge.prepare_input(sq, arch)))) for sq, _ in data] \
                    if variant != "keras" else preds
                y = [lbl for _, lbl in data]
                row["accuracy_%"] = round(100 * np.mean(np.array(preds) == np.array(y)), 2)
                row["agree_with_keras_%"] = round(100 * np.mean(np.array(preds) == np.array(ref)), 2)
            rows.append(row)
            print(f"   {label:<20} threads={row['threads']!s:<4} size={size:7.1f} MB  "
                  f"median={r['median_ms']:7.1f} ms  p95={r['p95_ms']:7.1f} ms  ~{r['fps']:.1f} img/s"
                  + (f"  acc={row['accuracy_%']}%" if data is not None else ""))
        print()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
