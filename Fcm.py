import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import skfuzzy as fuzz

# =========================================
# 
# =========================================
IMAGE_PATHS = [
    "IMG_3677(1).png",
    "IMG_3701(1).png",
]
OUT_DIR = "fcm_results"
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTERS = range(2, 7)   # من 2 إلى 6
M = 2.0                  # fuzzy exponent
ERROR = 1e-5
MAXITER = 150
SEED = 42
RESIZE_LONG_SIDE = 280   # الضلع الأطول = 280 بكسل

# =========================================

# =========================================
def load_and_resize_image(path, long_side=280):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    scale = long_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr = np.asarray(img_resized, dtype=np.float64) / 255.0
    return img_resized, arr

# =========================================
# 
# =========================================
def partition_entropy(u):
    eps = 1e-12
    return -np.sum(u * np.log(u + eps)) / u.shape[1]

# ======================================

# =========================================
def run_fcm_rgb(image_array, c, m=2.0, error=1e-5, maxiter=150, seed=42):
    h, w, ch = image_array.shape
    X = image_array.reshape(-1, ch).T  # shape = (features, N)

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X, c=c, m=m, error=error, maxiter=maxiter, seed=seed
    )

    # PC من skfuzzy = fpc
    pc = fpc

    # PE
    pe = partition_entropy(u)

    # XB
    # u shape: (c, N), d shape: (c, N)
    num = np.sum((u ** m) * (d ** 2))
    center_dist = np.linalg.norm(cntr[:, None, :] - cntr[None, :, :], axis=2)
    center_dist[center_dist == 0] = np.inf
    min_center_dist_sq = np.min(center_dist) ** 2
    xb = num / (u.shape[1] * min_center_dist_sq + 1e-12)

    # Crisp labels
    labels = np.argmax(u, axis=0)
    segmented = cntr[labels].reshape(h, w, ch)

    return {
        "centers": cntr,
        "u": u,
        "pc": pc,
        "pe": pe,
        "xb": xb,
        "labels": labels.reshape(h, w),
        "segmented": segmented
    }

# =========================================
# 
# =========================================
def choose_best_cluster(metrics_df):
    # PC: الأكبر أفضل
    # PE: الأصغر أفضل
    # XB: الأصغر أفضل
    pc_rank = metrics_df["PC"].rank(ascending=False, method="min")
    pe_rank = metrics_df["PE"].rank(ascending=True, method="min")
    xb_rank = metrics_df["XB"].rank(ascending=True, method="min")

    metrics_df["rank_sum"] = pc_rank + pe_rank + xb_rank

    # 
    pc_norm = (metrics_df["PC"] - metrics_df["PC"].min()) / (metrics_df["PC"].max() - metrics_df["PC"].min() + 1e-12)
    pe_norm = (metrics_df["PE"].max() - metrics_df["PE"]) / (metrics_df["PE"].max() - metrics_df["PE"].min() + 1e-12)
    xb_norm = (metrics_df["XB"].max() - metrics_df["XB"]) / (metrics_df["XB"].max() - metrics_df["XB"].min() + 1e-12)

    metrics_df["score"] = (pc_norm + pe_norm + xb_norm) / 3.0

    best_row = metrics_df.sort_values(["rank_sum", "score"], ascending=[True, False]).iloc[0]
    best_c = int(best_row["c"])
    return best_c, metrics_df

# =========================================
# 
# =========================================
def save_metrics_plot(metrics_df, out_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["c"], metrics_df["PC"], marker="o", label="PC (higher is better)")
    plt.plot(metrics_df["c"], metrics_df["PE"], marker="s", label="PE (lower is better)")
    plt.plot(metrics_df["c"], metrics_df["XB"], marker="^", label="XB (lower is better)")
    plt.xlabel("Number of clusters (c)")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =========================================
# 
# =========================================
def save_comparison(original_arr, segmented_arr, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original_arr)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(np.clip(segmented_arr, 0, 1))
    axes[1].set_title("Segmented Image")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =========================================
# 
# =========================================
def process_image(path):
    base = os.path.splitext(os.path.basename(path))[0]

    img_pil, img_arr = load_and_resize_image(path, long_side=RESIZE_LONG_SIDE)

    all_metrics = []
    all_results = {}

    for c in CLUSTERS:
        result = run_fcm_rgb(img_arr, c, m=M, error=ERROR, maxiter=MAXITER, seed=SEED)
        all_results[c] = result
        all_metrics.append({
            "c": c,
            "PC": result["pc"],
            "PE": result["pe"],
            "XB": result["xb"]
        })

    metrics_df = pd.DataFrame(all_metrics)
    best_c, metrics_df = choose_best_cluster(metrics_df)

    # 
    metrics_csv = os.path.join(OUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # 
    plot_path = os.path.join(OUT_DIR, f"{base}_plot.png")
    save_metrics_plot(metrics_df, plot_path, f"Cluster Validity Metrics - {base}")

    # 
    comparison_path = os.path.join(OUT_DIR, f"{base}_comparison.png")
    best_segmented = all_results[best_c]["segmented"]
    save_comparison(img_arr, best_segmented, comparison_path, f"Best c = {best_c}")

    print("=" * 60)
    print(f"Image: {base}")
    print(f"Best cluster = {best_c}")
    print(metrics_df)
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {comparison_path}")

for image_path in IMAGE_PATHS:
    process_image(image_path)
