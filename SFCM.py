import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter

# =========================================================
# 
# =========================================================
IMAGE_PATHS = [
    "IMG_3677(1).png",  # peacock
     "IMG_3701(1).png",   # tower
]

OUT_DIR = "spatial_fcm_noise_results"
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_RANGE = range(2, 7)   # 2..6
M = 2.0
MAX_ITER = 100
ERROR = 1e-5
SEED = 42

# Spatial parameters
WINDOW_SIZE = 5
ALPHA = 1.8   

# Image size
RESIZE_LONG_SIDE = 320  

# Feature weights
COLOR_WEIGHT = 1.0
XY_WEIGHT = 0.35  

# اختيار أفضل c
USE_PENALTY_FOR_SMALL_C = True
C2_PENALTY = 0.08   


# =========================================================

# =========================================================
def load_image(path, resize_long_side=None):
    img = Image.open(path).convert("RGB")

    if resize_long_side is not None:
        w, h = img.size
        scale = resize_long_side / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float64) / 255.0
    return img, arr


# =========================================================

# =========================================================
def rgb_to_gray(image_rgb):
    return (
        0.2989 * image_rgb[..., 0] +
        0.5870 * image_rgb[..., 1] +
        0.1140 * image_rgb[..., 2]
    )

def estimate_noise(image_rgb):
    """
    تقدير noise بشكل عملي:
    1) الفرق بين الصورة الأصلية و median filtered image
    2) MAD-based sigma estimate
    3) SNR estimate
    """
    gray = rgb_to_gray(image_rgb)

   
    smooth = median_filter(gray, size=3)
    residual = gray - smooth

    # Robust sigma via MAD
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma_est = 1.4826 * mad

    # Signal power / noise power
    signal_std = np.std(gray)
    noise_std = np.std(residual) + 1e-12
    snr_db = 20 * np.log10((signal_std + 1e-12) / noise_std)

    
    local_mean = uniform_filter(gray, size=5)
    local_mean_sq = uniform_filter(gray**2, size=5)
    local_var = np.mean(np.maximum(local_mean_sq - local_mean**2, 0))

    return {
        "noise_sigma": float(sigma_est),
        "noise_std_residual": float(noise_std),
        "signal_std": float(signal_std),
        "snr_db": float(snr_db),
        "local_variance_mean": float(local_var),
        "residual_map": residual
    }


# =========================================================

# =========================================================
def build_feature_matrix(image_rgb, color_weight=1.0, xy_weight=0.35):
    h, w, ch = image_rgb.shape

    # RGB
    color_feat = image_rgb.reshape(-1, 3) * color_weight

    # Spatial coordinates normalized
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float64) / max(w - 1, 1)
    yy = yy.astype(np.float64) / max(h - 1, 1)
    xy_feat = np.stack([xx, yy], axis=-1).reshape(-1, 2) * xy_weight

    X = np.concatenate([color_feat, xy_feat], axis=1)  # shape (N,5)
    return X


# =========================================================
#
# =========================================================
def initialize_membership(c, n, seed=42):
    rng = np.random.default_rng(seed)
    U = rng.random((c, n))
    U /= np.sum(U, axis=0, keepdims=True)
    return U


# =========================================================
# 
# =========================================================
def update_centers(X, U, m):
    um = U ** m
    numerator = um @ X
    denominator = np.sum(um, axis=1, keepdims=True)
    return numerator / (denominator + 1e-12)


# =========================================================
# 
# =========================================================
def compute_distances(X, centers):
    diff = X[None, :, :] - centers[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.maximum(dist, 1e-10)


# =========================================================
# 
# =========================================================
def update_membership_from_dist(dist, m):
    power = 2.0 / (m - 1.0)
    ratio = (dist[:, None, :] / dist[None, :, :]) ** power
    U = 1.0 / np.sum(ratio, axis=1)
    return U


# =========================================================
# 
# =========================================================
def spatial_smooth_memberships(membership_maps, window_size=5):
    c, h, w = membership_maps.shape
    S = np.zeros_like(membership_maps)
    for k in range(c):
        S[k] = uniform_filter(membership_maps[k], size=window_size, mode="reflect")
    return np.maximum(S, 1e-10)


# =========================================================

# =========================================================
def spatial_fcm_with_xy(
    image_rgb,
    c,
    m=2.0,
    alpha=1.8,
    window_size=5,
    max_iter=100,
    error=1e-5,
    seed=42,
    color_weight=1.0,
    xy_weight=0.35
):
    h, w, _ = image_rgb.shape
    X = build_feature_matrix(image_rgb, color_weight=color_weight, xy_weight=xy_weight)
    n = X.shape[0]

    U = initialize_membership(c, n, seed=seed)
    objective_history = []

    for _ in range(max_iter):
        U_old = U.copy()

        centers = update_centers(X, U, m)
        dist = compute_distances(X, centers)

        U_fcm = update_membership_from_dist(dist, m)

        membership_maps = U_fcm.reshape(c, h, w)
        S = spatial_smooth_memberships(membership_maps, window_size=window_size)

        
        U_new = U_fcm * (S.reshape(c, -1) ** alpha)
        U_new /= np.sum(U_new, axis=0, keepdims=True)

        U = U_new

        jm = np.sum((U ** m) * (dist ** 2))
        objective_history.append(jm)

        if np.linalg.norm(U - U_old) < error:
            break

    centers = update_centers(X, U, m)
    dist = compute_distances(X, centers)

    labels = np.argmax(U, axis=0)
    label_image = labels.reshape(h, w)

    
    rgb_centers = centers[:, :3]
    segmented_rgb = rgb_centers[labels].reshape(h, w, 3)

    return {
        "X": X,
        "centers": centers,
        "rgb_centers": rgb_centers,
        "U": U,
        "dist": dist,
        "labels": label_image,
        "segmented": segmented_rgb,
        "objective_history": objective_history,
        "shape": (h, w, 3)
    }


# =========================================================
# Cluster Validity
# =========================================================
def partition_coefficient(U):
    n = U.shape[1]
    return np.sum(U ** 2) / n

def partition_entropy(U):
    n = U.shape[1]
    eps = 1e-12
    return -np.sum(U * np.log(U + eps)) / n

def xie_beni(U, dist, centers, m):
    n = U.shape[1]
    numerator = np.sum((U ** m) * (dist ** 2))

    center_diff = centers[:, None, :] - centers[None, :, :]
    center_dist = np.linalg.norm(center_diff, axis=2)
    center_dist[center_dist == 0] = np.inf
    min_center_dist_sq = np.min(center_dist) ** 2

    return numerator / (n * min_center_dist_sq + 1e-12)


# =========================================================

# =========================================================
def choose_best_cluster(metrics_df, use_penalty=True, c2_penalty=0.08):
    df = metrics_df.copy()

    # Normalize
    df["PC_norm"] = (df["PC"] - df["PC"].min()) / (df["PC"].max() - df["PC"].min() + 1e-12)
    df["PE_norm_good"] = (df["PE"].max() - df["PE"]) / (df["PE"].max() - df["PE"].min() + 1e-12)
    df["XB_norm_good"] = (df["XB"].max() - df["XB"]) / (df["XB"].max() - df["XB"].min() + 1e-12)

    
    df["score_raw"] = (
        0.30 * df["PC_norm"] +
        0.30 * df["PE_norm_good"] +
        0.40 * df["XB_norm_good"]
    )

   
    df["penalty"] = 0.0
    if use_penalty:
        df.loc[df["c"] == 2, "penalty"] = c2_penalty

    df["score_final"] = df["score_raw"] - df["penalty"]

    
    df["rank_PC"] = df["PC"].rank(ascending=False, method="min")
    df["rank_PE"] = df["PE"].rank(ascending=True, method="min")
    df["rank_XB"] = df["XB"].rank(ascending=True, method="min")
    df["rank_sum"] = df["rank_PC"] + df["rank_PE"] + df["rank_XB"]

    best_row = df.sort_values(["score_final", "rank_sum"], ascending=[False, True]).iloc[0]
    best_c = int(best_row["c"])

    return best_c, df


# =========================================================

# =========================================================
def save_comparison_figure(original, segmented, label_img, out_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(segmented, 0, 1))
    axes[1].set_title("Segmented RGB")
    axes[1].axis("off")

    axes[2].imshow(label_img, cmap="tab20")
    axes[2].set_title("Segmented Labels")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_metrics_plot(metrics_df, best_c, out_path, title):
    plt.figure(figsize=(9, 5))
    plt.plot(metrics_df["c"], metrics_df["PC"], marker="o", linewidth=2, label="PC")
    plt.plot(metrics_df["c"], metrics_df["PE"], marker="s", linewidth=2, label="PE")
    plt.plot(metrics_df["c"], metrics_df["XB"], marker="^", linewidth=2, label="XB")
    plt.axvline(best_c, linestyle="--", linewidth=1.5, label=f"Best c = {best_c}")
    plt.xlabel("Number of clusters (c)")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_membership_maps(U, shape, out_path, title):
    h, w, _ = shape
    c = U.shape[0]

    fig, axes = plt.subplots(1, c, figsize=(4 * c, 4))
    if c == 1:
        axes = [axes]

    for i in range(c):
        heatmap = U[i].reshape(h, w)
        axes[i].imshow(heatmap, cmap="jet")
        axes[i].set_title(f"Cluster {i+1}")
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_noise_figure(image_rgb, residual_map, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(residual_map, cmap="gray")
    axes[1].set_title("Estimated Noise Residual")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# =========================================================

# =========================================================
def process_one_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing: {base_name}")

    _, image_rgb = load_image(image_path, resize_long_side=RESIZE_LONG_SIDE)

    # ---------- Noise estimation ----------
    noise_info = estimate_noise(image_rgb)

    noise_fig_path = os.path.join(OUT_DIR, f"{base_name}_noise.png")
    save_noise_figure(
        image_rgb,
        noise_info["residual_map"],
        noise_fig_path,
        f"{base_name} - Estimated Noise"
    )

    # ---------- Spatial FCM ----------
    metrics_records = []
    results_by_c = {}

    for c in CLUSTER_RANGE:
        print(f"  Running SFCM with c = {c} ...")

        result = spatial_fcm_with_xy(
            image_rgb,
            c=c,
            m=M,
            alpha=ALPHA,
            window_size=WINDOW_SIZE,
            max_iter=MAX_ITER,
            error=ERROR,
            seed=SEED,
            color_weight=COLOR_WEIGHT,
            xy_weight=XY_WEIGHT
        )

        U = result["U"]
        dist = result["dist"]
        centers = result["centers"]

        pc = partition_coefficient(U)
        pe = partition_entropy(U)
        xb = xie_beni(U, dist, centers, M)

        metrics_records.append({
            "c": c,
            "PC": pc,
            "PE": pe,
            "XB": xb
        })

        results_by_c[c] = result

    metrics_df = pd.DataFrame(metrics_records)
    best_c, metrics_df = choose_best_cluster(
        metrics_df,
        use_penalty=USE_PENALTY_FOR_SMALL_C,
        c2_penalty=C2_PENALTY
    )

    best_result = results_by_c[best_c]

    # ---------- Save outputs ----------
    csv_path = os.path.join(OUT_DIR, f"{base_name}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    comparison_path = os.path.join(OUT_DIR, f"{base_name}_comparison.png")
    save_comparison_figure(
        image_rgb,
        best_result["segmented"],
        best_result["labels"],
        comparison_path,
        f"{base_name} - Best c = {best_c}"
    )

    plot_path = os.path.join(OUT_DIR, f"{base_name}_metrics_plot.png")
    save_metrics_plot(
        metrics_df,
        best_c,
        plot_path,
        f"Spatial FCM Validity Metrics - {base_name}"
    )

    membership_path = os.path.join(OUT_DIR, f"{base_name}_membership_maps.png")
    save_membership_maps(
        best_result["U"],
        best_result["shape"],
        membership_path,
        f"{base_name} - Membership Maps (c={best_c})"
    )

    segmented_path = os.path.join(OUT_DIR, f"{base_name}_segmented.png")
    segmented_uint8 = (np.clip(best_result["segmented"], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(segmented_uint8).save(segmented_path)

    # ---------- Summary ----------
    best_row = metrics_df.loc[metrics_df["c"] == best_c].iloc[0]

    summary_path = os.path.join(OUT_DIR, f"{base_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Image: {base_name}\n")
        f.write(f"Best number of clusters: {best_c}\n")
        f.write(f"PC = {best_row['PC']:.6f}\n")
        f.write(f"PE = {best_row['PE']:.6f}\n")
        f.write(f"XB = {best_row['XB']:.6f}\n")
        f.write(f"score_raw = {best_row['score_raw']:.6f}\n")
        f.write(f"score_final = {best_row['score_final']:.6f}\n")
        f.write("\nNoise estimation:\n")
        f.write(f"noise_sigma = {noise_info['noise_sigma']:.6f}\n")
        f.write(f"noise_std_residual = {noise_info['noise_std_residual']:.6f}\n")
        f.write(f"signal_std = {noise_info['signal_std']:.6f}\n")
        f.write(f"snr_db = {noise_info['snr_db']:.6f}\n")
        f.write(f"local_variance_mean = {noise_info['local_variance_mean']:.6f}\n")
        f.write("\nParameters:\n")
        f.write(f"window_size = {WINDOW_SIZE}\n")
        f.write(f"alpha = {ALPHA}\n")
        f.write(f"xy_weight = {XY_WEIGHT}\n")
        f.write(f"resize_long_side = {RESIZE_LONG_SIDE}\n")

    print(f"  Best c = {best_c}")
    print(metrics_df[[
        "c", "PC", "PE", "XB",
        "score_raw", "penalty", "score_final", "rank_sum"
    ]])

    return {
        "image_name": base_name,
        "best_c": best_c,
        "metrics_df": metrics_df,
        "noise_info": noise_info,
        "files": {
            "csv": csv_path,
            "comparison": comparison_path,
            "plot": plot_path,
            "membership": membership_path,
            "segmented": segmented_path,
            "noise_fig": noise_fig_path,
            "summary": summary_path
        }
    }


# =========================================================
# Main
# =========================================================
def main():
    final_rows = []

    for image_path in IMAGE_PATHS:
        result = process_one_image(image_path)

        base = result["image_name"]
        noise = result["noise_info"]

        final_rows.append({
            "image_name": base,
            "best_c": result["best_c"],
            "noise_sigma": noise["noise_sigma"],
            "noise_std_residual": noise["noise_std_residual"],
            "snr_db": noise["snr_db"],
            "local_variance_mean": noise["local_variance_mean"]
        })

    final_df = pd.DataFrame(final_rows)
    final_csv = os.path.join(OUT_DIR, "final_summary_with_noise.csv")
    final_df.to_csv(final_csv, index=False)

    print("\nFinished.")
    print(final_df)
    print(f"\nSaved final summary to: {final_csv}")
    print(f"All outputs are in: {OUT_DIR}")


if __name__ == "__main__":
    main()
