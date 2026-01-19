import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances


def label_separation_metrics(
    embeddings_path: str,
    labels_path: str,
    metric: str = "euclidean",     # distance metric for pairwise distances + silhouette
    drop_labels: tuple = (),       # e.g. (-1,) if you ever have unknown labels
    min_class_size: int = 2,       # silhouette needs >=2 samples per class
) -> dict:
    """
    Quantify how well label groups (e.g., CH3=0/1/2) are separated in the ORIGINAL
    embedding space (N x d), without any 2D projection and without clustering.

    Returns:
      - silhouette (higher is better, range [-1, 1])
      - davies_bouldin (lower is better)
      - calinski_harabasz (higher is better)
      - within_mean_dist: mean pairwise distance within same-label groups
      - between_mean_dist: mean pairwise distance across different-label groups
      - separation_ratio = between_mean_dist / within_mean_dist (higher is better)
      - class_counts
    """
    X = np.load(embeddings_path)
    y = np.load(labels_path)

    # Flatten possible (N,1)
    X = np.asarray(X)
    y = np.ravel(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: embeddings N={X.shape[0]} vs labels N={y.shape[0]}")

    # Drop unwanted labels if needed
    if drop_labels:
        mask = np.ones_like(y, dtype=bool)
        for lab in drop_labels:
            mask &= (y != lab)
        X = X[mask]
        y = y[mask]

    # Ensure at least 2 classes remain
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("Need at least 2 label classes to compute separation metrics.")

    # Optionally filter tiny classes (silhouette requires >=2 per class)
    keep_classes = classes[counts >= min_class_size]
    if keep_classes.size < 2:
        raise ValueError(
            f"After filtering classes with size < {min_class_size}, fewer than 2 classes remain."
        )

    keep_mask = np.isin(y, keep_classes)
    X = X[keep_mask]
    y = y[keep_mask]

    # Recompute counts after filtering
    classes, counts = np.unique(y, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(classes, counts)}

    # --- Core separation metrics (original space) ---
    # Silhouette (higher better): uses distances implicitly
    sil = silhouette_score(X, y, metric=metric)

    # Davies–Bouldin (lower better): uses centroid dispersion/separation
    db = davies_bouldin_score(X, y)

    # Calinski–Harabasz (higher better): ratio of between/within dispersion
    ch = calinski_harabasz_score(X, y)

    # --- Simple interpretable distance-based summary ---
    D = pairwise_distances(X, metric=metric)
    same = (y[:, None] == y[None, :])
    diff = ~same

    # Exclude diagonal from within distances
    np.fill_diagonal(same, False)

    within_mean = float(D[same].mean()) if same.any() else np.nan
    between_mean = float(D[diff].mean()) if diff.any() else np.nan
    sep_ratio = float(between_mean / within_mean) if within_mean and within_mean > 0 else np.nan

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "within_mean_dist": within_mean,
        "between_mean_dist": between_mean,
        "separation_ratio": sep_ratio,
        "class_counts": class_counts,
        "metric": metric,
    }


def metrics_to_table(results: dict) -> pd.DataFrame:
    """Convenience: format results as a 1-row DataFrame (easy to print/save)."""
    flat = results.copy()
    # Keep class_counts as a string for tables
    flat["class_counts"] = str(flat["class_counts"])
    return pd.DataFrame([flat])


# ---------------- Example usage ----------------
embeddings = "./embeddings/repr_dim_128/final_full_embeddings.npy"
labels     = "./embeddings/repr_dim_128/labels_CH3.npy"
res = label_separation_metrics(embeddings, labels, metric="euclidean", min_class_size=2)
#print(res)
print(metrics_to_table(res).to_string(index=False))
