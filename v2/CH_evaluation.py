"""
CH_evaluation.py
================
Loads self-supervised embeddings from CH_ssl_embedder.py and produces:

  1. Separation metrics table — all 5 embedding dimensions, reported separately
     for training split and out-of-sample (held-out) split.

  2. Inference comparison — ESD centroid classifier vs 6 baselines over
     10 repeated stratified 80/20 splits:
       ESD          : nearest-centroid in embedding space (no extra training)
       Emb-kNN      : k-NN (k=5) on embeddings
       Emb-LR       : Logistic Regression (L2) on embeddings
       Emb-SVM      : RBF-kernel SVM on embeddings
       OF-kNN       : k-NN on raw (original) features
       OF-LR        : Logistic Regression on raw features
       OF-SVM       : RBF-kernel SVM on raw features

     Metrics: Macro-F1 and Balanced Accuracy (mean ± std across 10 splits).
     Statistical test: Wilcoxon signed-rank test, ESD vs each baseline.

  3. All tables saved as CSV and printed to console.

Usage:
    python CH_evaluation.py --data path/to/ch_data.csv --emb_dir ./ch_embeddings_ssl

  --best_dim   (int, default 128) : embedding dimension used for inference comparison.
               Run separation metrics first to choose the best dim; 128 is a reasonable default.
"""

import os, argparse, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances

SEED = 42
N_SPLITS = 10

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same column logic as embedder)
# ─────────────────────────────────────────────────────────────────────────────
def load_ch_data(path):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xls", ".xlsx") else pd.read_csv(path)
    y = df["CH3"].astype(int).values
    drop_cols = [c for c in ["CH", "CH3", "Year"] if c in df.columns]
    Xdf = df.drop(columns=drop_cols).copy()
    cat_cols = [c for c in Xdf.columns if Xdf[c].dtype == object]
    for c in ["kind", "sex"]:
        if c in Xdf.columns and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in Xdf.columns if c not in cat_cols]
    return Xdf, y, num_cols, cat_cols

def preprocess_raw_features(Xdf, num_cols, cat_cols, tr_idx):
    """Return imputed+scaled numeric features + one-hot-encoded categoricals."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xn = Xdf[num_cols].astype(float).values
    imp.fit(Xn[tr_idx]); scaler.fit(imp.transform(Xn[tr_idx]))
    Xn_all = scaler.transform(imp.transform(Xn))
    if cat_cols:
        Xc = pd.get_dummies(Xdf[cat_cols].astype(str).fillna("__NA__")).values.astype(float)
        Xn_all = np.hstack([Xn_all, Xc])
    return Xn_all

# ─────────────────────────────────────────────────────────────────────────────
# Geometry metrics
# ─────────────────────────────────────────────────────────────────────────────
def geometry_metrics(E, y):
    cls, cnts = np.unique(y, return_counts=True)
    if len(cls) < 2 or cnts.min() < 2:
        return {k: float("nan") for k in
                ["silhouette","davies_bouldin","calinski_harabasz",
                 "within_mean_dist","between_mean_dist","separation_ratio"]}
    D = euclidean_distances(E)
    same = (y[:, None] == y[None, :]); np.fill_diagonal(same, False)
    diff = ~(y[:, None] == y[None, :])
    w = float(D[same].mean()); b = float(D[diff].mean())
    return dict(
        silhouette=float(silhouette_score(E, y)),
        davies_bouldin=float(davies_bouldin_score(E, y)),
        calinski_harabasz=float(calinski_harabasz_score(E, y)),
        within_mean_dist=w, between_mean_dist=b,
        separation_ratio=b / w if w > 0 else float("nan"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# ESD centroid inference
# ─────────────────────────────────────────────────────────────────────────────
def esd_centroid_predict(E_train, y_train, E_test):
    classes = np.unique(y_train)
    centroids = np.stack([E_train[y_train == c].mean(axis=0) for c in classes])
    dists = euclidean_distances(E_test, centroids)   # (n_test, n_classes)
    return classes[np.argmin(dists, axis=1)]

# ─────────────────────────────────────────────────────────────────────────────
# Run inference comparison over N_SPLITS repeated splits
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_comparison(E_emb, X_raw, y, n_splits=N_SPLITS):
    """
    E_emb : (N, d) pre-trained self-supervised embeddings (labels NOT used in training)
    X_raw : (N, p) imputed+scaled original features
    y     : (N,)  integer class labels

    Returns DataFrame with per-split scores for every method.
    """
    methods = {
        "ESD":     None,               # handled separately
        "Emb-kNN": KNeighborsClassifier(n_neighbors=5),
        "Emb-LR":  LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "Emb-SVM": SVC(kernel="rbf", C=1.0, random_state=SEED),
        "OF-kNN":  KNeighborsClassifier(n_neighbors=5),
        "OF-LR":   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "OF-SVM":  SVC(kernel="rbf", C=1.0, random_state=SEED),
    }

    records = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.20, random_state=SEED)

    for split_i, (tr, te) in enumerate(sss.split(E_emb, y)):
        E_tr, E_te = E_emb[tr], E_emb[te]
        X_tr, X_te = X_raw[tr], X_raw[te]
        y_tr, y_te = y[tr], y[te]

        for name, clf in methods.items():
            if name == "ESD":
                y_pred = esd_centroid_predict(E_tr, y_tr, E_te)
            elif name.startswith("Emb"):
                clf.fit(E_tr, y_tr); y_pred = clf.predict(E_te)
            else:  # OF-*
                clf.fit(X_tr, y_tr); y_pred = clf.predict(X_te)

            records.append({
                "split": split_i,
                "method": name,
                "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
                "balanced_acc": balanced_accuracy_score(y_te, y_pred),
            })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# Summarise + Wilcoxon tests
# ─────────────────────────────────────────────────────────────────────────────
def summarise_inference(df_splits):
    summary_rows = []
    esd_f1  = df_splits[df_splits.method == "ESD"]["macro_f1"].values
    esd_ba  = df_splits[df_splits.method == "ESD"]["balanced_acc"].values

    for method in ["ESD", "Emb-kNN", "Emb-LR", "Emb-SVM", "OF-kNN", "OF-LR", "OF-SVM"]:
        sub = df_splits[df_splits.method == method]
        f1_vals = sub["macro_f1"].values
        ba_vals = sub["balanced_acc"].values

        if method == "ESD":
            p_f1 = p_ba = float("nan")
        else:
            try:
                _, p_f1 = wilcoxon(esd_f1, f1_vals, alternative="two-sided")
            except Exception:
                p_f1 = float("nan")
            try:
                _, p_ba = wilcoxon(esd_ba, ba_vals, alternative="two-sided")
            except Exception:
                p_ba = float("nan")

        summary_rows.append({
            "Method":       method,
            "Macro-F1 mean": f"{f1_vals.mean():.3f}",
            "Macro-F1 std":  f"{f1_vals.std():.3f}",
            "Bal.Acc mean":  f"{ba_vals.mean():.3f}",
            "Bal.Acc std":   f"{ba_vals.std():.3f}",
            "p (F1 vs ESD)":  f"{p_f1:.3f}" if not np.isnan(p_f1) else "—",
            "p (BA vs ESD)":  f"{p_ba:.3f}" if not np.isnan(p_ba) else "—",
        })

    return pd.DataFrame(summary_rows)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CH dataset CSV/Excel")
    parser.add_argument("--emb_dir", default="./ch_embeddings_ssl",
                        help="Output directory from CH_ssl_embedder.py")
    parser.add_argument("--best_dim", type=int, default=128,
                        help="Embedding dimension to use for inference comparison")
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32, 64, 128],
                        help="Embedding dimensions to include in separation metrics table")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("CH Evaluation: Separation Metrics + Inference Comparison")
    print("="*70)

    Xdf, y, num_cols, cat_cols = load_ch_data(args.data)
    N = len(Xdf)

    # ── 1. Separation metrics across all dimensions ──────────────────────────
    print("\n[1] Separation metrics across embedding dimensions")
    print("    (labels used here only as post-hoc evaluation criterion)")

    sep_rows = []
    for d in args.dims:
        dim_dir = os.path.join(args.emb_dir, f"repr_dim_{d}")
        emb_path = os.path.join(dim_dir, "embeddings.npy")
        idx_tr   = os.path.join(dim_dir, "train_idx.npy")
        idx_va   = os.path.join(dim_dir, "val_idx.npy")

        if not os.path.exists(emb_path):
            print(f"    WARNING: embeddings not found for dim={d}, skipping.")
            continue

        E  = np.load(emb_path)
        lbl = np.load(os.path.join(dim_dir, "labels_CH3.npy"))
        tr  = np.load(idx_tr)
        va  = np.load(idx_va)

        for split_name, idx in [("train", tr), ("val (OOS)", va)]:
            m = geometry_metrics(E[idx], lbl[idx])
            row = {"dim": d, "split": split_name}
            row.update(m)
            sep_rows.append(row)

    df_sep = pd.DataFrame(sep_rows)
    print("\n" + df_sep.to_string(index=False, float_format="{:.4f}".format))

    sep_path = os.path.join(args.emb_dir, "separation_metrics.csv")
    df_sep.to_csv(sep_path, index=False)
    print(f"\n    Saved → {sep_path}")

    # ── 2. Inference comparison ───────────────────────────────────────────────
    print(f"\n[2] Inference comparison  (embedding dim={args.best_dim}, {N_SPLITS} splits)")

    dim_dir = os.path.join(args.emb_dir, f"repr_dim_{args.best_dim}")
    E_best  = np.load(os.path.join(dim_dir, "embeddings.npy"))
    tr_idx  = np.load(os.path.join(dim_dir, "train_idx.npy"))   # used to fit raw-feature scaler

    X_raw = preprocess_raw_features(Xdf, num_cols, cat_cols, tr_idx)

    print("    Running 10 repeated stratified splits...")
    df_splits = run_inference_comparison(E_best, X_raw, y)

    splits_path = os.path.join(args.emb_dir, "inference_splits.csv")
    df_splits.to_csv(splits_path, index=False)

    df_summary = summarise_inference(df_splits)

    print("\n    Summary (mean ± std over 10 splits, Wilcoxon p-values vs ESD):")
    print("\n" + df_summary.to_string(index=False))

    summary_path = os.path.join(args.emb_dir, "inference_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n    Saved → {summary_path}")

    # ── 3. Quick note on results ──────────────────────────────────────────────
    esd_ba = df_splits[df_splits.method == "ESD"]["balanced_acc"].mean()
    best_ba_method = df_summary.loc[
        df_summary["Bal.Acc mean"].astype(float).idxmax(), "Method"]
    print(f"\n    ESD balanced accuracy: {esd_ba:.3f}")
    print(f"    Best balanced accuracy: {best_ba_method}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
