"""
FBC_analysis.py
===============
Cohort discovery and clinical characterisation for the FBC dataset.

Produces ALL tables and figures needed for the paper:

  1. Overall disease prevalence (π_c) for all 20 conditions — printed and saved.
  2. DBSCAN clustering in embedding space → membership confidence (ρ) and
     enrichment ratio (λ = ρ / π_c) at thresholds 0.80, 0.90, 1.00.
  3. Comparison: DBSCAN on raw (original) features vs embedding space,
     same parameters, same condition evaluation.
  4. Top enriched subspaces: ranked by λ for each condition.
  5. All results saved as CSV and LaTeX-formatted tables.

Usage:
    python FBC_analysis.py --data path/to/fbc_data.csv --emb_dir ./fbc_embeddings

The script reads column_map.json from emb_dir to resolve column names.
If column_map.json is missing, it uses the same COLUMN_MAP fallback as FBC_ssl_embedder.py.

DBSCAN parameters (default: eps=0.5, min_samples=5) can be changed via --eps / --min_samples.
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Column map fallback (same as FBC_ssl_embedder.py)
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "RBC":  ["RBC", "rbc", "red_blood_cells"],
    "WBC":  ["WBC", "wbc", "white_blood_cells", "Leukocytes"],
    "NEU":  ["NEU", "neu", "neutrophils", "Neutrophils", "NEUT"],
    "LYM":  ["LYM", "lym", "lymphocytes", "Lymphocytes", "LYMPH"],
    "EOS":  ["EOS", "eos", "eosinophils", "Eosinophils"],
    "MON":  ["MON", "mon", "monocytes", "Monocytes"],
    "BAS":  ["BAS", "bas", "basophils"],
    "HGB":  ["HGB", "hgb", "haemoglobin", "hemoglobin", "Hemoglobin", "HB"],
    "HCT":  ["HCT", "hct", "haematocrit", "hematocrit"],
    "PLT":  ["PLT", "plt", "platelets", "Platelets"],
    "MCV":  ["MCV", "mcv"],
    "MCH":  ["MCH", "mch"],
    "MCHC": ["MCHC", "mchc"],
    "MPV":  ["MPV", "mpv"],
    "RDW":  ["RDW", "rdw"],
    "FER":  ["FER", "fer", "ferritin", "Ferritin"],
    "B12":  ["B12", "b12", "vitamin_b12", "VitB12"],
    "FOL":  ["FOL", "fol", "folate"],
    "SEX":  ["SEX", "sex", "Sex", "gender"],
}

def resolve_columns(df, col_map):
    resolved = {}
    for canon, candidates in col_map.items():
        for cand in candidates:
            if cand in df.columns:
                resolved[canon] = cand
                break
    return resolved

# ─────────────────────────────────────────────────────────────────────────────
# Disease definitions
# All 20 conditions from the paper's Table.
# Each function takes a DataFrame row (Series) and the resolved column map,
# and returns True if the patient meets the criteria.
# Sex-specific thresholds: if SEX column is absent, we use sex-agnostic values
# (the more conservative threshold, i.e. the one that flags more cases).
# ─────────────────────────────────────────────────────────────────────────────
def get_sex(row, col, default="unknown"):
    if col is None:
        return default
    v = str(row[col]).strip().upper()
    if v in ("1", "M", "MALE"):   return "M"
    if v in ("0", "F", "FEMALE"): return "F"
    return default

def is_female(row, sex_col):
    return get_sex(row, sex_col) == "F"

def hgb_low(row, col, sex_col):
    """Hemoglobin below sex-specific threshold (or 12 if sex unknown)."""
    if col is None: return False
    v = row[col]
    if pd.isna(v): return False
    thresh = 12.0 if is_female(row, sex_col) or get_sex(row, sex_col) == "unknown" else 13.0
    return float(v) < thresh

def hgb_high(row, col, sex_col):
    if col is None: return False
    v = row[col]
    if pd.isna(v): return False
    thresh = 16.5 if is_female(row, sex_col) or get_sex(row, sex_col) == "unknown" else 18.5
    return float(v) > thresh

def hct_low(row, col, sex_col):
    if col is None: return False
    v = row[col]
    if pd.isna(v): return False
    thresh = 36.0 if is_female(row, sex_col) or get_sex(row, sex_col) == "unknown" else 39.0
    return float(v) < thresh

def _safe(row, col, op, val):
    if col is None: return False
    v = row.get(col, None) if isinstance(row, dict) else (row[col] if col in row.index else None)
    if v is None or pd.isna(v): return False
    return op(float(v), val)

# Vectorised helpers for whole-DataFrame evaluation (much faster than row-by-row)
def col_lt(df, col, val):
    if col is None or col not in df.columns: return pd.Series(False, index=df.index)
    return df[col].astype(float) < val

def col_gt(df, col, val):
    if col is None or col not in df.columns: return pd.Series(False, index=df.index)
    return df[col].astype(float) > val

def sex_thresh(df, sex_col, val_f, val_m, default_val=None):
    """Returns a Series of per-row thresholds based on sex column."""
    if default_val is None: default_val = val_f
    if sex_col is None or sex_col not in df.columns:
        return pd.Series(default_val, index=df.index)
    s = df[sex_col].astype(str).str.strip().str.upper()
    out = pd.Series(default_val, index=df.index)
    out[s.isin(["F", "0", "FEMALE"])] = val_f
    out[s.isin(["M", "1", "MALE"])]   = val_m
    return out

def hgb_below(df, hgb_col, sex_col, val_f=12.0, val_m=13.0):
    if hgb_col is None or hgb_col not in df.columns:
        return pd.Series(False, index=df.index)
    thresh = sex_thresh(df, sex_col, val_f, val_m, default_val=12.0)
    return df[hgb_col].astype(float) < thresh

def hgb_above(df, hgb_col, sex_col, val_f=16.5, val_m=18.5):
    if hgb_col is None or hgb_col not in df.columns:
        return pd.Series(False, index=df.index)
    thresh = sex_thresh(df, sex_col, val_f, val_m, default_val=16.5)
    return df[hgb_col].astype(float) > thresh

def hct_below(df, hct_col, sex_col, val_f=36.0, val_m=39.0):
    if hct_col is None or hct_col not in df.columns:
        return pd.Series(False, index=df.index)
    thresh = sex_thresh(df, sex_col, val_f, val_m, default_val=36.0)
    return df[hct_col].astype(float) < thresh

def rbc_below(df, rbc_col, sex_col, val_f=4.2, val_m=4.7):
    if rbc_col is None or rbc_col not in df.columns:
        return pd.Series(False, index=df.index)
    thresh = sex_thresh(df, sex_col, val_f, val_m, default_val=4.2)
    return df[rbc_col].astype(float) < thresh

def fer_above_sex(df, fer_col, sex_col, val_f=200.0, val_m=300.0):
    if fer_col is None or fer_col not in df.columns:
        return pd.Series(False, index=df.index)
    thresh = sex_thresh(df, sex_col, val_f, val_m, default_val=200.0)
    return df[fer_col].astype(float) > thresh

def build_condition_masks(df: pd.DataFrame, col: dict) -> dict:
    """
    Build boolean Series for all 20 conditions.
    col: resolved column map (canonical → actual column name, or None if absent)
    """
    def C(canon):
        return col.get(canon, None)

    masks = {}

    # 1 Anemia
    masks[1]  = hgb_below(df, C("HGB"), C("SEX")) | hct_below(df, C("HCT"), C("SEX"))

    # 2 Inflammation
    masks[2]  = col_gt(df, C("WBC"), 11.0) & col_gt(df, C("NEU"), 7.0)

    # 3 Iron Deficiency Anemia
    masks[3]  = (hgb_below(df, C("HGB"), C("SEX"))
                 & col_lt(df, C("FER"), 15.0)
                 & col_lt(df, C("MCV"), 80.0)
                 & col_gt(df, C("RDW"), 14.5))

    # 4 Vitamin B12 Deficiency Anemia
    masks[4]  = (hgb_below(df, C("HGB"), C("SEX"))
                 & col_lt(df, C("B12"), 200.0)
                 & col_gt(df, C("MCV"), 100.0)
                 & col_gt(df, C("RDW"), 14.5))

    # 5 Folate Deficiency Anemia
    masks[5]  = (hgb_below(df, C("HGB"), C("SEX"))
                 & col_lt(df, C("FOL"), 3.0)
                 & col_gt(df, C("MCV"), 100.0)
                 & col_gt(df, C("RDW"), 14.5))

    # 6 Anemia of Chronic Disease
    masks[6]  = hgb_below(df, C("HGB"), C("SEX")) & col_gt(df, C("FER"), 100.0)

    # 7 Polycythemia
    masks[7]  = hgb_above(df, C("HGB"), C("SEX")) & col_gt(df, C("HCT"), 48.0)

    # 8 Leukemia (extreme WBC, NEU, LYM, PLT abnormalities)
    masks[8]  = ((col_lt(df, C("WBC"), 4.0) | col_gt(df, C("WBC"), 100.0))
                 & (col_lt(df, C("NEU"), 1.5) | col_gt(df, C("NEU"), 7.0))
                 & (col_lt(df, C("LYM"), 1.0) | col_gt(df, C("LYM"), 5.0))
                 & (col_lt(df, C("PLT"), 150.0) | col_gt(df, C("PLT"), 450.0)))

    # 9 Viral Infections
    masks[9]  = (col_lt(df, C("WBC"), 4.0)
                 & col_lt(df, C("NEU"), 1.5)
                 & (col_lt(df, C("LYM"), 1.0) | col_gt(df, C("LYM"), 3.0)))

    # 10 Bacterial Infections
    masks[10] = col_gt(df, C("WBC"), 11.0) & col_gt(df, C("NEU"), 7.0)

    # 11 Parasitic Infections
    masks[11] = col_gt(df, C("EOS"), 0.5)

    # 12 Allergies
    masks[12] = col_gt(df, C("WBC"), 11.0) & col_gt(df, C("EOS"), 0.5)

    # 13 Thrombocytopenia
    masks[13] = col_lt(df, C("PLT"), 150.0)

    # 14 Thrombocytosis
    masks[14] = col_gt(df, C("PLT"), 450.0) & col_gt(df, C("MPV"), 12.0)

    # 15 Iron Overload (Hemochromatosis)
    masks[15] = fer_above_sex(df, C("FER"), C("SEX")) & hgb_above(df, C("HGB"), C("SEX"))

    # 16 Chronic Kidney Disease
    masks[16] = hgb_below(df, C("HGB"), C("SEX")) & col_gt(df, C("FER"), 100.0)

    # 17 Aplastic Anemia
    masks[17] = (col_lt(df, C("WBC"), 4.0)
                 & col_lt(df, C("PLT"), 150.0)
                 & rbc_below(df, C("RBC"), C("SEX")))

    # 18 Myeloproliferative Disorders (using 160 g/L = 16.0 g/dL)
    masks[18] = col_gt(df, C("PLT"), 450.0) & col_gt(df, C("HGB"), 16.0)

    # 19 Sickle Cell Disease
    masks[19] = (hgb_below(df, C("HGB"), C("SEX"))
                 & col_lt(df, C("MCV"), 80.0)
                 & col_gt(df, C("RDW"), 14.5))

    # 20 Sepsis
    masks[20] = ((col_lt(df, C("WBC"), 4.0) | col_gt(df, C("WBC"), 11.0))
                 & col_lt(df, C("PLT"), 150.0)
                 & col_gt(df, C("NEU"), 7.0))

    # Convert all to boolean numpy arrays
    for k in masks:
        masks[k] = masks[k].fillna(False).astype(bool).values

    return masks

DISEASE_NAMES = {
    1: "Anemia", 2: "Inflammation", 3: "Iron Deficiency Anemia",
    4: "Vit B12 Deficiency Anemia", 5: "Folate Deficiency Anemia",
    6: "Anemia of Chronic Disease", 7: "Polycythemia", 8: "Leukemia",
    9: "Viral Infections", 10: "Bacterial Infections", 11: "Parasitic Infections",
    12: "Allergies", 13: "Thrombocytopenia", 14: "Thrombocytosis",
    15: "Iron Overload (Hemochromatosis)", 16: "Chronic Kidney Disease",
    17: "Aplastic Anemia", 18: "Myeloproliferative Disorders",
    19: "Sickle Cell Disease", 20: "Sepsis",
}

# ─────────────────────────────────────────────────────────────────────────────
# Membership analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_clusters(labels: np.ndarray, cond_masks: dict, N: int):
    """
    For each cluster (excluding noise=-1), compute:
      - size
      - rho  (membership confidence)  for each condition
      - lambda (enrichment ratio)      for each condition
    Also computes overall prevalence pi_c for each condition.

    Returns:
      pi_c   : dict {condition_id: float}
      records: list of dicts (one per cluster × condition)
    """
    pi_c = {k: float(m.mean()) for k, m in cond_masks.items()}
    cluster_ids = sorted(set(labels) - {-1})
    records = []

    for cid in cluster_ids:
        members = (labels == cid)
        size = int(members.sum())
        for k, mask in cond_masks.items():
            rho = float(mask[members].mean()) if size > 0 else 0.0
            pi  = pi_c[k]
            lam = (rho / pi) if pi > 1e-9 else float("nan")
            records.append({"cluster": cid, "size": size,
                             "condition": k, "condition_name": DISEASE_NAMES[k],
                             "rho": rho, "pi_c": pi, "lambda": lam})

    return pi_c, records

def prevalence_table(pi_c: dict) -> pd.DataFrame:
    rows = [{"Tag": k, "Disease": DISEASE_NAMES[k],
             "Prevalence (pi_c)": f"{pi_c[k]*100:.2f}%",
             "pi_c (fraction)": round(pi_c[k], 4)}
            for k in sorted(pi_c)]
    return pd.DataFrame(rows)

def threshold_table(records, threshold: float) -> pd.DataFrame:
    df = pd.DataFrame(records)
    filt = df[df["rho"] >= threshold].copy()
    filt = filt.sort_values(["cluster", "condition"])
    return filt[["cluster", "size", "condition", "condition_name",
                 "rho", "pi_c", "lambda"]].reset_index(drop=True)

def top_enriched(records, top_n=10) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[df["rho"] >= 0.80].copy()
    df = df.sort_values("lambda", ascending=False).head(top_n)
    return df[["cluster", "size", "condition_name", "rho", "pi_c", "lambda"]].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table helpers
# ─────────────────────────────────────────────────────────────────────────────
def membership_to_latex(threshold_df: pd.DataFrame, threshold: float, out_path: str):
    if threshold_df.empty:
        return
    lines = [
        r"\begin{longtable}{cclrrr}",
        r"\hline",
        r"\textbf{Subspace} & \textbf{Size} & \textbf{Condition} & "
        r"$\boldsymbol{\rho}$ & $\boldsymbol{\pi_c}$ & $\boldsymbol{\lambda}$ \\",
        r"\hline \endhead",
    ]
    for _, row in threshold_df.iterrows():
        lines.append(
            f"\\#{int(row['cluster'])} & {int(row['size'])} & "
            f"{row['condition_name']} & "
            f"{row['rho']:.2f} & {row['pi_c']:.3f} & {row['lambda']:.1f} \\\\")
    lines += [r"\hline",
              rf"\caption{{Subspace–condition associations at $\rho \geq {threshold}$}}",
              r"\end{longtable}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot: embedding-space vs raw-feature clustering
# ─────────────────────────────────────────────────────────────────────────────
def comparison_summary(rec_emb, rec_raw, threshold=0.90) -> pd.DataFrame:
    """Compare number of high-confidence associations at given threshold."""
    def count_hits(records, thr):
        df = pd.DataFrame(records)
        return int((df["rho"] >= thr).sum())

    def mean_rho(records):
        df = pd.DataFrame(records)
        return df["rho"].mean()

    def n_clusters(labels):
        return len(set(labels) - {-1})

    return pd.DataFrame([
        {"Space": "Embedding",    "n_subspaces": n_clusters([r["cluster"] for r in rec_emb]),
         f"Associations >= rho {threshold}": count_hits(rec_emb, threshold),
         "Mean rho (all)": round(mean_rho(rec_emb), 4)},
        {"Space": "Raw features", "n_subspaces": n_clusters([r["cluster"] for r in rec_raw]),
         f"Associations >= rho {threshold}": count_hits(rec_raw, threshold),
         "Mean rho (all)": round(mean_rho(rec_raw), 4)},
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", default="./fbc_embeddings")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for tables (default: <emb_dir>/analysis)")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.80, 0.90, 1.00])
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.emb_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*65)
    print("FBC Cohort Discovery and Membership Analysis")
    print("="*65)

    # Load data
    ext = os.path.splitext(args.data)[1].lower()
    df = pd.read_excel(args.data) if ext in (".xls", ".xlsx") else pd.read_csv(args.data)
    N = len(df)
    print(f"Loaded {N} records.")

    # Resolve columns
    col_map_path = os.path.join(args.emb_dir, "column_map.json")
    if os.path.exists(col_map_path):
        with open(col_map_path) as f:
            col_resolved = json.load(f)
        print("Column map loaded from embedder output.")
    else:
        col_resolved = resolve_columns(df, COLUMN_MAP)
        print("Column map resolved from defaults.")

    print("Columns in use:")
    for k, v in col_resolved.items():
        print(f"  {k:6s} → {v}")

    # Build condition masks
    print("\nBuilding disease condition masks...")
    cond_masks = build_condition_masks(df, col_resolved)

    # Overall prevalence
    pi_c = {k: float(m.mean()) for k, m in cond_masks.items()}
    df_prev = prevalence_table(pi_c)
    print("\n[1] Overall disease prevalence in FBC dataset:")
    print(df_prev.to_string(index=False))
    prev_path = os.path.join(out_dir, "prevalence.csv")
    df_prev.to_csv(prev_path, index=False)
    print(f"\n    Saved → {prev_path}")

    # Load embeddings
    emb_path = os.path.join(args.emb_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}\nRun FBC_ssl_embedder.py first.")
    E = np.load(emb_path)
    print(f"\nLoaded embeddings: {E.shape}")

    # ── DBSCAN in embedding space ───────────────────────────────────────────
    print(f"\n[2] DBSCAN in embedding space (eps={args.eps}, min_samples={args.min_samples})")
    db_emb = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="euclidean", n_jobs=-1)
    labels_emb = db_emb.fit_predict(E)
    n_clusters_emb = len(set(labels_emb) - {-1})
    noise_emb = int((labels_emb == -1).sum())
    print(f"    Subspaces found: {n_clusters_emb}  |  Noise points: {noise_emb} ({100*noise_emb/N:.1f}%)")

    _, records_emb = analyse_clusters(labels_emb, cond_masks, N)

    # ── DBSCAN on raw features (comparison) ────────────────────────────────
    print(f"\n[3] DBSCAN on raw (standardised) features — same parameters (comparison baseline)")
    feat_cols_all = list(col_resolved.values())
    feat_cols_num = [c for c in feat_cols_all
                     if c in df.columns and c != col_resolved.get("SEX")
                     and df[c].dtype != object]
    imp = SimpleImputer(strategy="median"); scaler = StandardScaler()
    X_raw = scaler.fit_transform(imp.fit_transform(df[feat_cols_num].astype(float).values))

    db_raw = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="euclidean", n_jobs=-1)
    labels_raw = db_raw.fit_predict(X_raw)
    n_clusters_raw = len(set(labels_raw) - {-1})
    noise_raw = int((labels_raw == -1).sum())
    print(f"    Subspaces found: {n_clusters_raw}  |  Noise points: {noise_raw} ({100*noise_raw/N:.1f}%)")

    _, records_raw = analyse_clusters(labels_raw, cond_masks, N)

    # ── Comparison summary ──────────────────────────────────────────────────
    print("\n[4] Comparison: embedding space vs raw feature space")
    df_comp = comparison_summary(records_emb, records_raw, threshold=0.90)
    print("\n" + df_comp.to_string(index=False))
    df_comp.to_csv(os.path.join(out_dir, "comparison_emb_vs_raw.csv"), index=False)

    # ── Membership tables at each threshold ────────────────────────────────
    print("\n[5] Membership confidence tables")
    for thr in args.thresholds:
        df_thr = threshold_table(records_emb, thr)
        print(f"\n    Threshold rho >= {thr}:  {len(df_thr)} associations "
              f"across {df_thr['cluster'].nunique() if not df_thr.empty else 0} subspaces")
        if not df_thr.empty:
            print(df_thr.head(20).to_string(index=False))

        thr_str = str(thr).replace(".", "p")
        csv_path = os.path.join(out_dir, f"membership_rho{thr_str}.csv")
        df_thr.to_csv(csv_path, index=False)
        tex_path = os.path.join(out_dir, f"membership_rho{thr_str}.tex")
        membership_to_latex(df_thr, thr, tex_path)
        print(f"    Saved → {csv_path}")

    # ── Top enriched subspaces ─────────────────────────────────────────────
    print("\n[6] Top enriched subspaces (highest λ, rho >= 0.80):")
    df_top = top_enriched(records_emb, top_n=15)
    print("\n" + df_top.to_string(index=False))
    df_top.to_csv(os.path.join(out_dir, "top_enriched.csv"), index=False)

    # ── Enrichment ratio summary per condition ─────────────────────────────
    print("\n[7] Per-condition enrichment summary (max λ across all subspaces, rho >= 0.80):")
    df_all = pd.DataFrame(records_emb)
    df_high = df_all[df_all["rho"] >= 0.80]
    if not df_high.empty:
        enr_rows = []
        for k in sorted(DISEASE_NAMES):
            sub = df_high[df_high["condition"] == k]
            if sub.empty:
                enr_rows.append({"Tag": k, "Disease": DISEASE_NAMES[k],
                                  "π_c": round(pi_c[k], 4), "max_lambda": 0.0,
                                  "n_subspaces (rho>=0.80)": 0})
            else:
                enr_rows.append({"Tag": k, "Disease": DISEASE_NAMES[k],
                                  "π_c": round(pi_c[k], 4),
                                  "max_lambda": round(sub["lambda"].max(), 2),
                                  "n_subspaces (rho>=0.80)": len(sub)})
        df_enr = pd.DataFrame(enr_rows).sort_values("max_lambda", ascending=False)
        print("\n" + df_enr.to_string(index=False))
        df_enr.to_csv(os.path.join(out_dir, "enrichment_summary.csv"), index=False)

    # ── Cluster size distribution plot ────────────────────────────────────
    sizes = [int((labels_emb == c).sum()) for c in sorted(set(labels_emb) - {-1})]
    plt.figure(figsize=(8, 4))
    plt.hist(sizes, bins=30, color="#4477AA", edgecolor="white")
    plt.xlabel("Subspace size (n patients)"); plt.ylabel("Number of subspaces")
    plt.title(f"FBC Subspace Size Distribution ({n_clusters_emb} subspaces, eps={args.eps})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subspace_sizes.png"), dpi=150)
    plt.close()

    # Save full records
    pd.DataFrame(records_emb).to_csv(os.path.join(out_dir, "all_records_emb.csv"), index=False)
    pd.DataFrame(records_raw).to_csv(os.path.join(out_dir, "all_records_raw.csv"), index=False)

    print(f"\n{'='*65}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
