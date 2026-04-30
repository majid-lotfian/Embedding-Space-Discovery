"""
CH_ssl_embedder.py
==================
Self-Supervised FT-Transformer Embedder for the Congenital Hypothyroidism dataset.

IMPORTANT: CH3 labels are NEVER used during training.
They are extracted from the dataset, saved alongside embeddings, and used
only for post-hoc geometry evaluation (printed at the end).

Training objective: masked feature reconstruction.
  - Each training step randomly masks ~30 % of numeric features and categorical features.
  - Masked numeric positions are replaced by a learned MASK token vector.
  - Masked categorical positions are replaced with a dedicated MASK embedding.
  - The transformer's CLS output is used to reconstruct the original masked values.
  - Loss: MSE for numeric features + cross-entropy for categorical features (masked only).

Usage:
    python CH_ssl_embedder.py --data path/to/ch_data.csv --out_dir ./ch_embeddings_ssl

Expected columns (any extra columns are silently ignored):
    kind, sex, Year, CH, CH3, pregnancy_dur, birthweight, hp_hour, T4-sd, T4, TSH, TBG,
    T4TBG, c101, c14, c142, c161, c181oh, c6, tyr, C14:1/C16, c16, c2, c5c2, c5, c5dc,
    c5oh, c8, c8c10, c10, c141, c141c2, c16oh, phe, sa, val, phetyr, leu, c0

Outputs per embedding dimension d in <out_dir>/repr_dim_<d>/:
    embeddings.npy       (N x d)  — all patient embeddings
    labels_CH3.npy       (N,)     — integer labels {0=no-CH, 1=CH-T, 2=CH-C} (NOT used in training)
    train_idx.npy                 — row indices used for SSL pretraining split
    val_idx.npy                   — row indices used for SSL validation split
    pca_ch3.png / tsne_ch3.png / umap_ch3.png
    summary.json

Requirements:
    pip install torch numpy pandas scikit-learn matplotlib
    pip install umap-learn   # optional but recommended
"""

import os, json, argparse, random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap as umap_module
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_ch_data(path: str):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xls", ".xlsx") else pd.read_csv(path)

    if "CH3" not in df.columns:
        raise ValueError("Column 'CH3' not found.")

    y = df["CH3"].astype(int).values
    drop_cols = [c for c in ["CH", "CH3", "Year"] if c in df.columns]
    Xdf = df.drop(columns=drop_cols).copy()

    cat_cols = [c for c in Xdf.columns if Xdf[c].dtype == object]
    for c in ["kind", "sex"]:
        if c in Xdf.columns and c not in cat_cols:
            cat_cols.append(c)
    num_cols = [c for c in Xdf.columns if c not in cat_cols]
    return Xdf, y, num_cols, cat_cols

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
class PrepArtifacts:
    def __init__(self, imp, scaler, cat_maps, num_cols, cat_cols):
        self.imp, self.scaler, self.cat_maps = imp, scaler, cat_maps
        self.num_cols, self.cat_cols = num_cols, cat_cols

def fit_prep(df_tr: pd.DataFrame, num_cols, cat_cols) -> PrepArtifacts:
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xn = df_tr[num_cols].astype(float).values
    scaler.fit(imp.fit_transform(Xn))
    cat_maps = {}
    for c in cat_cols:
        vals = sorted(df_tr[c].astype(str).fillna("__NA__").unique().tolist())
        cat_maps[c] = {v: i for i, v in enumerate(vals)}
    return PrepArtifacts(imp, scaler, cat_maps, num_cols, cat_cols)

def apply_prep(df: pd.DataFrame, art: PrepArtifacts):
    Xn = art.scaler.transform(art.imp.transform(df[art.num_cols].astype(float).values))
    if art.cat_cols:
        cols = []
        for c in art.cat_cols:
            vals = df[c].astype(str).fillna("__NA__").values
            m = art.cat_maps[c]
            cols.append(np.array([m.get(v, len(m)) for v in vals], dtype=np.int64))
        Xc = np.stack(cols, axis=1)
    else:
        Xc = np.zeros((len(df), 0), dtype=np.int64)
    return Xn.astype(np.float32), Xc

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TabDataset(Dataset):
    def __init__(self, Xn, Xc):
        self.Xn, self.Xc = Xn, Xc
    def __len__(self): return len(self.Xn)
    def __getitem__(self, i):
        return torch.from_numpy(self.Xn[i]), torch.from_numpy(self.Xc[i])

# ─────────────────────────────────────────────────────────────────────────────
# Model: FT-Transformer with masked feature reconstruction
# ─────────────────────────────────────────────────────────────────────────────
class SSLFTTransformer(nn.Module):
    """
    FT-Transformer trained via masked feature reconstruction.
    No classifier head. Labels are never seen during forward/backward.
    """
    def __init__(self, n_num: int, cat_cardinalities: List[int],
                 d_token=64, n_heads=8, n_layers=3, dropout=0.15, repr_dim=32):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token
        self.cat_cardinalities = cat_cardinalities

        # Numeric: one linear tokenizer + one learnable MASK vector per feature
        if n_num > 0:
            self.num_tok = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)])
            self.num_mask = nn.ParameterList(
                [nn.Parameter(torch.randn(d_token) * 0.02) for _ in range(n_num)])

        # Categorical: embedding table with cardinality+1 entries (last = MASK id)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(c + 1, d_token) for c in cat_cardinalities])

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)

        # Representation head: CLS → compact embedding
        h = max(d_token, repr_dim)
        self.repr_head = nn.Sequential(
            nn.Linear(d_token, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, repr_dim))

        # Reconstruction heads (from CLS): numeric → n_num values; cat → one logit-vector each
        if n_num > 0:
            self.num_recon = nn.Linear(d_token, n_num)
        if self.n_cat > 0:
            self.cat_recon = nn.ModuleList(
                [nn.Linear(d_token, c) for c in cat_cardinalities])

    def forward(self, x_num, x_cat, mask_num=None, mask_cat=None):
        B = x_num.size(0)
        tokens = []

        for i in range(self.n_num):
            actual = self.num_tok[i](x_num[:, i].unsqueeze(-1))          # (B, d)
            if mask_num is not None:
                masked_tok = self.num_mask[i].unsqueeze(0).expand(B, -1) # (B, d)
                m = mask_num[:, i].float().unsqueeze(-1)
                ti = actual * (1.0 - m) + masked_tok * m
            else:
                ti = actual
            tokens.append(ti.unsqueeze(1))

        for j in range(self.n_cat):
            x_j = x_cat[:, j].clone()
            if mask_cat is not None:
                x_j = x_j.clone()
                x_j[mask_cat[:, j]] = self.cat_cardinalities[j]           # MASK id
            tokens.append(self.cat_emb[j](x_j).unsqueeze(1))

        seq = torch.cat(tokens, dim=1) if tokens else torch.zeros(B, 0, self.d_token, device=x_num.device)
        seq = torch.cat([self.cls.expand(B, -1, -1), seq], dim=1)
        seq = self.norm(self.encoder(seq))

        cls_out = seq[:, 0, :]
        repr_vec = self.repr_head(cls_out)

        num_recon = self.num_recon(cls_out) if self.n_num > 0 else None
        cat_recon = [self.cat_recon[j](cls_out) for j in range(self.n_cat)] if self.n_cat > 0 else []

        return repr_vec, num_recon, cat_recon

    @torch.no_grad()
    def embed(self, x_num, x_cat):
        self.eval()
        r, _, _ = self.forward(x_num, x_cat)
        return r

# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
def sample_mask(B, n_feat, frac, device):
    return torch.bernoulli(torch.full((B, n_feat), frac, device=device)).bool()

def ssl_loss(num_recon, cat_recon, x_num, x_cat, mask_num, mask_cat):
    loss = torch.tensor(0.0, device=x_num.device)
    n = 0
    if num_recon is not None and mask_num is not None and mask_num.any():
        masked_pred = num_recon[mask_num]
        masked_orig = x_num[mask_num]
        if masked_orig.numel() > 0:
            loss = loss + F.mse_loss(masked_pred, masked_orig)
            n += 1
    for j, rj in enumerate(cat_recon):
        if mask_cat is not None:
            mj = mask_cat[:, j]
            if mj.any():
                loss = loss + F.cross_entropy(rj[mj], x_cat[mj, j])
                n += 1
    return loss / max(n, 1)

def train_ssl(model, tr_loader, va_loader, device,
              lr=1e-3, wd=1e-4, max_epochs=300, patience=25, mask_frac=0.30):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss, nb = 0.0, 0
        for xn, xc in tr_loader:
            xn, xc = xn.to(device), xc.to(device)
            mn = sample_mask(xn.size(0), xn.size(1), mask_frac, device) if model.n_num > 0 else None
            mc = sample_mask(xc.size(0), xc.size(1), mask_frac, device) if model.n_cat > 0 else None
            _, nr, cr = model(xn, xc, mn, mc)
            loss = ssl_loss(nr, cr, xn, xc, mn, mc)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item(); nb += 1

        model.eval(); va_loss, vb = 0.0, 0
        with torch.no_grad():
            for xn, xc in va_loader:
                xn, xc = xn.to(device), xc.to(device)
                mn = sample_mask(xn.size(0), xn.size(1), mask_frac, device) if model.n_num > 0 else None
                mc = sample_mask(xc.size(0), xc.size(1), mask_frac, device) if model.n_cat > 0 else None
                _, nr, cr = model(xn, xc, mn, mc)
                va_loss += ssl_loss(nr, cr, xn, xc, mn, mc).item(); vb += 1
        va_loss /= max(vb, 1)

        if epoch % 25 == 0:
            print(f"    epoch {epoch:3d}  train={tr_loss/nb:.4f}  val={va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val

@torch.no_grad()
def extract_all(model, Xn, Xc, device, bs=256):
    model.eval(); model.to(device); out = []
    for i in range(0, len(Xn), bs):
        r = model.embed(torch.from_numpy(Xn[i:i+bs]).to(device),
                        torch.from_numpy(Xc[i:i+bs]).to(device))
        out.append(r.cpu().numpy())
    return np.vstack(out)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry metrics (post-hoc against CH3 labels)
# ─────────────────────────────────────────────────────────────────────────────
def geometry_metrics(E, y):
    cls, cnts = np.unique(y, return_counts=True)
    if len(cls) < 2 or cnts.min() < 2:
        return {}
    D = euclidean_distances(E)
    same = (y[:, None] == y[None, :]); np.fill_diagonal(same, False)
    diff = ~(y[:, None] == y[None, :])
    return dict(
        silhouette=float(silhouette_score(E, y)),
        davies_bouldin=float(davies_bouldin_score(E, y)),
        calinski_harabasz=float(calinski_harabasz_score(E, y)),
        within_mean_dist=float(D[same].mean()),
        between_mean_dist=float(D[diff].mean()),
        separation_ratio=float(D[diff].mean() / D[same].mean()),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
COLORS = ["#4477AA", "#EE6677", "#228833"]
LABELS = {0: "no-CH", 1: "CH-T", 2: "CH-C"}

def plot_2d(Z, y, title, path):
    plt.figure(figsize=(6, 5))
    for lab, col in zip([0, 1, 2], COLORS):
        idx = y == lab
        if idx.any():
            plt.scatter(Z[idx, 0], Z[idx, 1], s=15, alpha=0.75,
                        label=LABELS[lab], color=col)
    plt.title(title, fontsize=11); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def make_plots(E, y, d, out_dir):
    n = len(E)
    Z = PCA(n_components=2, random_state=SEED).fit_transform(E)
    plot_2d(Z, y, f"PCA (dim={d})", os.path.join(out_dir, "pca_ch3.png"))

    perp = min(30, max(5, (n - 1) // 3))
    Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=SEED).fit_transform(E)
    plot_2d(Z, y, f"t-SNE (dim={d})", os.path.join(out_dir, "tsne_ch3.png"))

    if HAVE_UMAP:
        Z = umap_module.UMAP(n_components=2, n_neighbors=min(15, n - 1),
                             min_dist=0.1, random_state=SEED).fit_transform(E)
        plot_2d(Z, y, f"UMAP (dim={d})", os.path.join(out_dir, "umap_ch3.png"))
    else:
        print("    umap-learn not installed — skipping UMAP plot.")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="./ch_embeddings_ssl")
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--mask_frac", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CH Self-Supervised FT-Transformer Embedder")
    print("CH3 labels: saved for evaluation only — NOT used in training")
    print("="*60)

    Xdf, y, num_cols, cat_cols = load_ch_data(args.data)
    N = len(Xdf)
    cls, cnts = np.unique(y, return_counts=True)
    print(f"N={N}  numeric_features={len(num_cols)}  cat_features={len(cat_cols)}")
    print(f"Class distribution (for reference only):")
    for c, n in zip(cls, cnts):
        print(f"  CH3={c} ({LABELS.get(c,'?')}): {n} ({100*n/N:.1f}%)")

    # SSL split (stratified for balance — labels used ONLY to ensure representative split,
    # not in the training objective)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, va_idx = next(sss.split(Xdf, y))

    art = fit_prep(Xdf.iloc[tr_idx], num_cols, cat_cols)
    Xtr_n, Xtr_c = apply_prep(Xdf.iloc[tr_idx], art)
    Xva_n, Xva_c = apply_prep(Xdf.iloc[va_idx], art)
    Xall_n, Xall_c = apply_prep(Xdf, art)
    cat_cards = [len(art.cat_maps[c]) for c in cat_cols]

    tr_loader = DataLoader(TabDataset(Xtr_n, Xtr_c), batch_size=args.batch_size,
                           shuffle=True, drop_last=True)
    va_loader = DataLoader(TabDataset(Xva_n, Xva_c), batch_size=args.batch_size, shuffle=False)

    overall = {"seed": SEED, "device": args.device, "mask_frac": args.mask_frac,
               "training_objective": "masked_feature_reconstruction",
               "labels_used_in_training": False,
               "num_cols": num_cols, "cat_cols": cat_cols,
               "class_distribution": {int(k): int(v) for k, v in zip(cls, cnts)},
               "dims": []}

    for d in args.dims:
        print(f"\n{'─'*60}")
        print(f"  Embedding dimension: {d}")
        d_token = 64 if d >= 32 else 32
        n_heads = 8 if d_token >= 64 else 4

        dim_dir = os.path.join(args.out_dir, f"repr_dim_{d}")
        os.makedirs(dim_dir, exist_ok=True)

        model = SSLFTTransformer(len(num_cols), cat_cards,
                                  d_token=d_token, n_heads=n_heads, n_layers=3,
                                  dropout=0.15, repr_dim=d)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        best_val_loss = train_ssl(model, tr_loader, va_loader, args.device,
                                   max_epochs=args.max_epochs, patience=args.patience,
                                   mask_frac=args.mask_frac)
        print(f"  Best validation reconstruction loss: {best_val_loss:.4f}")

        E = extract_all(model, Xall_n, Xall_c, args.device)
        np.save(os.path.join(dim_dir, "embeddings.npy"), E)
        np.save(os.path.join(dim_dir, "labels_CH3.npy"), y)
        np.save(os.path.join(dim_dir, "train_idx.npy"), tr_idx)
        np.save(os.path.join(dim_dir, "val_idx.npy"), va_idx)

        m_all   = geometry_metrics(E, y)
        m_train = geometry_metrics(E[tr_idx], y[tr_idx])
        m_val   = geometry_metrics(E[va_idx], y[va_idx])

        print(f"\n  Post-hoc geometry (evaluated against CH3 labels — not used in training):")
        print(f"  {'Metric':<25} {'All N':>10} {'Train':>10} {'Val (OOS)':>12}")
        print(f"  {'─'*60}")
        for k in ["silhouette", "davies_bouldin", "calinski_harabasz", "separation_ratio"]:
            print(f"  {k:<25} {m_all.get(k, float('nan')):>10.4f} "
                  f"{m_train.get(k, float('nan')):>10.4f} "
                  f"{m_val.get(k, float('nan')):>12.4f}")

        print("\n  Generating plots...")
        make_plots(E, y, d, dim_dir)

        summary = {"repr_dim": d, "d_token": d_token, "n_heads": n_heads,
                   "best_val_loss": float(best_val_loss),
                   "geometry_all": m_all, "geometry_train": m_train, "geometry_val": m_val}
        with open(os.path.join(dim_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        overall["dims"].append(summary)

    with open(os.path.join(args.out_dir, "overall_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
