"""
FBC_ssl_embedder.py
===================
Self-Supervised FT-Transformer Embedder for the Full Blood Count dataset.

No labels available or used. Training objective: masked feature reconstruction.

Usage:
    python FBC_ssl_embedder.py --data path/to/fbc_data.csv --out_dir ./fbc_embeddings

Column mapping — edit COLUMN_MAP below if your column names differ.
The script will print all detected columns at startup so you can verify.

Expected haematological features (adapt COLUMN_MAP to your dataset):
    RBC, WBC, NEU (neutrophils), LYM (lymphocytes), EOS (eosinophils),
    HGB (haemoglobin), HCT (haematocrit), PLT (platelets),
    MCV, MPV, RDW, FER (ferritin), and derived indices.

Optional (used for sex-specific disease thresholds in FBC_analysis.py):
    SEX  — binary or string column (0/1 or 'M'/'F')

Output in <out_dir>/:
    embeddings.npy      (N x 128)
    summary.json
"""

import os, json, argparse, random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Column map: canonical name → list of possible column names in your data
# Edit this if your CSV uses different column names.
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "RBC":  ["RBC", "rbc", "red_blood_cells", "Red Blood Cells"],
    "WBC":  ["WBC", "wbc", "white_blood_cells", "White Blood Cells", "Leukocytes"],
    "NEU":  ["NEU", "neu", "neutrophils", "Neutrophils", "NEUT", "neut"],
    "LYM":  ["LYM", "lym", "lymphocytes", "Lymphocytes", "LYMPH"],
    "EOS":  ["EOS", "eos", "eosinophils", "Eosinophils"],
    "MON":  ["MON", "mon", "monocytes", "Monocytes", "MONO"],
    "BAS":  ["BAS", "bas", "basophils", "Basophils"],
    "HGB":  ["HGB", "hgb", "haemoglobin", "hemoglobin", "Hemoglobin", "Haemoglobin", "HB"],
    "HCT":  ["HCT", "hct", "haematocrit", "hematocrit", "Hematocrit"],
    "PLT":  ["PLT", "plt", "platelets", "Platelets", "Thrombocytes"],
    "MCV":  ["MCV", "mcv", "mean_corpuscular_volume"],
    "MCH":  ["MCH", "mch"],
    "MCHC": ["MCHC", "mchc"],
    "MPV":  ["MPV", "mpv", "mean_platelet_volume"],
    "RDW":  ["RDW", "rdw", "red_cell_distribution_width"],
    "FER":  ["FER", "fer", "ferritin", "Ferritin", "FERRITIN"],
    "B12":  ["B12", "b12", "vitamin_b12", "VitB12"],
    "FOL":  ["FOL", "fol", "folate", "Folate", "folic_acid"],
    "SEX":  ["SEX", "sex", "Sex", "gender", "Gender"],
}

def resolve_columns(df: pd.DataFrame, col_map: dict) -> dict:
    """Map canonical names to actual column names found in df."""
    resolved = {}
    for canon, candidates in col_map.items():
        for cand in candidates:
            if cand in df.columns:
                resolved[canon] = cand
                break
    return resolved

# ─────────────────────────────────────────────────────────────────────────────
# Shared SSL model (same architecture as CH version)
# ─────────────────────────────────────────────────────────────────────────────
class SSLFTTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: List[int],
                 d_token=64, n_heads=8, n_layers=3, dropout=0.15, repr_dim=128):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token
        self.cat_cardinalities = cat_cardinalities

        if n_num > 0:
            self.num_tok = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)])
            self.num_mask = nn.ParameterList(
                [nn.Parameter(torch.randn(d_token) * 0.02) for _ in range(n_num)])

        self.cat_emb = nn.ModuleList(
            [nn.Embedding(c + 1, d_token) for c in cat_cardinalities])

        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)

        h = max(d_token, repr_dim)
        self.repr_head = nn.Sequential(
            nn.Linear(d_token, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, repr_dim))

        if n_num > 0:
            self.num_recon = nn.Linear(d_token, n_num)
        if self.n_cat > 0:
            self.cat_recon = nn.ModuleList(
                [nn.Linear(d_token, c) for c in cat_cardinalities])

    def forward(self, x_num, x_cat, mask_num=None, mask_cat=None):
        B = x_num.size(0)
        tokens = []
        for i in range(self.n_num):
            actual = self.num_tok[i](x_num[:, i].unsqueeze(-1))
            if mask_num is not None:
                mt = self.num_mask[i].unsqueeze(0).expand(B, -1)
                m  = mask_num[:, i].float().unsqueeze(-1)
                ti = actual * (1.0 - m) + mt * m
            else:
                ti = actual
            tokens.append(ti.unsqueeze(1))
        for j in range(self.n_cat):
            xj = x_cat[:, j].clone()
            if mask_cat is not None:
                xj[mask_cat[:, j]] = self.cat_cardinalities[j]
            tokens.append(self.cat_emb[j](xj).unsqueeze(1))

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

class TabDataset(Dataset):
    def __init__(self, Xn, Xc):
        self.Xn, self.Xc = Xn, Xc
    def __len__(self): return len(self.Xn)
    def __getitem__(self, i):
        return torch.from_numpy(self.Xn[i]), torch.from_numpy(self.Xc[i])

def sample_mask(B, n, frac, device):
    return torch.bernoulli(torch.full((B, n), frac, device=device)).bool()

def ssl_loss(num_recon, cat_recon, x_num, x_cat, mask_num, mask_cat):
    loss = torch.tensor(0.0, device=x_num.device); n = 0
    if num_recon is not None and mask_num is not None and mask_num.any():
        p, o = num_recon[mask_num], x_num[mask_num]
        if o.numel() > 0:
            loss = loss + F.mse_loss(p, o); n += 1
    for j, rj in enumerate(cat_recon):
        if mask_cat is not None:
            mj = mask_cat[:, j]
            if mj.any():
                loss = loss + F.cross_entropy(rj[mj], x_cat[mj, j]); n += 1
    return loss / max(n, 1)

def train_ssl(model, tr_loader, va_loader, device,
              lr=1e-3, wd=1e-4, max_epochs=200, patience=20, mask_frac=0.30):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train(); tr_loss, nb = 0.0, 0
        for xn, xc in tr_loader:
            xn, xc = xn.to(device), xc.to(device)
            mn = sample_mask(xn.size(0), xn.size(1), mask_frac, device) if model.n_num > 0 else None
            mc = sample_mask(xc.size(0), xc.size(1), mask_frac, device) if model.n_cat > 0 else None
            _, nr, cr = model(xn, xc, mn, mc)
            loss = ssl_loss(nr, cr, xn, xc, mn, mc)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tr_loss += loss.item(); nb += 1

        model.eval(); va_loss, vb = 0.0, 0
        with torch.no_grad():
            for xn, xc in va_loader:
                xn, xc = xn.to(device), xc.to(device)
                mn = sample_mask(xn.size(0), xn.size(1), mask_frac, device) if model.n_num > 0 else None
                mc = sample_mask(xc.size(0), xc.size(1), mask_frac, device) if model.n_cat > 0 else None
                _, nr, cr = model(xn, xc, mn, mc)
                va_loss += ssl_loss(nr, cr, xn, xc, mn, mc).item(); vb += 1
        va_loss /= max(vb, 1)

        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train={tr_loss/nb:.4f}  val={va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stop at epoch {epoch}"); break

    if best_state: model.load_state_dict(best_state)
    return best_val

@torch.no_grad()
def extract_all(model, Xn, Xc, device, bs=512):
    model.eval(); model.to(device); out = []
    for i in range(0, len(Xn), bs):
        r = model.embed(torch.from_numpy(Xn[i:i+bs]).to(device),
                        torch.from_numpy(Xc[i:i+bs]).to(device))
        out.append(r.cpu().numpy())
    return np.vstack(out)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to FBC dataset CSV")
    parser.add_argument("--out_dir", default="./fbc_embeddings")
    parser.add_argument("--repr_dim", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mask_frac", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("FBC Self-Supervised FT-Transformer Embedder")
    print("="*60)

    ext = os.path.splitext(args.data)[1].lower()
    df = pd.read_excel(args.data) if ext in (".xls", ".xlsx") else pd.read_csv(args.data)
    N = len(df)
    print(f"Loaded {N} records with {df.shape[1]} columns.")

    col_resolved = resolve_columns(df, COLUMN_MAP)
    print("\nColumn mapping resolved:")
    for canon, actual in col_resolved.items():
        print(f"  {canon:6s} → {actual}")
    missing = [k for k in COLUMN_MAP if k not in col_resolved and k != "SEX"]
    if missing:
        print(f"\nWARNING: Could not find columns for: {missing}")
        print("  Edit COLUMN_MAP at the top of this script to match your data.")

    # Identify numeric and categorical features (exclude SEX from embedding if present)
    exclude = list(col_resolved.get(k, "__none__") for k in ["SEX"])
    feat_cols = [c for c in df.columns if c not in exclude]

    cat_cols = [c for c in feat_cols if df[c].dtype == object]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    print(f"\nFeatures used: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # Save column map for FBC_analysis.py
    with open(os.path.join(args.out_dir, "column_map.json"), "w") as f:
        json.dump({k: v for k, v in col_resolved.items()}, f, indent=2)

    # Preprocessing
    tr_idx, va_idx = train_test_split(np.arange(N), test_size=0.10, random_state=SEED)
    imp = SimpleImputer(strategy="median"); scaler = StandardScaler()
    Xn = df[num_cols].astype(float).values
    scaler.fit(imp.fit_transform(Xn[tr_idx]))
    Xn_all = scaler.transform(imp.transform(Xn)).astype(np.float32)

    if cat_cols:
        cat_maps = {}
        for c in cat_cols:
            vals = sorted(df[c].astype(str).fillna("__NA__").unique().tolist())
            cat_maps[c] = {v: i for i, v in enumerate(vals)}
        Xc_all = np.stack([
            np.array([cat_maps[c].get(v, len(cat_maps[c]))
                      for v in df[c].astype(str).fillna("__NA__").values], dtype=np.int64)
            for c in cat_cols], axis=1)
        cat_cards = [len(cat_maps[c]) for c in cat_cols]
    else:
        Xc_all = np.zeros((N, 0), dtype=np.int64)
        cat_cards = []

    tr_loader = DataLoader(TabDataset(Xn_all[tr_idx], Xc_all[tr_idx]),
                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(TabDataset(Xn_all[va_idx], Xc_all[va_idx]),
                           batch_size=args.batch_size, shuffle=False)

    model = SSLFTTransformer(len(num_cols), cat_cards,
                              d_token=64, n_heads=8, n_layers=3,
                              dropout=0.15, repr_dim=args.repr_dim)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val_loss = train_ssl(model, tr_loader, va_loader, args.device,
                               max_epochs=args.max_epochs, patience=args.patience,
                               mask_frac=args.mask_frac)
    print(f"\nBest validation reconstruction loss: {best_val_loss:.4f}")

    print("Extracting embeddings for all records...")
    E = extract_all(model, Xn_all, Xc_all, args.device)
    print(f"Embedding matrix shape: {E.shape}")

    np.save(os.path.join(args.out_dir, "embeddings.npy"), E)

    # Quick PCA plot (no labels, coloured by density via scatter alpha)
    Z = PCA(n_components=2, random_state=SEED).fit_transform(E)
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.15, color="#4477AA")
    plt.title("FBC Embedding Space — PCA (128-dim SSL)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pca_fbc.png"), dpi=150)
    plt.close()

    summary = {"N": N, "repr_dim": args.repr_dim, "mask_frac": args.mask_frac,
               "best_val_loss": float(best_val_loss),
               "num_cols": num_cols, "cat_cols": cat_cols,
               "col_resolved": col_resolved}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Saved to: {args.out_dir}\n")
    print("Next step: python FBC_analysis.py --data <fbc_data.csv> --emb_dir ./fbc_embeddings")


if __name__ == "__main__":
    main()
