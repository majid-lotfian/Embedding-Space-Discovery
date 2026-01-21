import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec

# KDE helper
try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY_KDE = True
except Exception:
    HAVE_SCIPY_KDE = False


def _density_curve(x, grid_points=256):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return None, None

    xmin, xmax = np.min(x), np.max(x)
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6

    grid = np.linspace(xmin, xmax, grid_points)

    if HAVE_SCIPY_KDE and len(x) >= 5:
        kde = gaussian_kde(x)
        return grid, kde(grid)
    else:
        hist, edges = np.histogram(x, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, hist


def _joint_pca_with_marginals(
    Z, y, evr, pc_a, pc_b,
    cmap, norm,
    class_labels=(0, 1, 2),
    class_names=None,
    s=12,
    alpha=0.45,
):
    ia, ib = pc_a - 1, pc_b - 1
    x = Z[:, ia]
    z = Z[:, ib]

    fig = plt.figure(figsize=(7.6, 6.6))
    gs = GridSpec(
        2, 2,
        width_ratios=[4.5, 1.5],
        height_ratios=[1.5, 4.5],
        wspace=0.05, hspace=0.05
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # ---- Scatter (joint) ----
    ax_scatter.scatter(x, z, c=y, cmap=cmap, norm=norm, s=s, alpha=alpha)
    ax_scatter.set_xlabel(f"PC{pc_a} ({evr[ia]*100:.1f}% var)")
    ax_scatter.set_ylabel(f"PC{pc_b} ({evr[ib]*100:.1f}% var)")

    # ---- Marginal densities (by class) ----
    for cls in class_labels:
        idx = (y == cls)
        if not np.any(idx):
            continue

        color = cmap(norm(cls))
        label = str(cls) if class_names is None else class_names.get(cls, str(cls))

        gx, dx = _density_curve(x[idx])
        if gx is not None:
            ax_top.plot(gx, dx, color=color, label=label)

        gz, dz = _density_curve(z[idx])
        if gz is not None:
            ax_right.plot(dz, gz, color=color)

    # =========================================================
    # Density plot styling: keep ONLY the separator spine,
    # but DO NOT touch tick LOCATIONS (no set_xticks([]) etc.)
    # =========================================================

    # Top density: show only bottom spine as separator line
    ax_top.spines["bottom"].set_visible(True)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    # Hide ticks/labels on top axis WITHOUT changing tick locations
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Right density: show only left spine as separator line
    ax_right.spines["left"].set_visible(True)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    # Hide ticks/labels on right axis WITHOUT changing tick locations
    ax_right.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)

    # ---- Ensure PCA axis has normal ticks/labels ----
    ax_scatter.tick_params(axis="both", which="both", labelbottom=True, labelleft=True, length=4)

    # ---- Clean scatter axes ----
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    # ---- Legend ----
    ax_top.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    plt.show()



def pca_3plots_from_csv(
    csv_path: str,
    label_col: str = "CH3",
    sex_col: str = "sex",
    date_col: str = "date",
    standardize: bool = True,
):
    df = pd.read_csv(csv_path)

    y = df[label_col].astype("Int64")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int).to_numpy()

    Xdf = df.drop(columns=[label_col], errors="ignore")

    if sex_col in Xdf.columns:
        sex_map = {"m": 0, "v": 1, "M": 0, "V": 1}
        Xdf[sex_col] = Xdf[sex_col].map(sex_map)

    if date_col in Xdf.columns:
        dt = pd.to_datetime(Xdf[date_col], errors="coerce", utc=True)
        ns = dt.view("int64")
        ns = ns.where(dt.notna(), np.nan)
        Xdf[date_col] = (ns / 1e9) / (60 * 60 * 24)

    Xnum = Xdf.select_dtypes(include=[np.number]).copy()
    X = SimpleImputer(strategy="median").fit_transform(Xnum)

    if standardize:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=3, random_state=0)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    cmap = ListedColormap(["blue", "orange", "green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    class_names = {0: "no-CH", 1: "CH-T", 2: "CH-C"}

    _joint_pca_with_marginals(Z, y, evr, 1, 2, cmap, norm, class_names=class_names)
    _joint_pca_with_marginals(Z, y, evr, 1, 3, cmap, norm, class_names=class_names)
    _joint_pca_with_marginals(Z, y, evr, 2, 3, cmap, norm, class_names=class_names)


# Example:
pca_3plots_from_csv("Data_2018_2021_2022.csv", label_col="CH3", sex_col="sex", date_col="Year")
