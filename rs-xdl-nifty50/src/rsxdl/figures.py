"""Publication figures from artifacts/ CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_figures(output_dir: Path | str) -> list[str]:
    out = Path(output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    wf_path = out / "walk_forward_folds.csv"
    if wf_path.exists():
        folds = pd.read_csv(wf_path)
        if not folds.empty and "rmse" in folds.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(folds["fold"].astype(str), folds["rmse"], color="#534AB7")
            ax.set_xlabel("Fold")
            ax.set_ylabel("RMSE")
            ax.set_title("Walk-forward RMSE by fold")
            ax.grid(True, axis="y", alpha=0.3)
            path = fig_dir / "walk_forward_rmse.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    shap_path = out / "rc_shap.csv"
    if shap_path.exists():
        shap = pd.read_csv(shap_path)
        regime_slices = shap[shap["slice"].str.startswith("regime_")]
        if not regime_slices.empty:
            pivot = regime_slices.pivot_table(
                index="feature",
                columns="slice",
                values="mean_abs_shap",
                aggfunc="first",
            )
            top = pivot.mean(axis=1).nlargest(12).index
            heat = pivot.loc[top]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels([c.replace("regime_", "R") for c in heat.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(heat.index, fontsize=8)
            ax.set_title("RC-SHAP mean |SHAP| by regime (top 12)")
            fig.colorbar(im, ax=ax, fraction=0.03)
            path = fig_dir / "rc_shap_heatmap.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    book_path = out / "portfolio.csv"
    if book_path.exists():
        book = pd.read_csv(book_path, parse_dates=["date"])
        if {"net_return", "equal_weight_return"}.issubset(book.columns):
            book = book.sort_values("date")
            book["net_cum"] = (1 + book["net_return"]).cumprod()
            book["eq_cum"] = (1 + book["equal_weight_return"]).cumprod()
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(book["date"], book["net_cum"], label="Top-forecast (net)", lw=1.2)
            ax.plot(book["date"], book["eq_cum"], label="Equal weight", lw=1.0, ls="--")
            ax.set_ylabel("Growth of 1")
            ax.set_title("Portfolio cumulative return (hold-out test)")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            path = fig_dir / "portfolio_cumulative.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    kw_path = out / "kruskal_wallis_shap.csv"
    if kw_path.exists():
        kw = pd.read_csv(kw_path).dropna(subset=["p_value"]).sort_values("p_value").head(12)
        if not kw.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#047857" if s else "#64748b" for s in kw["significant"]]
            ax.barh(kw["feature"][::-1], -np.log10(kw["p_value"].clip(1e-6, 1))[::-1], color=colors[::-1])
            ax.set_xlabel("-log10(p-value)")
            ax.set_title("Kruskal–Wallis: regime shift in |SHAP|")
            path = fig_dir / "kruskal_wallis.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    return written