"""Step 3: Exploratory data analysis on the CLEANED data (descriptive only, no model fitting).

Outputs:
  figures/fig1_class_balance.png ... fig5_ports_and_time.png
  tables/feature_separability.csv, high_correlation_pairs.csv, class_summary_stats.csv,
         top_ports.csv, eda_findings.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import PORTSCAN_CLEAN, LABEL_COL, TARGET, PORT_COL, COLORS, TAB_DIR, CONSTANT_COLS, DUPLICATE_COLS
from utils import save_table, save_fig, style

style()
NON_FEATURES = {LABEL_COL, TARGET, "raw_row"}


def fig1_class_balance(df):
    cd = pd.read_csv(TAB_DIR / "class_distribution.csv").set_index("label").loc[["BENIGN", "PortScan"]]
    clean = df[LABEL_COL].value_counts()
    cd["dedup_count"] = clean.reindex(cd.index)            # final cleaned counts (after all steps)
    cd["dedup_pct"] = cd.dedup_count / cd.dedup_count.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(2); w = 0.38
    for i, lab in enumerate(cd.index):
        for j, (col, pcol, alpha) in enumerate([("raw_count", "raw_pct", 0.45), ("dedup_count", "dedup_pct", 1.0)]):
            v = cd.loc[lab, col]
            b = ax.bar(x[j] + (i - .5) * w, v, w * 0.94, color=COLORS[lab], alpha=alpha,
                       label=lab if j == 1 else None)
            ax.text(x[j] + (i - .5) * w, v + 2500, f"{v:,.0f}\n({cd.loc[lab, pcol]:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["Raw file", "After cleaning (incl. de-duplication)"])
    ax.set_ylabel("Number of flows")
    ax.set_ylim(0, 200000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_title("Class balance before and after cleaning")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save_fig(fig, "fig1_class_balance")


def fig2_distributions(df):
    feats = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
             "Fwd Packet Length Max", "Flow Packets/s", "Init_Win_bytes_backward"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, f in zip(axes.flat, feats):
        allv = np.log1p(df[f].clip(lower=0))
        bins = np.linspace(allv.min(), allv.quantile(.999), 50)
        for lab in ["BENIGN", "PortScan"]:
            v = np.log1p(df.loc[df[LABEL_COL] == lab, f].clip(lower=0))
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=COLORS[lab])
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=COLORS[lab], label=lab)
        ax.set_title(f, fontsize=11)
        ax.set_xlabel("log(1 + value)")
        ax.set_ylabel("Density")
    axes.flat[0].legend()
    fig.suptitle("Port-scan flows are tiny and short: distribution of key features by class",
                 fontweight="bold", fontsize=13)
    save_fig(fig, "fig2_feature_distributions")


def separability(df, feats):
    y = df[TARGET].values
    rows = []
    for f in feats:
        x = df[f].values.astype(float)
        auc = roc_auc_score(y, x) if np.nanstd(x) > 0 else 0.5
        r = np.corrcoef(x, y)[0, 1] if np.nanstd(x) > 0 else 0.0
        rows.append({"feature": f, "univariate_auc": max(auc, 1 - auc),
                     "direction": "higher in PortScan" if auc >= 0.5 else "lower in PortScan",
                     "point_biserial_r": r, "abs_r": abs(r)})
    sep = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    save_table(sep, "feature_separability")
    return sep


def fig3_separability(sep):
    top = sep.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [COLORS["PortScan"] if d == "higher in PortScan" else COLORS["BENIGN"] for d in top.direction]
    ax.barh(top.feature, top.univariate_auc - 0.5, left=0.5, color=colors, height=0.7)
    for i, v in enumerate(top.univariate_auc):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.03)
    ax.set_xlabel("Univariate ROC-AUC (0.5 = useless alone, 1.0 = perfect alone)")
    ax.set_title("Top 20 single features for separating scans from benign")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["PortScan"], label="Higher in PortScan"),
                       Patch(color=COLORS["BENIGN"], label="Lower in PortScan")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "fig3_feature_separability")


def fig4_correlation(df, sep, feats):
    corr = df[feats].corr()
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.95:
                pairs.append({"feature_a": a, "feature_b": b, "r": r})
    save_table(pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False), "high_correlation_pairs")

    top = sep.head(20).feature.tolist()
    c = df[top].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top)), top, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(top)), top, fontsize=8.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pearson correlation")
    ax.set_title("Correlation among the top-20 separating features")
    save_fig(fig, "fig4_correlation_heatmap")
    return len(pairs)


def fig5_ports_and_time(df):
    rows = []
    for lab, g in df.groupby(LABEL_COL):
        vc = g[PORT_COL].value_counts()
        top = vc.head(15)
        for p, n in top.items():
            rows.append({"label": lab, "port": int(p), "flows": int(n), "pct_of_class": n / len(g) * 100})
    save_table(pd.DataFrame(rows), "top_ports")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1, 1.4, 1.6]})
    # (a) distinct ports
    dp = df.groupby(LABEL_COL)[PORT_COL].nunique()
    a1.bar(dp.index, dp.values, color=[COLORS[l] for l in dp.index], width=0.6)
    for i, v in enumerate(dp.values):
        a1.text(i, v * 1.02, f"{v:,}", ha="center", va="bottom", fontsize=10)
    a1.set_ylim(0, dp.max() * 1.15)
    a1.set_ylabel("Distinct destination ports")
    a1.set_title("(a) Distinct ports targeted", fontsize=11)
    a1.grid(axis="x", visible=False)
    # (b) share of flows on well-known ports (<1024)
    wk = df.assign(wk=df[PORT_COL] < 1024).groupby(LABEL_COL)["wk"].mean() * 100
    a2.bar(wk.index, wk.values, color=[COLORS[l] for l in wk.index], width=0.6)
    for i, v in enumerate(wk.values):
        a2.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    a2.set_ylim(0, 105)
    a2.set_ylabel("% of flows to ports < 1024")
    a2.set_title("(b) Flows to well-known ports", fontsize=11)
    a2.grid(axis="x", visible=False)
    # (c) scan share across file order (time proxy)
    bins = pd.qcut(df["raw_row"], 20, labels=False)
    share = df.groupby(bins)[TARGET].mean() * 100
    a3.plot(share.index + 1, share.values, color=COLORS["PortScan"], lw=2, marker="o", ms=5)
    a3.set_ylim(0, 100)
    a3.set_xticks([1, 5, 10, 15, 20])
    a3.set_xlabel("Position in file (20 equal bins, earliest -> latest)")
    a3.set_ylabel("% of flows that are PortScan")
    a3.set_title("(c) Scan share over file order (time proxy)", fontsize=11)
    fig.suptitle("Port behavior and when scans occur", fontweight="bold", fontsize=13)
    save_fig(fig, "fig5_ports_and_time")
    return dp, wk, share


def main():
    df = pd.read_parquet(PORTSCAN_CLEAN)
    # EDA describes the distinct information in the data, so constant and exact-copy columns are
    # left out of the separability/correlation analysis (they are still kept for modeling).
    feats = [c for c in df.columns if c not in NON_FEATURES and c not in CONSTANT_COLS + DUPLICATE_COLS]
    print(f"EDA on {len(df):,} rows, {len(feats)} features")

    fig1_class_balance(df)
    fig2_distributions(df)
    sep = separability(df, feats)
    fig3_separability(sep)
    n_pairs = fig4_correlation(df, sep, feats)
    dp, wk, share = fig5_ports_and_time(df)

    # class summary stats
    stats = []
    for f in feats:
        row = {"feature": f}
        for lab, g in df.groupby(LABEL_COL):
            q1, med, q3 = g[f].quantile([.25, .5, .75])
            row[f"{lab}_median"] = med
            row[f"{lab}_IQR"] = q3 - q1
        stats.append(row)
    stats = save_table(pd.DataFrame(stats), "class_summary_stats").set_index("feature")

    w1024 = df.assign(w=df["Init_Win_bytes_forward"] == 1024).groupby(LABEL_COL)["w"].mean() * 100

    def m(f, lab): return stats.loc[f, f"{lab}_median"]
    findings = [
        ("Duration", f"Median Flow Duration: PortScan {m('Flow Duration','PortScan'):,.0f} µs vs BENIGN {m('Flow Duration','BENIGN'):,.0f} µs"),
        ("Packets", f"Median fwd/bwd packets: PortScan {m('Total Fwd Packets','PortScan'):.0f}/{m('Total Backward Packets','PortScan'):.0f} vs BENIGN {m('Total Fwd Packets','BENIGN'):.0f}/{m('Total Backward Packets','BENIGN'):.0f}"),
        ("Payload", f"Median Fwd Packet Length Max: PortScan {m('Fwd Packet Length Max','PortScan'):.0f} bytes vs BENIGN {m('Fwd Packet Length Max','BENIGN'):.0f} bytes"),
        ("Separability", f"{(sep.univariate_auc > 0.9).sum()} features have univariate AUC > 0.90; best is {sep.iloc[0].feature} ({sep.iloc[0].univariate_auc:.3f})"),
        ("Redundancy", f"{n_pairs} feature pairs have |r| > 0.95"),
        ("Ports", f"PortScan hits {dp['PortScan']:,} distinct ports vs {dp['BENIGN']:,} for BENIGN; {wk['PortScan']:.1f}% of scan flows vs {wk['BENIGN']:.1f}% of benign go to ports < 1024"),
        ("Tool fingerprint", f"Init_Win_bytes_forward == 1024 (a window value Nmap commonly uses in raw SYN probes) in {w1024['PortScan']:.1f}% of PortScan flows vs {w1024['BENIGN']:.1f}% of BENIGN flows"),
        ("Time", f"Scan share per file-order bin ranges {share.min():.1f}% to {share.max():.1f}%; scans are concentrated, not spread evenly"),
    ]
    save_table(pd.DataFrame(findings, columns=["topic", "finding"]), "eda_findings")
    for t, f in findings:
        print(f"- {t}: {f}")


if __name__ == "__main__":
    main()
