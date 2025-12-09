#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0) Load data
# ─────────────────────────────────────────────────────────────────────────────
# df = pd.read_csv("/data/debug-proc/CPIP_bids/derivatives/mriqc/qc_metrics.csv")
# df_all = pd.read_csv("/Users/milton/Downloads/all3sites_qc_metrics.csv")
df_all = pd.read_csv("/Users/milton/Desktop/qc_metrics_all_sites_anat.tsv", sep="\t")

# create a bids-id column for each of the data
df_all["bids-id"] = (
    df_all["subject"].fillna("").apply(lambda x: f"sub-{x}" if x else "")
    + df_all["session"].fillna("").apply(lambda x: f"_ses-{x}" if x else "")
    + df_all["reconstruction"].fillna("").apply(lambda x: f"_rec-{x}" if x else "")
    + df_all["run"].fillna("").apply(lambda x: f"_run-{x}" if x else "")
)

# df_all = df_all[df_all["subject"] != 3547]
df_all = df_all[df_all["suffix"] == "T1w"]
df_all = df_all[(df_all["reconstruction"] == "filtered") | (df_all["reconstruction"].isna())]

# Group by subject and select run 2 if available, otherwise keep run 1 asuming that a second run was necessary because of poor quality of run 1
def pick_run(group):
    if 2 in group["run"].values:
        return group[group["run"] == 2]
    else:
        return group[group["run"].isna()]

df_all = df_all.groupby("subject", group_keys=False).apply(pick_run).reset_index(drop=True)

# Output directory
os.makedirs("figures", exist_ok=True)

id_vars = ["site", "bids-id", "echo", "run", "subject"]
metric_columns = [
    "cnr", "snr_total", "snr_gm", "snr_wm", "snr_csf", "snrd_total", "cjv",
]

# Replace inf / NaN → 0  (pick another strategy if you prefer)
df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
df_all[metric_columns] = df_all[metric_columns].fillna(0)

# Tidy format
df_melted = df_all.melt(
    id_vars=id_vars, value_vars=metric_columns,
    var_name="metric", value_name="value"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) fixed site order
# ─────────────────────────────────────────────────────────────────────────────
fixed_site_order = ["montreal", "calgary", "toronto"]
site_palette = dict(zip(fixed_site_order, sns.color_palette("tab10", len(fixed_site_order))))

# ─────────────────────────────────────────────────────────────────────────────
# 2) Consistent palette (site → colour)
# ─────────────────────────────────────────────────────────────────────────────
site_palette = dict(zip(fixed_site_order, sns.color_palette("tab10", len(fixed_site_order))))
# site_palette['MYSITE'] = (0.9, 0.1, 0.4)  # ← tweak manually if you ever need to

# ─────────────────────────────────────────────────────────────────────────────
# 3) Box-plots (per metric)
# ─────────────────────────────────────────────────────────────────────────────

for metric in metric_columns:
    try:
        plt.figure(figsize=(10, 6))
        df_metric = df_melted.query("metric == @metric").copy()
        df_metric["site"] = pd.Categorical(df_metric["site"], categories=fixed_site_order, ordered=True)
        df_metric = df_metric.sort_values("site")

        # minimal addition: count plotted (non-NaN) values per site & silence FutureWarning
        counts = (df_metric.dropna(subset=["value"])
                  .groupby("site", observed=True)   # observed=True = future default, no warning
                  .size()
                  .reindex(fixed_site_order)
                  .fillna(0).astype(int))

        ax = sns.boxplot(
            data=df_metric,
            x="site", y="value",
            hue="site", hue_order=fixed_site_order, palette=site_palette,
            dodge=False
        )

        sns.stripplot(
            data=df_metric,
            x="site", y="value",
            hue="site", hue_order=fixed_site_order, palette=site_palette,
            dodge=False, jitter=True, size=3, alpha=0.5, marker='o', linewidth=0
        )

        # minimal addition: custom legend because seaborn suppresses legend when hue == x
        sites_present = [s for s in fixed_site_order if s in df_metric["site"].unique()]
        legend_handles = [
            Patch(facecolor=site_palette[s], edgecolor='none', label=f"{s} (n={counts.loc[s]})")
            for s in sites_present
        ]
        ax.legend(handles=legend_handles, title="Site", loc="best")

        plt.title(f"{metric} by Site")
        plt.xlabel("Site")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"figures/{metric}_boxplot_by_site.png", dpi=600)
        plt.close()

    except Exception as e:
        print(f"⚠️  Skipping boxplot for {metric}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4) Point-plots (per metric, by subject)
# ─────────────────────────────────────────────────────────────────────────────
for metric in metric_columns:
    try:
        plt.figure(figsize=(14, 6))
        df_metric = df_melted.query("metric == @metric").copy()
        df_metric["site"] = pd.Categorical(df_metric["site"], categories=fixed_site_order, ordered=True)
        df_metric = df_metric.sort_values("site")

        sns.pointplot(
            data=df_metric,
            x="subject", y="value",
            hue="site", hue_order=fixed_site_order, palette=site_palette,
            dodge=0.5 if len(fixed_site_order) > 1 else False, markers="o"
        )
        plt.xticks(rotation=90)
        plt.title(f"{metric} by Subject (Grouped by Site)")
        plt.xlabel("Subject")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"figures/{metric}_by_subject.png", dpi=600)
        plt.close()

    except ZeroDivisionError:
        print(f"⚠️  Skipping {metric}: zero-division error")
    except (AttributeError, ValueError) as e:
        print(f"⚠️  Skipping {metric}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Value-label plots – each point labelled with its bids-id
# ─────────────────────────────────────────────────────────────────────────────
text_offset = 0.05
font_size = 5
label_angle = 15

try:
    for metric in metric_columns:
        df_metric = df_melted.query("metric == @metric").copy()
        df_metric["site"] = pd.Categorical(df_metric["site"], categories=fixed_site_order, ordered=True)
        df_metric = df_metric.sort_values("site")

        # Ensure consistent subject order
        df_metric["subject_order"] = (
            df_metric["subject"]
            .astype(str)
            .rank(method="dense", ascending=True)
            .astype(int)
        )

        df_metric["x_pos"] = df_metric["subject_order"]

        plt.figure(figsize=(18, 6))
        ax = sns.scatterplot(
            data=df_metric,
            x="x_pos",
            y="value",
            hue="site",
            hue_order=fixed_site_order,
            palette=site_palette,
            s=70,
            edgecolor="black",
            linewidth=0.3,
            zorder=3,
        )

        y_offset = (df_metric["value"].max() - df_metric["value"].min()) * 0.01

        for _, row in df_metric.iterrows():
            ax.text(
                row["x_pos"] + text_offset,
                row["value"] + y_offset,
                row["bids-id"],
                rotation=label_angle,
                ha="left",
                va="bottom",
                fontsize=font_size,
                color="black",
                zorder=4
            )

        ax.set_title(f"{metric}: Individual Values with BIDS ID Labels")
        ax.set_xlabel("Subjects (ordered)")
        ax.set_ylabel(metric)
        ax.set_xticks([])  # clean look
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"figures/{metric}_values_with_labels.png", dpi=600)
        plt.close()

except Exception as e:
    print(f"⚠️  Skipping value-label plot for {metric}: {e}")