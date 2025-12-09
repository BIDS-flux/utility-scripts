#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

# ─────────────────────────────────────────────────────────────────────────────
# 0) Load data
# ─────────────────────────────────────────────────────────────────────────────
# df = pd.read_csv("/data/debug-proc/CPIP_bids/derivatives/mriqc/qc_metrics.csv")
# df_all = pd.read_csv("/Users/milton/Downloads/all3sites_qc_metrics.csv")
df_all = pd.read_csv("/Users/milton/Desktop/qc_metrics_all_sites_dwi.tsv", sep="\t")

# create a bids-id column for each of the data
df_all["bids-id"] = (
    df_all["subject"].fillna("").apply(lambda x: f"sub-{x}" if x else "")
    + df_all["session"].fillna("").apply(lambda x: f"_ses-{x}" if x else "")
    + df_all["reconstruction"].fillna("").apply(lambda x: f"_rec-{x}" if x else "")
    + df_all["run"].fillna("").apply(lambda x: f"_run-{x}" if x else "")
)

# df_all = df_all[df_all["subject"] != 3547]
df_all = df_all[df_all["suffix"] == "dwi"]
df_all = df_all[(df_all["reconstruction"] == "raw") | (df_all["reconstruction"] == "orig") | (df_all["reconstruction"].isna())]

#group by bids-id and select for each bids-id create new columns for each shell in the [0].append(df_all["bValuesEstimation"]) create a new column shell with the where the snr_per shell value is stored with the name being snr_shell_{shell_value} the per shell snr values are stored in the collumns snr_cc_shell0	snr_cc_shell1_best	snr_cc_shell1_worst	snr_cc_shell2_best	snr_cc_shell2_worst	snr_cc_shell3_best	snr_cc_shell3_worst	snr_cc_shell4_best	snr_cc_shell4_worst with the shell number correstponding to the order in the bValuesEstimation column.
def _as_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return None
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            # e.g., string like "[500, 1000, ...]" but literal_eval failed -> fallback
            x = x.strip().strip("[]")
            return [float(t) for t in x.split(",")] if x else []
    return None

rows = []
for _, row in df_all.iterrows():
    r = row.copy()
    bvals = _as_list(row.get("bValuesEstimation"))
    if not bvals:  # None or empty -> still keep row
        rows.append(r)
        continue

    # shells in order: 0 plus the estimated shells
    shells = [0.0] + [float(b) for b in bvals]

    for i, b in enumerate(shells):
        # tidy label (ints like 500 not 500.0)
        blabel = int(b) if float(b).is_integer() else b

        if i == 0:
            val0 = row.get("snr_cc_shell0")
            if pd.notna(val0):
                r[f"snr_shell_{blabel}"] = float(val0)         # preferred for b=0
                r[f"snr_shell_{blabel}_single"] = float(val0)  # optional detail
        else:
            best = row.get(f"snr_cc_shell{i}_best")
            worst = row.get(f"snr_cc_shell{i}_worst")

            # keep detailed columns if present
            if pd.notna(best):
                r[f"snr_shell_{blabel}_best"] = float(best)
            if pd.notna(worst):
                r[f"snr_shell_{blabel}_worst"] = float(worst)

            # preferred column: best > worst (since shell0 already handled)
            pref = best if pd.notna(best) else worst
            if pd.notna(pref):
                r[f"snr_shell_{blabel}"] = float(pref)

    rows.append(r)

df_wide_rows = pd.DataFrame(rows)

# ---- collapse by bids-id: take first non-null per column ---------------------
def _first_non_null(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan

value_cols = [c for c in df_wide_rows.columns if c.startswith("snr_shell_")]
keep_cols  = ["bids-id"] + [c for c in df_wide_rows.columns if c not in value_cols or c == "bids-id"]

# group; for non-snr columns you may want 'first'—adjust as needed
df_final = (df_wide_rows
            .groupby("bids-id", as_index=False)
            .agg({**{c: "first" for c in keep_cols if c != "bids-id"},
                  **{c: _first_non_null for c in value_cols}}))

# df_final now has columns:
#   bids-id, ... (original metadata 'first'), and
#   snr_shell_{b} (preferred), plus optional snr_shell_{b}_best / _worst / _single

df_all = df_final.copy()

# Output directory
os.makedirs("figures", exist_ok=True)

id_vars = ["site", "bids-id", "run", "subject"]
metric_columns = [
 'snr_shell_0',
 'snr_shell_500_best',
 'snr_shell_500_worst',
 'snr_shell_1000_best',
 'snr_shell_1000_worst',
 'snr_shell_2000_best',
 'snr_shell_2000_worst',
 'snr_shell_3000_best',
 'snr_shell_3000_worst']

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
#2.5) Box-plots (per metric) with count of plotted values
# ─────────────────────────────────────────────────────────────────────────────

for metric in metric_columns:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch  # ← minimal addition for custom legend

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
# 3) Box-plots (per metric)
# ─────────────────────────────────────────────────────────────────────────────

for metric in metric_columns:
    try:
        plt.figure(figsize=(10, 6))
        df_metric = df_melted.query("metric == @metric").copy()
        df_metric["site"] = pd.Categorical(df_metric["site"], categories=fixed_site_order, ordered=True)
        df_metric = df_metric.sort_values("site")

        # --- NEW: count plotted values per site (rows in df_metric) ------------
        counts = (df_metric.groupby("site").size()
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

        # Fix legend: one entry per site, with n
        handles, labels = ax.get_legend_handles_labels()
        # keep first occurrence of each site
        by_label = {}
        for h, l in zip(handles, labels):
            if l not in by_label:
                by_label[l] = h

        ordered_handles = [by_label[s] for s in fixed_site_order if s in by_label]
        ordered_labels  = [f"{s} (n={counts.get(s, 0)})" for s in fixed_site_order if s in by_label]

        ax.legend(ordered_handles, ordered_labels, title="Site", loc="best")

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