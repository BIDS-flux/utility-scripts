import numpy as np
import pandas as pd
from typing import Sequence, Iterable
import logging
import itertools
import ast

# ── Logging helper ──────────────────────────────────────────────────────────
def setup_corr_logging(level=logging.INFO, to_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("roi_corr")
    logger.setLevel(level)
    # avoid duplicate handlers if re-run in notebooks
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

def _read_tsv_max_precision(path: str) -> pd.DataFrame:
    """
    Read TSV with ROI_* columns parsed through Decimal→float (faithful round-trip),
    leaving other columns as-is. Falls back to NaN on blanks.
    """
    # Grab header to detect ROI columns
    cols = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    roi_cols = [c for c in cols if c.startswith("ROI_")]

    # Read everything as string first to avoid premature float casting
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Convert ROI columns through Decimal → float64 (IEEE-754 double)
    for c in roi_cols:
        df[c] = df[c].map(lambda x: float(Decimal(x)) if isinstance(x, str) and x != "" else np.nan)

    # (Optional) try to coerce any non-ROI numeric columns
    for c in df.columns:
        if c not in roi_cols:
            # best-effort numeric, but don't break non-numerics
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

parcellation_labels_txt = "/data/debug-proc/Schaefer200_labels_from_nilearn.txt"

with open(parcellation_labels_txt, "r", encoding="utf-8") as f:
    parcellation_labels = [ast.literal_eval(line.strip()).decode("utf-8")
              for line in f if line.strip()]


ID_GROUP_DEFAULT: tuple[str, ...] = ("site", "task", "session", "run")
VOL_COL   = "vol_idx"
SUBJ_COL  = "subject"
MOTIONCOL = "motion_outlier_flag"

# ── Safe correlation ────────────────────────────────────────────────────────
def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    # If any NaNs sneak in, result becomes NaN; treat as 0.0 for robustness
    if np.isnan(x).any() or np.isnan(y).any():
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    sx = x.std(ddof=1); sy = y.std(ddof=1)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float((x * y).sum() / ((x.size - 1) * sx * sy))

def compute_pairwise_roi_correlations_all(
    df: pd.DataFrame,
    parcellation_labels: Sequence[str],
    within: Iterable[str] = ID_GROUP_DEFAULT,
    vol_col: str = VOL_COL,
    subject_col: str = SUBJ_COL,
    motion_col: str = MOTIONCOL,
    identifier_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute ROI-wise Pearson correlations for all subject pairs within each group
    (default group: site, task, session, run).

    Adds two columns:
      - subject_a_identifier
      - subject_b_identifier
    taken from `identifier_col` (e.g., "site"). If not provided/available -> NaN.

    Rules:
    - Align by vol_idx (inner-join).
    - Exclude any TR where either subject has motion_outlier_flag == 1.
    - If TR counts differ after motion filtering, cap by the lowest available count.
    - n_used is computed once per subject pair and shared across all ROIs in that pair.
    - If <2 usable TRs or zero variance, r = 0.0.

    Returns a DataFrame with columns:
      [within...], roi, subject_a, subject_b,
      n_A_total, n_B_total, n_intersection_total,
      n_A_ok, n_B_ok, n_common_ok, n_min_ok, n_used, r,
      subject_a_identifier, subject_b_identifier
    """
    within = tuple(within)
    needed = set(within) | {vol_col, subject_col, motion_col} | set(parcellation_labels)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Copy & normalize motion flag to {0,1}
    work = df.copy()
    work[motion_col] = (work[motion_col].astype(float) >= 1.0).astype(int)

    rows = []

    for group_vals, g in work.groupby(list(within), dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_key = dict(zip(within, group_vals))

        subjects = sorted(g[subject_col].unique().tolist())
        if len(subjects) < 2:
            continue

        # Per-subject identifier lookup (e.g., site per subject)
        subj_identifier: dict = {}
        if identifier_col is not None and identifier_col in g.columns:
            for s in subjects:
                vals = g.loc[g[subject_col] == s, identifier_col].dropna().unique()
                subj_identifier[s] = vals[0] if len(vals) else np.nan
        else:
            subj_identifier = {s: np.nan for s in subjects}

        # Pre-slices per subject to avoid repeated filtering
        subj_slices = {}
        subj_counts = {}
        good_vols = {}
        for s in subjects:
            gs = (
                g.loc[g[subject_col] == s, [vol_col, motion_col] + list(parcellation_labels)]
                .rename(columns={motion_col: "motion", **{roi: f"{roi}" for roi in parcellation_labels}})
                .copy()
            )
            subj_slices[s] = gs
            subj_counts[s] = {
                "total": int(gs[vol_col].nunique()),
                "ok":    int(gs.loc[gs["motion"] == 0, vol_col].nunique()),
            }
            good_vols[s] = set(gs.loc[gs["motion"] == 0, vol_col].unique())

        # All unique pairs without order
        for a, b in itertools.combinations(subjects, 2):
            gA = subj_slices[a].rename(columns={"motion": "motion_a", **{roi: f"{roi}_a" for roi in parcellation_labels}})
            gB = subj_slices[b].rename(columns={"motion": "motion_b", **{roi: f"{roi}_b" for roi in parcellation_labels}})

            nA_total = subj_counts[a]["total"]
            nB_total = subj_counts[b]["total"]
            nA_ok    = subj_counts[a]["ok"]
            nB_ok    = subj_counts[b]["ok"]

            # Align by vol_idx (inner join)
            merged = pd.merge(gA, gB, on=vol_col, how="inner").sort_values(vol_col)
            n_intersection_total = int(merged[vol_col].nunique())

            # Exclude motion-outlier TRs for either subject
            merged = merged[(merged["motion_a"] == 0) & (merged["motion_b"] == 0)]

            # Common OK TRs after motion filtering
            common_ok = sorted(good_vols[a].intersection(good_vols[b]))
            n_common_ok = len(common_ok)

            # Cap by the lowest per-subject OK count
            n_min_ok = min(nA_ok, nB_ok)

            # TRs actually used
            n_used = min(n_common_ok, n_min_ok)

            if n_used < 2:
                # Not enough usable frames; emit r=0 for all ROIs
                for roi in parcellation_labels:
                    rows.append({
                        **group_key,
                        "roi": roi,
                        "subject_a": a,
                        "subject_b": b,
                        "n_A_total": nA_total,
                        "n_B_total": nB_total,
                        "n_intersection_total": n_intersection_total,
                        "n_A_ok": nA_ok,
                        "n_B_ok": nB_ok,
                        "n_common_ok": n_common_ok,
                        "n_min_ok": n_min_ok,
                        "n_used": n_used,
                        "r": 0.0,
                        f"subject_a_{identifier_col}": subj_identifier.get(a, np.nan),
                        f"subject_b_{identifier_col}": subj_identifier.get(b, np.nan),
                    })
                continue

            # Select the first n_used common TRs (deterministic)
            used_vols = set(common_ok[:n_used])
            used = merged[merged[vol_col].isin(used_vols)].sort_values(vol_col)

            # ROI-wise correlations (n_used constant within the pair)
            for roi in parcellation_labels:
                xa = used[f"{roi}_a"].to_numpy()
                xb = used[f"{roi}_b"].to_numpy()
                r = _safe_pearsonr(xa, xb)

                rows.append({
                    **group_key,
                    "roi": roi,
                    "subject_a": a,
                    "subject_b": b,
                    "n_A_total": nA_total,
                    "n_B_total": nB_total,
                    "n_intersection_total": n_intersection_total,
                    "n_A_ok": nA_ok,
                    "n_B_ok": nB_ok,
                    "n_common_ok": n_common_ok,
                    "n_min_ok": n_min_ok,
                    "n_used": n_used,
                    "r": r,
                    f"subject_a_{identifier_col}": subj_identifier.get(a, np.nan),
                    f"subject_b_{identifier_col}": subj_identifier.get(b, np.nan),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(list(within) + ["roi", "subject_a", "subject_b"]).reset_index(drop=True)
    return out

# read the parcellation labels
parcellation_labels_path = "/data/debug-proc/Schaefer200_labels_7network.csv"

parcellation_labels = pd.read_csv(parcellation_labels_path,sep=',')

parcellation_labels = list(parcellation_labels["ROI Name"].values)

ID_GROUP_DEFAULT: tuple[str, ...] = ("site", "task", "session", "run")
VOL_COL   = "vol_idx"
SUBJ_COL  = "subject"
MOTIONCOL = "motion_outlier_flag"

# read the full tsv
df = _read_tsv_max_precision("/data/debug-proc/censored_mega_timeseries_motion_outliers.tsv")

configs = {}
configs["ll"] = df["task"].isin(["laluna"])
configs["pc"] = df["task"].isin(["partlycloudy"])

# create the unique grouping of task and session and run the correlation for each of the tasks
for task in configs.keys():
    df_acq = df.loc[configs[task]]
    accross_site_corr = compute_pairwise_roi_correlations_all(df_acq, parcellation_labels, within=("task","session"), identifier_col="site")
    accross_site_corr.to_csv(f"/data/debug-proc/across_sites_corr_{task}.tsv",sep="\t")



