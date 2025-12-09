# ==== Average per-ROI correlation overlay (one figure per input df) ====
import os
import logging
from typing import Sequence, Iterable, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def setup_plot_logging(level=logging.INFO, to_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("avg_corr_overlay")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        if to_file:
            fh = logging.FileHandler(to_file); fh.setFormatter(fmt); logger.addHandler(fh)
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

def _as_canonical_img(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)

def _make_roi_to_label_map(parcellation_labels: Sequence[str]) -> dict:
    # Assumes atlas labels are 1..N in the same order as parcellation_labels
    return {roi: i+1 for i, roi in enumerate(parcellation_labels)}

def _choose_informative_slices(vol3d: np.ndarray) -> dict:
    absvol = np.nan_to_num(np.abs(vol3d), nan=0.0)
    return {
        "sagittal": int(np.argmax(absvol.sum(axis=(1,2)))),  # x
        "coronal":  int(np.argmax(absvol.sum(axis=(0,2)))),  # y
        "axial":    int(np.argmax(absvol.sum(axis=(0,1)))),  # z
    }

def _paint_volume(atlas_data: np.ndarray, roi_to_label: dict, roi_to_value: dict, fill=np.nan) -> np.ndarray:
    out = np.full(atlas_data.shape, fill, dtype=float)
    for roi, val in roi_to_value.items():
        lab = roi_to_label.get(roi)
        if lab is not None:
            out[atlas_data == lab] = val
    return out

def _compute_vmin_vmax(
    values: np.ndarray,
    percentile_clip: Optional[Tuple[float, float]] = None,
    symmetric: bool = False
) -> Tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (-1.0, 1.0) if symmetric else (-0.5, 0.5)
    if percentile_clip is not None:
        lo, hi = np.nanpercentile(v, [percentile_clip[0], percentile_clip[1]])
        vmin, vmax = float(lo), float(hi)
    else:
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if vmin == vmax:  # avoid zero span
        eps = max(1e-6, abs(vmin) * 1e-3, 1e-3)
        vmin, vmax = vmin - eps, vmax + eps
    return vmin, vmax

# ----------------- main function -----------------
def plot_average_roi_correlation_overlay(
    run_corr_df: pd.DataFrame,
    parcellation_labels: Sequence[str],
    atlas_path: str,
    out_path: str,
    title_fields: Iterable[str] = ("site","task","session"),
    weight_by: Literal["none","n_used","n_used_roi"] = "none",
    percentile_clip: Optional[Tuple[float, float]] = None,  # e.g., (2,98) or None
    symmetric: bool = False,        # True → color scale symmetric around 0
    cmap: Optional[str] = "coolwarm",
    vmax = None,   # color scale is [-vmax, +vmax]
    save_nifti: bool = False,       # also write a NIfTI with the averaged values painted
    logger: Optional[logging.Logger] = None,
    fixed_vmax: Optional[float] = None,                     # e.g., 0.5 -> [-0.5, 0.5] when symmetric=True
    fixed_range: Optional[Tuple[float, float]] = None,      # e.g., (-0.4, 0.6)
) -> str:
    """
    Compute mean r per ROI over the provided rows, paint onto Schaefer atlas,
    and save a single 3-panel figure (sagittal/coronal/axial).

    Parameters
    ----------
    run_corr_df : DataFrame
        Must contain columns ['roi','r'] (others optional). You typically pass
        a subset already filtered to one cohort/group you want to summarize.
    parcellation_labels : list[str]
        ROI names ordered exactly like label integers 1..N in the atlas.
    atlas_path : str
        Path to Schaefer atlas NIfTI (e.g., Schaefer2018_200Parcels_...2mm.nii.gz).
    out_path : str
        Where to save the PNG figure.
    title_fields : tuple[str]
        Columns to display in the figure title (if present, unique values shown).
    weight_by : {"none","n_used","n_used_roi"}
        If "n_used" or "n_used_roi" exists in run_corr_df, compute a weighted mean per ROI.
    percentile_clip : (low, high) or None
        Optional robust scaling of color range.
    symmetric : bool
        If True, force colorbar to be symmetric around zero.
    cmap : str or None
        Matplotlib colormap name.
    vmax : bool or a tuple
        If set, forces the intensity range to be used by the plot
    save_nifti : bool
        If True, also saves painted NIfTI next to the PNG.
    """
    if logger is None:
        logger = setup_plot_logging(logging.INFO)

    # Validate
    if "roi" not in run_corr_df.columns or "r" not in run_corr_df.columns:
        raise ValueError("run_corr_df must contain columns 'roi' and 'r'.")

    # Averaging
    df_use = run_corr_df.copy()
    if weight_by in ("n_used","n_used_roi") and weight_by in df_use.columns:
        logger.info("Computing weighted mean r per ROI using weights=%s", weight_by)
        w = df_use[weight_by].astype(float).fillna(0.0)
        # Guard against zero weight
        df_use["_rw"] = df_use["r"] * w
        avg = (
            df_use.groupby("roi", as_index=True)
                  .agg(r_sum=("_rw","sum"), w_sum=(weight_by,"sum"))
        )
        avg["r_mean"] = np.where(avg["w_sum"] > 0, avg["r_sum"] / avg["w_sum"], np.nan)
        roi_to_r = avg["r_mean"].to_dict()
    else:
        logger.info("Computing unweighted mean r per ROI")
        roi_to_r = df_use.groupby("roi")["r"].mean().to_dict()

    # Report coverage / missing ROIs
    have = set(roi_to_r.keys())
    want = set(parcellation_labels)
    missing = sorted(list(want - have))
    if missing:
        logger.warning("Missing %d ROIs in input (first 5): %s", len(missing), missing[:5])

    # Load atlas & prep
    logger.info("Loading atlas: %s", atlas_path)
    atlas_img = _as_canonical_img(nib.load(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_to_label = _make_roi_to_label_map(parcellation_labels)

    # Paint volume with averaged r
    vol = _paint_volume(atlas_data, roi_to_label, roi_to_r, fill=np.nan)

    # Choose slices & scale
    slices = _choose_informative_slices(vol)
    vals = np.array(list(roi_to_r.values()), dtype=float)
    if vmax:
        vmin, vmax = vmax
    else:
        vmin, vmax = _compute_vmin_vmax(vals, percentile_clip=percentile_clip, symmetric=symmetric)
    logger.info("Avg r range: [%.4f, %.4f] | color scale: [%.4f, %.4f] | slices x/y/z=%s/%s/%s",
                np.nanmin(vals) if vals.size else np.nan, np.nanmax(vals) if vals.size else np.nan,
                vmin, vmax, slices["sagittal"], slices["coronal"], slices["axial"])

    # Compose title from requested fields (if present)
    pieces = []
    for c in title_fields:
        if c in run_corr_df.columns:
            uniq = pd.unique(run_corr_df[c].astype(str))
            if len(uniq) == 1:
                pieces.append(f"{c}={uniq[0]}")
            else:
                pieces.append(f"{c}=<mixed>")
    title = " | ".join(pieces) if pieces else "Average ROI correlation"

    # Plot 3-panel
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(title, fontsize=10)

    x = slices["sagittal"]; sag = np.rot90(vol[x, :, :])
    y = slices["coronal"];  cor = np.rot90(vol[:, y, :])
    z = slices["axial"];    axi = np.rot90(vol[:, :, z])

    im0 = axs[0].imshow(sag, vmin=vmin, vmax=vmax, cmap=cmap); axs[0].set_title(f"Sagittal x={x}"); axs[0].axis("off")
    im1 = axs[1].imshow(cor, vmin=vmin, vmax=vmax, cmap=cmap); axs[1].set_title(f"Coronal y={y}");  axs[1].axis("off")
    im2 = axs[2].imshow(axi, vmin=vmin, vmax=vmax, cmap=cmap); axs[2].set_title(f"Axial z={z}");    axs[2].axis("off")

    cbar = fig.colorbar(im2, ax=axs, shrink=0.8, location="right")
    cbar.set_label("mean r (Pearson)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)

    if save_nifti:
        nii_path = os.path.splitext(out_path)[0] + ".nii.gz"
        nib.save(nib.Nifti1Image(vol, affine=atlas_img.affine, header=atlas_img.header), nii_path)
        logger.info("Saved %s", nii_path)

    return out_path

# Define site configurations as a dictionary
site_configs = {
    "calgary": {
        "bids_path": "/data/debug-proc/CPIP_bids",
        "tedana_dir": "/data/debug-proc/CPIP_bids/derivatives/tedan2/",
        "fmriprep_dir": "/data/debug-proc/CPIP_bids/derivatives/fmriprep2/",
        "subjects": "2532,3398,3136,2976,3328",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", None]}
    },
    "toronto": {
        "bids_path": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/",
        "tedana_dir": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/tedana/",
        "fmriprep_dir": "/data/debug-proc/CPIP_fMRI_July042025/CPIP_fMRI_July042025/derivatives/fmriprep2/",
        "subjects": "3536,3675,3504,3592,3803",
        "ses": None,
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
    },
    "montreal": {
        "bids_path": "/data/debug-proc/CPIP_bids_montreal/",
        "tedana_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/tedana2/",
        "fmriprep_dir": "/data/debug-proc/CPIP_bids_montreal/derivatives/fmriprep/",
        "subjects": "1118,1312,1466,1760,2117",
        "ses": "1a",
        "bids_filters": {"task": ["laluna", "partlycloudy", "rest"], "run": ["1", "2", "3", None]}
    }
}

site_configs["calgary"]["parcorr_ll"] = df["site"].eq("calgary") & df["task"].isin(["laluna"])
site_configs["calgary"]["parcorr_pc"] = df["site"].eq("calgary") & df["task"].isin(["partlycloudy"])
site_configs["montreal"]["parcorr_ll"] = df["site"].eq("montreal") & df["task"].isin(["laluna"])
site_configs["montreal"]["parcorr_pc"] = df["site"].eq("montreal") & df["task"].isin(["partlycloudy"])
site_configs["toronto"]["parcorr_ll"] = df["site"].eq("toronto") & df["task"].isin(["rest"]) & df["run"].eq(2)
site_configs["toronto"]["parcorr_pc"] = df["site"].eq("toronto") & df["task"].isin(["rest"]) & df["run"].eq(1)

acq_masks = ["parcorr_ll","parcorr_pc"]

parcellation_labels_path = "/data/debug-proc/Schaefer200_labels_7network.csv"

parcellation_labels = pd.read_csv(parcellation_labels_path,sep=',')

parcellation_labels = list(parcellation_labels["ROI Name"].values)

for site in site_configs:
    for acq in acq_masks:
        #get the df
        corr_df = _read_tsv_max_precision(f"/data/debug-proc/corr_{site}_{acq}.tsv")
        logger = setup_plot_logging(logging.INFO, to_file="/data/debug-proc/corr_overlay_avg.log")

        saved = plot_average_roi_correlation_overlay(
            run_corr_df=corr_df,                                  # your table
            parcellation_labels=parcellation_labels,                  # your 200-name list
            atlas_path="/home/milton.camachocamach/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
            out_path=f"/data/debug-proc/corr_figs_{site}_{acq}/corr_figs_{site}_{acq}_avg_no_high_motion_runs.png",
            title_fields=("site","task","session"),                        
            # weight_by="n_used",
            vmax = (-0.23,0.23),
            symmetric=False,
            save_nifti=True,                                          # also write a NIfTI
            percentile_clip=None,
            cmap="coolwarm",
            logger=logger,
        )
