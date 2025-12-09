# ====== Pairwise correlation overlays on Schaefer atlas ======
import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Sequence, Iterable, Optional
import itertools

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

def setup_plot_logging(level=logging.INFO, to_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("corr_overlay")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

def _as_canonical_img(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return image in RAS (closest canonical) orientation for sane slicing/plotting."""
    return nib.as_closest_canonical(img)

def _make_roi_to_label_map(parcellation_labels: Sequence[str]) -> dict:
    """
    Map ROI name -> atlas integer label.
    Assumes atlas uses 1..N labels in the same order as parcellation_labels.
    """
    return {roi: i+1 for i, roi in enumerate(parcellation_labels)}

def _pair_key(row, within: Iterable[str]) -> tuple:
    return tuple(row[c] for c in within) + (row["subject_a"], row["subject_b"])

def _choose_informative_slices(vol3d: np.ndarray) -> dict:
    """
    Pick one slice per axis where |vol| energy is maximal.
    Returns dict with indices for 'sagittal' (x), 'coronal' (y), 'axial' (z).
    """
    absvol = np.nan_to_num(np.abs(vol3d), nan=0.0)
    # Sum of absolute values over planes
    sag_scores = absvol.sum(axis=(1,2))  # x index
    cor_scores = absvol.sum(axis=(0,2))  # y index
    axi_scores = absvol.sum(axis=(0,1))  # z index
    return {
        "sagittal": int(np.argmax(sag_scores)),
        "coronal":  int(np.argmax(cor_scores)),
        "axial":    int(np.argmax(axi_scores)),
    }

def _paint_volume_for_pair(
    atlas_data: np.ndarray,
    roi_to_label: dict,
    roi_to_value: dict,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Create a 3D volume where each parcel's voxels are set to the pair's correlation value.
    Voxels not belonging to any parcel (label 0) are set to `fill`.
    """
    out = np.full(atlas_data.shape, fill, dtype=float)
    # Only iterate ROIs we actually have values for
    for roi, r in roi_to_value.items():
        lab = roi_to_label.get(roi)
        if lab is None:
            continue
        out[atlas_data == lab] = r
    return out

def plot_pairwise_correlation_overlays(
    run_corr_df: pd.DataFrame,
    parcellation_labels: Sequence[str],
    atlas_path: str,
    out_dir: str,
    within: Iterable[str] = ("site","task","session"),
    vmax: float = 1.0,   # color scale is [-vmax, +vmax]
    save_nifti: bool = False,   # also save corr volume as NIfTI per pair
    logger: Optional[logging.Logger] = None,
) -> list[str]:
    """
    For each unique (within..., subject_a, subject_b) in run_corr_df,
    generate a 3-panel figure (sagittal/coronal/axial) with parcel correlations overlaid.

    Returns a list of saved PNG paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    if logger is None:
        logger = setup_plot_logging(logging.INFO)

    # Validate columns
    needed = set(within) | {"roi","subject_a","subject_b","r"}
    missing = needed - set(run_corr_df.columns)
    if missing:
        raise ValueError(f"run_corr_df missing columns: {sorted(missing)}")

    # Load atlas and bring to canonical RAS orientation
    logger.info("Loading atlas: %s", atlas_path)
    atlas_img = _as_canonical_img(nib.load(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(int)  # labels as ints

    # Basic atlas sanity
    pos_labels = sorted([int(x) for x in np.unique(atlas_data) if x > 0])
    logger.info("Atlas shape=%s | positive labels: %d..%d (count=%d)",
                atlas_data.shape, pos_labels[0] if pos_labels else 0,
                pos_labels[-1] if pos_labels else 0, len(pos_labels))

    # Build ROI->label map
    roi_to_label = _make_roi_to_label_map(parcellation_labels)
    max_expected_label = len(parcellation_labels)
    if not pos_labels or pos_labels[-1] < max_expected_label:
        logger.warning("Atlas max label (%s) < number of ROIs (%s). "
                       "Check that atlas matches your label set.",
                       pos_labels[-1] if pos_labels else None, max_expected_label)

    # Prep groups
    group_cols = list(within) + ["subject_a","subject_b"]
    saved = []

    # Iterate pairs
    for gkey, subdf in run_corr_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, gkey))
        # Build roi -> r mapping for this pair (mean if duplicates)
        vals = (subdf.groupby("roi")["r"].mean()).to_dict()

        # Debug: check coverage
        have = set(vals.keys())
        want = set(parcellation_labels)
        missing_rois = sorted(list(want - have))
        extra_rois   = sorted(list(have - want))
        if missing_rois:
            logger.debug("Missing %d ROIs for pair %s (first 5): %s",
                         len(missing_rois), key_dict, missing_rois[:5])
        if extra_rois:
            logger.debug("Found %d unknown ROIs for pair %s (first 5): %s",
                         len(extra_rois), key_dict, extra_rois[:5])

        # Paint the 3D correlation volume
        corr_vol = _paint_volume_for_pair(atlas_data, roi_to_label, vals, fill=np.nan)
        rmin = np.nanmin(corr_vol); rmax = np.nanmax(corr_vol)
        logger.info("Pair %s | r-range: [%.3f, %.3f]", key_dict, rmin, rmax)

        # Choose slices
        slices = _choose_informative_slices(corr_vol)
        logger.info("Pair %s | chosen slices (x/y/z): %s/%s/%s",
                    key_dict, slices["sagittal"], slices["coronal"], slices["axial"])

        # Create a 3-panel figure
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        fig.suptitle(f"{key_dict} | overlay of ROI correlations", fontsize=10)

        # Sagittal (x) — display y (vertical) by z (horizontal)
        x = slices["sagittal"]
        sag = np.rot90(corr_vol[x, :, :])  # rotate for nicer orientation
        im0 = axs[0].imshow(sag, vmin=-vmax, vmax=vmax)
        axs[0].set_title(f"Sagittal x={x}")
        axs[0].axis("off")

        # Coronal (y) — display x by z
        y = slices["coronal"]
        cor = np.rot90(corr_vol[:, y, :])
        im1 = axs[1].imshow(cor, vmin=-vmax, vmax=vmax)
        axs[1].set_title(f"Coronal y={y}")
        axs[1].axis("off")

        # Axial (z) — display x by y
        z = slices["axial"]
        axi = np.rot90(corr_vol[:, :, z])
        im2 = axs[2].imshow(axi, vmin=-vmax, vmax=vmax)
        axs[2].set_title(f"Axial z={z}")
        axs[2].axis("off")

        # One shared colorbar (diverging scale, default colormap)
        cbar = fig.colorbar(im2, ax=axs, shrink=0.8, location="right")
        cbar.set_label("r (Pearson)")

        # File name
        safe = "_".join([str(k) for k in gkey])
        png_path = os.path.join(out_dir, f"corr_overlay_{safe}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        saved.append(png_path)
        logger.info("Saved %s", png_path)

        # Optional: also save NIfTI of the correlation volume
        if save_nifti:
            corr_img = nib.Nifti1Image(corr_vol, affine=atlas_img.affine, header=atlas_img.header)
            nii_path = os.path.join(out_dir, f"corr_overlay_{safe}.nii.gz")
            nib.save(corr_img, nii_path)
            logger.info("Saved %s", nii_path)

    if not saved:
        logger.warning("No figures were saved — did run_corr_df have matching groups?")
    return saved

site_configs["calgary"]["parcorr_ll"] = df["site"].eq("calgary") & df["task"].isin(["laluna"])
site_configs["calgary"]["parcorr_pc"] = df["site"].eq("calgary") & df["task"].isin(["partlycloudy"])
site_configs["montreal"]["parcorr_ll"] = df["site"].eq("montreal") & df["task"].isin(["laluna"])
site_configs["montreal"]["parcorr_pc"] = df["site"].eq("montreal") & df["task"].isin(["partlycloudy"])
site_configs["toronto"]["parcorr_ll"] = df["site"].eq("toronto") & df["task"].isin(["rest"]) & df["run"].eq(2)
site_configs["toronto"]["parcorr_pc"] = df["site"].eq("toronto") & df["task"].isin(["rest"]) & df["run"].eq(1)

acq_masks = ["parcorr_ll","parcorr_pc"]

#read the parcelation labels
parcellation_labels_path = "/data/debug-proc/Schaefer200_labels_7network.csv"

parcellation_labels = pd.read_csv(parcellation_labels_path,sep=',')

parcellation_labels = list(parcellation_labels["ROI Name"].values)

for site in site_configs:
    for acq in acq_masks:
        #get the df
        corr_df = _read_tsv_max_precision(f"/data/debug-proc/corr_{site}_{acq}.tsv")
        logger = setup_plot_logging(logging.INFO, to_file="/data/debug-proc/corr_overlay.log")

        saved = plot_pairwise_correlation_overlays(
            run_corr_df=corr_df,                                  # your table
            parcellation_labels=parcellation_labels,                  # your 200-name list
            atlas_path="/home/milton.camachocamach/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
            out_dir=f"/data/debug-proc/corr_figs_{site}_{acq}",
            within=("site","task","session"),                         # groups that define “same acquisition”
            vmax=0.5,                                                 # scale [-1, 1]
            save_nifti=True,                                          # also write a NIfTI per pair
            logger=logger,
        )
