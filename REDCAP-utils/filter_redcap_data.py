import pandas as pd
import numpy as np

df = pd.read_csv("/Users/milton/Downloads/CPIPPRODUCTION_DATA_2025-09-11_1436.csv")

# cols
RECORD_COL = "record_id"
SUBJECT_COL = "study_id"

# fill within each record_id using first forward then backward fill
df[SUBJECT_COL] = (
    df.groupby(RECORD_COL)[SUBJECT_COL]
      .transform(lambda s: s.ffill().bfill())
)

AGE_COL = "youth_age_y"
# fill within each record_id using first forward then backward fill
df[AGE_COL] = (
    df.groupby(RECORD_COL)[AGE_COL]
      .transform(lambda s: s.ffill().bfill())
)

#filter by date of mri_date
df = df[
    (df["mri_date"] >= '2025-08-01') &
    (df["mri_scan_stage"] >= 2) &
    (pd.isna(df["mri_scan_stage_isue"]))
]

df.to_csv("table_for_merv.csv")
