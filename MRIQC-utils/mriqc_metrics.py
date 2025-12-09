import json
import pandas as pd
from bids import BIDSLayout

mriqc_data = {
    "toronto": {"site": "toronto",
    "directory": "/Users/milton/Downloads/mriqc_jsons_toronto"},
    "montreal": {"site": "montreal",
    "directory": "/Users/milton/Downloads/mriqc_jsons_montreal"},
    "calgary": {"site": "calgary",
    "directory": "/Users/milton/Desktop/test-clone/CPIP/mriqc"},
}

merged_df_list = []

for site in mriqc_data.keys():
    print(f"Processing site: {site}")
    layout = BIDSLayout(mriqc_data[site]["directory"], validate=False)

    # List of JSON file paths
    json_files = layout.get(datatype=["anat","dwi","func"], extension='.json')

    files = []
    for f in json_files:
        if "fmriprep" not in str(f.path) and "tedana" not in str(f.path) and "MTR" not in str(f.path) and "MP2RAGE" not in str(f.path) and "timeseries" not in str(f.path):
            files.append(f)

    rows = []

    for bidsfile in files:
        with open(bidsfile.path, 'r') as bf:
            data = json.load(bf)
            bf_entities = bidsfile.get_entities()
            merged_entities_qc = mriqc_data[site] |  bf_entities | data
            rows.append(merged_entities_qc)  # Add BIDS entities and qc to the row

    # Convert list of dicts to DataFrame
    merged_df_list.append(pd.DataFrame(rows))



output_csv = f"qc_metrics_all_sites.tsv"
all_sites = pd.concat(merged_df_list, ignore_index=True)
all_sites.drop(["bids_meta","global"],axis=1,inplace=True)
all_sites.to_csv(output_csv, sep='\t', index=False)
print(f"Saved QC metrics to {output_csv}")

# divide into 3 data types
all_sites_anat = all_sites[all_sites['datatype'] == 'anat']
all_sites_func = all_sites[all_sites['datatype'] == 'func']
all_sites_dwi = all_sites[all_sites['datatype'] == 'dwi']
# Drop columns with all NaN values
all_sites_anat = all_sites_anat.dropna(axis=1, how='all')
all_sites_func = all_sites_func.dropna(axis=1, how='all')
all_sites_dwi = all_sites_dwi.dropna(axis=1, how='all')

all_sites_anat.to_csv("qc_metrics_all_sites_anat.tsv", sep='\t', index=False)
all_sites_func.to_csv("qc_metrics_all_sites_func.tsv", sep='\t', index=False)
all_sites_dwi.to_csv("qc_metrics_all_sites_dwi.tsv", sep='\t', index=False)
