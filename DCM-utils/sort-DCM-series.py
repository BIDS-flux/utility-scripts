import os
import glob
import pydicom
import shutil
from collections import defaultdict
import argparse

def process_dicom_files(input_folder, output_folder, sort_by):
    # List all DICOM files in the input folder
    dicom_files = glob.glob(os.path.join(input_folder, '*'))

    # Dictionary to group files by sort_by
    grouped_files = defaultdict(list)

    # Extract sort_by and group files
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            grouped_by = dcm.get(sort_by, "Unknown")
            grouped_files[grouped_by].append(f)
        except Exception as e:
            print(f"Skipping invalid DICOM: {f} - {e}")
	
    # Process each series separately
    for grouped_by, files in grouped_files.items():
        try:
            series_dir = os.path.join(output_folder, str(grouped_by).replace(" ", "_"))  # Replace spaces for directory naming
        except:
            series_dir = os.path.join(output_folder, str(grouped_by))
        os.makedirs(series_dir, exist_ok=True)
        
        for f in files:
            shutil.copyfile(f, os.path.join(series_dir, os.path.basename(f)))
    
    print(f"DICOM files have been grouped by {sort_by} and copied to '{output_folder}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group DICOM files by SeriesDescription.")
    parser.add_argument("input_folder", help="Path to the input folder containing DICOM files.")
    parser.add_argument("output_folder", help="Path to the output folder where grouped files will be stored.")
    parser.add_argument("--sort_by", default="SeriesDescription", help="Attribute to sort DICOM files by (default: SeriesDescription).")
    args = parser.parse_args()
    
    process_dicom_files(args.input_folder, args.output_folder, args.sort_by)
