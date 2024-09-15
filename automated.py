import subprocess
import os
import sys

# Path to the directory containing the .tiff files
path = r"E:\WHK\V10082\output_10082_1"
# Path to the script to run
script_to_run = 'result_main.py'

# List all .tiff files in the directory
tiff_files = [f for f in os.listdir(path) if f.endswith('.tiff')]

# Iterate through each .tiff file
for tiff_file in tiff_files:
    print(f"Processing file: {tiff_file}")

    # Run the target Python script using the current Python interpreter
    process = subprocess.Popen([sys.executable, script_to_run, tiff_file])

    # Wait for the script to finish
    process.wait()

    print(f"Finished processing file: {tiff_file}\n")

print("All files have been processed.")
