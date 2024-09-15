import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage.measure import label, regionprops


# Function to load void information from a CSV file
def load_void_info(filename):
    void_info = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            void_info.append({
                'x': float(row['x']),
                'y': float(row['y']),
                'size': float(row['size'])
            })
    return void_info


# Function to calculate the Euclidean distance between two voids
def distance(v1, v2):
    return np.sqrt((v1['x'] - v2['x']) ** 2 + (v1['y'] - v2['y']) ** 2)


# Function to calculate the radius of a void based on its area
def calculate_radius(size):
    return np.sqrt(size / np.pi)

# Function to find matching voids within a given distance threshold
def find_matches(auto_voids, manual_voids):
    matches = []
    for m_void in manual_voids:
        for a_void in auto_voids:
            m_radius = calculate_radius(m_void['size'])
            a_radius = calculate_radius(a_void['size'])
            if distance(m_void, a_void) < min(m_radius, a_radius):
                matches.append((m_void, a_void))
                break
    return matches


# Function to calculate true positives, false positives, and false negatives
def calculate_tp_fp_fn(auto_voids, manual_voids, matches):
    tp = len(matches)
    fp = len(auto_voids) - tp
    fn = len(manual_voids) - tp
    return tp, fp, fn


# Set the base path
base_path = r"E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_3_8\bilateral"

# Path to the manually marked voids CSV
manual_file = os.path.join(base_path, 'manual_voids_info.csv')

# Load the manually marked voids
manual_voids = load_void_info(manual_file)
total_manual_voids = 5  # Number of manually labeled voids

# List of threshold folders to process
threshold_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
threshold_folders.sort(key=float)

# Initialize lists for storing results
threshold_values = []
tps, fps, fns = [], [], []


for threshold_folder in threshold_folders:
    print(f'Processing threshold folder: {threshold_folder}')
    threshold_value = float(threshold_folder)
    threshold_values.append(threshold_value)

    # Find the CSV file in the current threshold folder
    folder_path = os.path.join(base_path, threshold_folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV file found in {threshold_folder}. Skipping this folder.")
        continue

    auto_voids_file = os.path.join(folder_path, csv_files[0])
    auto_voids = load_void_info(auto_voids_file)

    # Find matches between automated and manual voids
    matches = find_matches(auto_voids, manual_voids)

    # Calculate TP, FP, FN
    tp, fp, fn = calculate_tp_fp_fn(auto_voids, manual_voids, matches)
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)



    # Print TP, FP, FN values for the current threshold
    print(f'Threshold: {threshold_value} - TP: {tp}, FP: {fp}, FN: {fn}')

# Plotting TP, FP, FN vs Threshold Values
plt.figure(figsize=(12, 8))
plt.plot(threshold_values, tps, label='True Positives (TP)', color='g')
plt.plot(threshold_values, fps, label='False Positives (FP)', color='r')
plt.plot(threshold_values, fns, label='False Negatives (FN)', color='b')



# Adding a horizontal line for the number of manually labeled voids
plt.axhline(y=total_manual_voids, color='k', linestyle='-', label=f'Manual Voids (Total: {total_manual_voids})')

plt.xlabel('Threshold Value')
plt.ylabel('Count')
plt.title('Comparison of TP, FP, FN vs Threshold Value for Void Detection')
plt.legend()
plt.grid(True)
plt.savefig('TP_FP_FN_vs_Threshold_with_Relative_and_Manual_Line.png')
plt.show()
plt.close()

print('Threshold analysis and comparison completed. Results saved.')
