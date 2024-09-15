import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import csv

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
    return np.sqrt((v1['x'] - v2['x'])**2 + (v1['y'] - v2['y'])**2)

# Function to calculate the radius of a void based on its area
def calculate_radius(size):
    return np.sqrt(size / np.pi)

# Function to find matching voids using the radius method
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

# Function to compare the sizes of matched voids
def compare_sizes(matches):
    size_differences = []
    for m_void, a_void in matches:
        size_diff = abs(m_void['size'] - a_void['size'])
        size_differences.append(size_diff)
    return size_differences

# Set the matplotlib backend to Agg
plt.switch_backend('Agg')

# Path and filename
path = r"E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\bilateral"
auto_file = 'Final_Voids_Original_Image_1.tiff'
manual_file = 'Manual_marked.tiff'
os.chdir(path)

# Load automated void detection results
auto_voids = load_void_info('bilateral_voids_0.1.csv')

# Load manual void detection results
manual_voids = load_void_info('manual_voids_info.csv')

# Find matches between automated and manual voids using the radius method
matches = find_matches(auto_voids, manual_voids)

# Calculate TP, FP, FN
tp, fp, fn = calculate_tp_fp_fn(auto_voids, manual_voids, matches)
print(f'True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}')

# Compare sizes of matched voids
size_differences = compare_sizes(matches)
avg_size_difference = np.mean(size_differences)
print(f'Average Size Difference: {avg_size_difference}')

# Plot the results
plt.figure(figsize=(10, 8))

# Load the manually marked image for background
manual_image_data = io.imread(manual_file)
plt.imshow(manual_image_data)

# Plot manual voids (blue)
for info in manual_voids:
    plt.plot(info['x'], info['y'], 'b.', markersize=8)

# Plot automated voids (green)
for info in auto_voids:
    plt.plot(info['x'], info['y'], 'g.', markersize=8)

# Plot matched voids (red)
for m_void, a_void in matches:
    plt.plot(m_void['x'], m_void['y'], 'r.', markersize=12)

plt.savefig('comparison_Voids_Detected.png')
plt.close()

print('Comparison completed and results saved.')
