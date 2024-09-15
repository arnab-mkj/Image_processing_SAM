import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import csv
from bilateral import apply_bilateral_and_detect_voids
import matplotlib

matplotlib.use('TkAgg')  # TkAgg is interactive

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

# Set the matplotlib backend to Agg
plt.switch_backend('Agg')

# Set the base path
base_path = r"E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\heatmap"
manual_file = os.path.join(base_path, 'manual_voids_info.csv')

# Load the manually marked voids
manual_voids = load_void_info(manual_file)

# Load the image data
image_file = os.path.join(base_path, 'tile_4_0.tiff')  # Adjust the filename accordingly
image_data = io.imread(image_file)

# Check if the image is 2D or 3D
if image_data.ndim == 2:
    num_images = 1
    image_data = np.expand_dims(image_data, axis=0)
elif image_data.ndim == 3:
    num_images = image_data.shape[0]
else:
    raise ValueError("Unsupported image dimension: {}".format(image_data.ndim))

# Prepare for storing void information
void_info = []

# Preprocess each layer of the image
for n in range(num_images):
    # Convert color values
    image_data[n] = 2 * np.abs(image_data[n].astype(np.float64) - 128)

# Bilateral filter parameter ranges
sigma_color_values = np.linspace(0.00, 0.20, 20)
sigma_spatial_values = np.linspace(0, 20, 20)

# Initialize arrays to store TP, FP, and FN results
tp_results = np.zeros((len(sigma_color_values), len(sigma_spatial_values)))
fp_results = np.zeros((len(sigma_color_values), len(sigma_spatial_values)))
fn_results = np.zeros((len(sigma_color_values), len(sigma_spatial_values)))

# Fixed threshold
threshold_value = 0.1

# Loop over all combinations of sigma_color and sigma_spatial
for i, sigma_color in enumerate(sigma_color_values):
    for j, sigma_spatial in enumerate(sigma_spatial_values):
        print(f'Processing: sigma_color={sigma_color}, sigma_spatial={sigma_spatial}')

        # Call the function from bilateral_detection.py
        auto_voids = apply_bilateral_and_detect_voids(image_data[0], sigma_color, sigma_spatial, threshold_value)

        # Find matches between automated and manual voids
        matches = find_matches(auto_voids, manual_voids)

        # Calculate TP, FP, FN and store them
        tp, fp, fn = calculate_tp_fp_fn(auto_voids, manual_voids, matches)
        tp_results[i, j] = tp
        fp_results[i, j] = fp
        fn_results[i, j] = fn

        # Print TP, FP, FN values for the current parameters
        print(f'TP: {tp}, FP: {fp}, FN: {fn}')

# Plotting TP results as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(tp_results, cmap='hot', interpolation='nearest')
plt.colorbar(label='True Positives (TP)')
plt.xlabel('Sigma Spatial')
plt.ylabel('Sigma Color')
plt.title('True Positives (TP) Heatmap - Varying Bilateral Filter Parameters')
plt.xticks(np.arange(len(sigma_spatial_values)), labels=np.round(sigma_spatial_values, 2), rotation=90)
plt.yticks(np.arange(len(sigma_color_values)), labels=np.round(sigma_color_values, 2), rotation=0)
plt.savefig(r'E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\heatmap\TP_Heatmap_Bilateral_Filter.png')
plt.show()

# Plotting FP results as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(fp_results, cmap='hot', interpolation='nearest')
plt.colorbar(label='False Positives (FP)')
plt.xlabel('Sigma Spatial')
plt.ylabel('Sigma Color')
plt.title('False Positives (FP) Heatmap - Varying Bilateral Filter Parameters')
plt.xticks(np.arange(len(sigma_spatial_values)), labels=np.round(sigma_spatial_values, 2), rotation=90)
plt.yticks(np.arange(len(sigma_color_values)), labels=np.round(sigma_color_values, 2), rotation=0)
plt.savefig(r'E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\heatmap\FP_Heatmap_Bilateral_Filter.png')
plt.show()

# Optionally, you can also plot FN (False Negatives) as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(fn_results, cmap='hot', interpolation='nearest')
plt.colorbar(label='False Negatives (FN)')
plt.xlabel('Sigma Spatial')
plt.ylabel('Sigma Color')
plt.title('False Negatives (FN) Heatmap - Varying Bilateral Filter Parameters')
plt.xticks(np.arange(len(sigma_spatial_values)), labels=np.round(sigma_spatial_values, 2), rotation=90)
plt.yticks(np.arange(len(sigma_color_values)), labels=np.round(sigma_color_values, 2), rotation=0)
plt.savefig(r'E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\heatmap\FN_Heatmap_Bilateral_Filter.png')
plt.show()
