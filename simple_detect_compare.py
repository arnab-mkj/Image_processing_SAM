import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
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


# Set the matplotlib backend to Agg
plt.switch_backend('Agg')

# Path and filename
path = r"E:\WHK\V10082\split_images_10082_1\layer_6\split_images_tile_1_6\testtile_0_7\threshold"
file = 'tile_0_0.tiff'
manual_file = 'manual_voids_info.csv'  # Path to manually marked voids CSV
os.chdir(path)

# Load the manually marked voids
manual_voids = load_void_info(manual_file)

# Get image info
image_data = io.imread(file)

# Check if the image is 2D or 3D
if image_data.ndim == 2:
    num_images = 1
    image_data = np.expand_dims(image_data, axis=0)
elif image_data.ndim == 3:
    num_images = image_data.shape[0]
else:
    raise ValueError("Unsupported image dimension: {}".format(image_data.ndim))

# Initialize lists for storing results
threshold_values = np.linspace(0.01, 1, 20)  #Threshold values from 0.1 to 1.0
tps, fps, fns = [], [], []

for threshold_value in threshold_values:
    print(f'Processing threshold: {threshold_value}')

    for n in range(num_images):
        # Convert color values
        image_data[n] = 2 * np.abs(image_data[n].astype(np.float64) - 128)

        # Apply Sobel filter to detect edges
        edges = filters.sobel(image_data[n])

        # Plot edges detected image
        plt.figure(figsize=(10, 8))
        plt.imshow(edges, cmap='gray')
        plt.title(f'Edges Detected (Sobel) Image (Threshold: {threshold_value})')
        plt.savefig(f'Edges_Detected_Image_{n + 1}_Threshold_{threshold_value}.png')
        plt.close()

        # Manual Thresholding without Otsu
        bin_im = edges > threshold_value  # Direct thresholdin

        # Plot binary thresholded image
        plt.figure(figsize=(10, 8))
        plt.imshow(bin_im, cmap='gray')
        plt.title(f'Binary Thresholded Image (Threshold: {threshold_value})')
        #plt.savefig(f'Binary_Thresholded_Image_{n + 1}_Threshold_{threshold_value}.png')
        plt.close()

        # Morphological operations to clean up the image
        bin_im = morphology.remove_small_objects(bin_im, 15)
        bin_im = morphology.remove_small_holes(bin_im, 100)

        # Clustering (Label connected components)
        BW = label(bin_im)

        # Find and store voids
        auto_voids = []
        for region in measure.regionprops(BW):
            if 15 <= region.area < 60:
                auto_voids.append({
                    'x': region.centroid[1],
                    'y': region.centroid[0],
                    'size': region.area
                })

        # Find matches between automated and manual voids
        matches = find_matches(auto_voids, manual_voids)

        # Calculate TP, FP, FN
        tp, fp, fn = calculate_tp_fp_fn(auto_voids, manual_voids, matches)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

        # Plot final voids detection image without filtering
        plt.figure(figsize=(10, 8))

        # Plot original grayscale image
        plt.subplot(3, 1, 1)
        plt.imshow(image_data[n], cmap='gray')
        plt.title(f'Original Grayscale Image (Threshold: {threshold_value})')

        # Plot binary image after thresholding
        plt.subplot(3, 1, 2)
        plt.imshow(bin_im, cmap='gray')
        plt.title('Binary Image After Thresholding')

        # Plot detected voids on original image
        plt.subplot(3, 1, 3)
        plt.imshow(image_data[n], cmap='gray')
        plt.title('Detected Voids on Original Image')

        for void in auto_voids:
            plt.plot(void['x'], void['y'], 'r.', markersize=10)

        #plt.savefig(f'Final_Voids_Detection_{n + 1}_Threshold_{threshold_value}.png')
        plt.close()

# Plotting TP, FP, FN vs Threshold Values
plt.figure(figsize=(10, 6))
plt.plot(threshold_values, tps, label='True Positives (TP)')
plt.plot(threshold_values, fps, label='False Positives (FP)')
plt.plot(threshold_values, fns, label='False Negatives (FN)')
plt.xlabel('Threshold Value')
plt.ylabel('Count')
plt.title('TP, FP, FN vs Threshold Value')
plt.legend()
plt.savefig('TP_FP_FN_vs_Threshold.png')
plt.close()

print('Threshold analysis completed. Results saved.')
