import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
import csv
from skimage.restoration import denoise_bilateral
from skimage.measure import label
from skimage.io import imsave

# Function to detect voids using optimized bilateral filter parameters
def detect_voids_3d(image_3d, sigma_color, sigma_spatial, threshold_value=0.1):
    voids_3d = []
    for n in range(image_3d.shape[0]):
        # Apply bilateral filter
        bilateral_filtered_image = denoise_bilateral(image_3d[n], sigma_color=sigma_color, sigma_spatial=sigma_spatial)

        # Apply Sobel filter to detect edges
        edges = filters.sobel(bilateral_filtered_image)

        # Thresholding
        bin_im = edges > threshold_value

        # Morphological operations to clean up the image
        bin_im = morphology.remove_small_objects(bin_im, 5)
        bin_im = morphology.remove_small_holes(bin_im, 100)

        # Clustering (Label connected components)
        BW = label(bin_im)

        # Find and store voids
        voids = []
        for region in measure.regionprops(BW):
            if 30 <= region.area < 100:
                voids.append({
                    'slice_index': n,
                    'x': region.centroid[1],
                    'y': region.centroid[0],
                    'size': region.area
                })
        voids_3d.append(voids)
    return voids_3d

# Function to save a 3D image as a TIFF file
def save_3d_tiff(image_3d, output_path):
    imsave(output_path, image_3d.astype(np.uint8))

# Function to analyze voids detected across layers
def analyze_voids_across_layers(voids_3d):
    layer_counts = []
    for voids in voids_3d:
        layer_counts.append(len(voids))
    return layer_counts

# Set the base path
base_path = r"E:\WHK\V10082\split_images_10082_1"
layers = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Sort the layers to ensure they are in the correct order
layers.sort(key=lambda x: int(x.split('_')[-1]))  # Extract the number from the directory name and sort

# Collect the sub-tiles from each layer into a 3D array
sub_tile_name = 'tile_0_0.tiff'
image_3d = []
for layer in layers:
    layer_path = os.path.join(base_path, layer, 'layercheck_split_tile_1_6')
    sub_tile_path = os.path.join(layer_path, sub_tile_name)
    if os.path.exists(sub_tile_path):
        image_2d = io.imread(sub_tile_path)
        image_3d.append(image_2d)
    else:
        print(f"{sub_tile_name} not found in {layer_path}")

# Convert list to 3D numpy array
image_3d = np.array(image_3d)

# Apply color value conversion
for n in range(image_3d.shape[0]):
    image_3d[n] = 2 * np.abs(image_3d[n].astype(np.float64) - 128)

# Save the 3D image as a TIFF file
output_3d_tiff = os.path.join(base_path, 'combined_3d_image.tiff')
save_3d_tiff(image_3d, output_3d_tiff)
print(f'Saved 3D image to {output_3d_tiff}')

# Detect voids in the 3D image
sigma_color = 0.042105263157894736
sigma_spatial = 14.736842105263158
voids_3d = detect_voids_3d(image_3d, sigma_color, sigma_spatial, threshold_value=0.1)

# Analyze how many layers contain voids
layer_counts = analyze_voids_across_layers(voids_3d)

# Output the results
for i, count in enumerate(layer_counts):
    print(f'Layer {i+1} has {count} voids.')

# Save the void information to a CSV file
csv_filename = os.path.join(base_path, 'voids_across_layers.csv')
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['slice_index', 'x', 'y', 'size'])
    writer.writeheader()
    for voids in voids_3d:
        for void in voids:
            writer.writerow(void)
print(f'Voids analysis across layers completed. Results saved in {csv_filename}')
