import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, color
from skimage.filters import threshold_otsu
import csv
import os

# Path and filename
path = r"E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\bilateral"  # Adjusted for the provided file path
file = 'Manual_marked.tiff'
os.chdir(path)

# Load image
image_data = io.imread(file)

# Check if the image is RGB
#if image_data.ndim != 3 or image_data.shape[-1] != 3:
 #   raise ValueError("The provided image is not an RGB image.")
# Check if the image has an alpha channel (i.e., 4 channels)
if image_data.shape[-1] == 4:
    # Drop the alpha channel
    image_data = image_data[..., :3]
# Convert the image to the HSV color space
hsv_image = color.rgb2hsv(image_data)

# Isolate the blue color
# Define thresholds for the blue hue
hue_min, hue_max = 0.4, 0.8  # Adjust these values if necessary

# Create a binary mask for the blue color
blue_mask = (hsv_image[:, :, 0] >= hue_min) & (hsv_image[:, :, 0] <= hue_max)

# Apply morphological operations to clean up the mask
blue_mask = morphology.remove_small_objects(blue_mask, min_size=5)
#blue_mask = morphology.remove_small_holes(blue_mask, area_threshold=5)

# Label the connected components in the mask
labeled_mask = measure.label(blue_mask)

# Prepare for storing void information
manual_void_info = []

# Measure properties of labeled regions
regions = measure.regionprops(labeled_mask)
for region in regions:
    manual_void_info.append({
        'x': region.centroid[1],
        'y': region.centroid[0],
        'size': region.area
    })

# Save manual void information to CSV
csv_filename = 'manual_voids_info.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['x', 'y', 'size'])
    writer.writeheader()
    for info in manual_void_info:
        writer.writerow(info)

print(f'Manually marked voids detection completed. Results saved in {csv_filename}')

# Plotting and saving the labeled image
plt.figure(figsize=(10, 8))
plt.imshow(image_data)

# Overlay the voids on the original image
for info in manual_void_info:
    plt.plot(info['x'], info['y'], 'r.', markersize=12)

plt.savefig('manual_Voids_Detected.png')
plt.close()
