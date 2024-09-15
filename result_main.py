import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
import csv
from skimage.restoration import denoise_bilateral
from skimage.measure import label
import sys

# Set the matplotlib backend to Agg
plt.switch_backend('Agg')

# Get file name from command-line arguments
file = sys.argv[1]

# Path and filename
path = r"E:\WHK\V10082\output_10082_1"
os.chdir(path)

# Get image info
image_data = io.imread(file)
print("Image loaded successfully")

# Check if the image is 2D or 3D
if image_data.ndim == 2:
    num_images = 1
    image_data = np.expand_dims(image_data, axis=0)
elif image_data.ndim == 3:
    num_images = image_data.shape[0]
else:
    raise ValueError(f"Unsupported image dimension: {image_data.ndim}")

print(f"Number of layers in the image: {num_images}")

# Prepare for storing void information
void_info = []

# Define the size of the smaller parts
tile_size = (1024, 1024)  # Adjust this as needed

for n in range(num_images):
    print(f"Processing layer {n + 1}/{num_images}")

    # Convert color values
    image_data[n] = 2 * np.abs(image_data[n].astype(np.float64) - 128)
    print(f"Layer {n + 1} color values converted")

    # Get image dimensions
    image_height, image_width = image_data[n].shape
    print(f"Image dimensions (HxW): {image_height}x{image_width}")

    # Calculate the number of tiles in each dimension
    num_tiles_x = (image_width + tile_size[0] - 1) // tile_size[0]
    num_tiles_y = (image_height + tile_size[1] - 1) // tile_size[1]
    print(f"Number of tiles (XxY): {num_tiles_x}x{num_tiles_y}")

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            print(f"Processing tile ({i + 1}, {j + 1})")

            # Define the boundaries of the current tile
            start_x = j * tile_size[0]
            start_y = i * tile_size[1]
            end_x = min(start_x + tile_size[0], image_width)
            end_y = min(start_y + tile_size[1], image_height)

            # Extract the current tile
            tile = image_data[n][start_y:end_y, start_x:end_x]
            print(f"Tile ({i + 1}, {j + 1}) extracted with size: {tile.shape}")

            # Apply bilateral filter
            bilateral_filtered_tile = denoise_bilateral(tile, sigma_color=0.042, sigma_spatial=14.736)
            print(f"Bilateral filter applied to tile ({i + 1}, {j + 1})")

            # Apply Sobel filter to detect edges
            edges = filters.sobel(bilateral_filtered_tile)
            print(f"Sobel filter applied to tile ({i + 1}, {j + 1})")

            # Thresholding
            threshold_value = 0.1
            bin_im = edges > threshold_value  # Direct thresholding
            print(f"Thresholding applied to tile ({i + 1}, {j + 1})")

            # Morphological operations to clean up the image
            bin_im = morphology.remove_small_objects(bin_im, 15)
            bin_im = morphology.remove_small_holes(bin_im, 100)
            print(f"Morphological operations applied to tile ({i + 1}, {j + 1})")

            # Clustering
            BW = label(bin_im)
            print(f"Labeling completed for tile ({i + 1}, {j + 1})")

            # Find and store voids in the current tile
            tile_voids_count = 0
            for region in measure.regionprops(BW):
                if 15 <= region.area < 100:
                    void_info.append({
                        'image_index': n + 1,
                        'x': region.centroid[1] + start_x,
                        'y': region.centroid[0] + start_y,
                        'size': region.area
                    })
                    tile_voids_count += 1
            print(f"Found {tile_voids_count} voids in tile ({i + 1}, {j + 1})")

print(f"Total voids detected: {len(void_info)}")

# Save void information to CSV
csv_filename = f'{file}.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['image_index', 'x', 'y', 'size'])
    writer.writeheader()
    for info in void_info:
        writer.writerow(info)

print(f'Voids detection completed. Results saved in {csv_filename}')
