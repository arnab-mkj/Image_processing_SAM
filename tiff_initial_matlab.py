import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
import csv
from skimage.measure import label, regionprops
from skimage.restoration import denoise_bilateral

# Set the matplotlib backend to Agg
plt.switch_backend('Agg')

# Path and filename
path = r"E:\WHK\V10082\split_images_10082_1\layer_6\layercheck_split_tile_2_7\bilateral"
file = 'tile_4_0.tiff'
os.chdir(path)

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

# Prepare for storing void information
void_info = []

for n in range(num_images):
    # Convert color values
    image_data[n] = 2 * np.abs(image_data[n].astype(np.float64) - 128)

    # Plot original adjusted image
    plt.figure(figsize=(10, 8))
    plt.imshow(image_data[n], cmap='gray')
    plt.title('Original Adjusted Image')
    #plt.savefig(f'Original_Adjusted_Image_{n + 1}.png')
    plt.close()

    # Apply bilateral filter
    bilateral_filtered_image = denoise_bilateral(image_data[n], sigma_color=0.042, sigma_spatial=14.736)

    # Plot bilateral filtered image
    plt.figure(figsize=(10, 8))
    plt.imshow(bilateral_filtered_image, cmap='gray')
    plt.title('Bilateral Filtered Image')
    plt.savefig(f'Bilateral_Filtered_Image_{n + 1}.png')
    plt.close()

    # Apply Sobel filter to detect edges
    edges = filters.sobel(bilateral_filtered_image)

    # Plot edges detected image
    plt.figure(figsize=(10, 8))
    plt.imshow(edges, cmap='gray')
    plt.title('Edges Detected (Sobel) Image')
    plt.savefig(f'Edges_Detected_Image_{n + 1}.png')
    plt.close()

    # Thresholding
    threshold_value = 0.1
    bin_im = edges > threshold_value  # Direct thresholding
    # thresh = filters.threshold_otsu(edges)
    # Print the threshold value
    #print(f"Otsu's threshold value: {thresh}")
    #bin_im = edges > thresh

    # Plot binary thresholded image
    plt.figure(figsize=(10, 8))
    plt.imshow(bin_im, cmap='gray')
    plt.title('Binary Thresholded Image')
    plt.savefig(f'Binary_Thresholded_Image_{n + 1}.png')
    plt.close()

    # Morphological operations to clean up the image
    bin_im = morphology.remove_small_objects(bin_im, 5)
    bin_im = morphology.remove_small_holes(bin_im, 100)

    # Plot binary thresholded image
    plt.figure(figsize=(10, 8))
    plt.imshow(bin_im, cmap='gray')
    plt.title('Binary Thresholded Image after holes')
    plt.savefig(f'Binary_Thresholded_Image_holes_{n + 1}.png')
    plt.close()

    # Clustering
    BW = label(bin_im)

    # Find and store voids
    voids = []
    for region in measure.regionprops(BW):
        if 15 <= region.area < 100:
            voids.append({
                'region': region,
                'size': region.area
            })

    # Save final voids on binary image
    plt.figure(figsize=(10, 8))
    plt.imshow(bin_im, cmap='gray')
    plt.title('Final Voids Detection on Binary Image')
    for void in voids:
        region = void['region']
        plt.plot(region.coords[:, 1], region.coords[:, 0], 'r.', markersize=6)
    plt.savefig(f'Final_Voids_Binary_Image_{n + 1}.png')
    plt.close()

    # Save final voids on original grayscale image
    plt.figure(figsize=(10, 8))
    plt.imshow(image_data[n], cmap='gray')
    plt.title('Final Voids Detection on Original Image')
    for void in voids:
        region = void['region']
        plt.plot(region.coords[:, 1], region.coords[:, 0], 'r.', markersize=6)
    plt.savefig(f'Final_Voids_Original_Image_{n + 1}.png')
    plt.close()

    # Save void information for the voids
    for void in voids:
        region = void['region']
        void_info.append({
            'image_index': n + 1,
            'x': region.centroid[1],
            'y': region.centroid[0],
            'size': region.area
        })

    # Final comparison plot of original and voids
    plt.figure(figsize=(12, 10))

    # Original grayscale image
    plt.subplot(2, 1, 1)
    plt.imshow(image_data[n], cmap='gray')
    plt.title('Original Grayscale Image')

    # Final voids detection on original grayscale image
    plt.subplot(2, 1, 2)
    plt.imshow(image_data[n], cmap='gray')
    plt.title('Final Voids Detection on Original Image')
    for void in voids:
        region = void['region']
        plt.plot(region.coords[:, 1], region.coords[:, 0], 'r.', markersize=3)

    plt.savefig(f'Final_Comparison_Original_Voids_{n + 1}.png')
    plt.close()

# Save void information to CSV
csv_filename = f'bilateral_voids_{threshold_value}.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['image_index', 'x', 'y', 'size'])
    writer.writeheader()
    for info in void_info:
        writer.writerow(info)

print(f'Voids detection completed. Results saved in {csv_filename}')
