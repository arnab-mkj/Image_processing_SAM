import numpy as np
from skimage import filters, morphology, measure
from skimage.restoration import denoise_bilateral
from skimage.measure import label

def apply_bilateral_and_detect_voids(image_data, sigma_color, sigma_spatial, threshold_value):
    # Apply Bilateral filter
    filtered_image = denoise_bilateral(image_data, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

    # Apply Sobel filter to detect edges
    edges = filters.sobel(filtered_image)

    # Apply the fixed threshold
    bin_im = edges > threshold_value

    # Morphological operations to clean up the image
    bin_im = morphology.remove_small_objects(bin_im, 15)
    bin_im = morphology.remove_small_holes(bin_im, 100)

    # Clustering (Label connected components)
    BW = label(bin_im)

    # Find and store voids
    auto_voids = []
    for region in measure.regionprops(BW):
        if 15 <= region.area < 100:
            auto_voids.append({
                'x': region.centroid[1],
                'y': region.centroid[0],
                'size': region.area
            })

    return auto_voids
