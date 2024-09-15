import numpy as np
from PIL import Image
import cv2

# Function to create a synthetic TIFF file with a speckled pattern
def create_speckled_tiff(filename, num_images=5, image_size=(1024, 1024), num_speckles=1000):
    images = []
    for _ in range(num_images):
        # Create a black image
        image = np.zeros(image_size, dtype=np.uint8)

        # Add random speckles
        for _ in range(num_speckles):
            x, y = np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1])
            image[x, y] = 255  # White speckles on a black background

        # Optionally, apply Gaussian blur to soften the speckles
        image = cv2.GaussianBlur(image, (3, 3), 0)

        images.append(Image.fromarray(image))

    # Save images as multi-page TIFF
    images[0].save(filename, save_all=True, append_images=images[1:], compression="tiff_deflate")


# Create speckled TIFF file
create_speckled_tiff('speckled_voids.tiff')
