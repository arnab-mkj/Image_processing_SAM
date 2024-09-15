import os
import numpy as np
from PIL import Image

# Define the base directory (current directory) containing the TIFF file
base_path = r"E:\WHK\V10082\split_images_10082_1\layer_6"  # Update the path accordingly

# Define the size of the smaller parts
tile_size = (200, 200)  # Size of each smaller part (e.g., 200x200 pixels)

# Path to the specific TIFF file to process
tiff_file = 'tile_2_7.tiff'
img_path = os.path.join(base_path, tiff_file)

# Check if the TIFF file exists
if os.path.exists(img_path):
    print(f'Processing {tiff_file} in the current directory...')

    try:
        # Load the image
        img = Image.open(img_path)
        img_array = np.array(img)

        # Create an output directory for the split tiles
        output_dir = os.path.join(base_path, 'layercheck_split_tile_2_7')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Calculate the number of tiles in each dimension
        num_tiles_x = img_array.shape[1] // tile_size[0]
        num_tiles_y = img_array.shape[0] // tile_size[1]

        # Ensure the image is large enough to be split
        if num_tiles_x == 0 or num_tiles_y == 0:
            print(f"Image size is too small for the tile size.")
        else:
            # Split the image into smaller parts
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    # Define the box for cropping the image
                    box = (j * tile_size[0], i * tile_size[1], (j + 1) * tile_size[0], (i + 1) * tile_size[1])
                    tile = img.crop(box)

                    # Save the smaller part
                    tile.save(os.path.join(output_dir, f'tile_{i}_{j}.tiff'))

            print(f'Successfully split {tiff_file} into {num_tiles_x * num_tiles_y} parts.')

    except Exception as e:
        print(f"Failed to process {tiff_file}: {e}")

else:
    print(f"File {tiff_file} does not exist in the current directory.")
