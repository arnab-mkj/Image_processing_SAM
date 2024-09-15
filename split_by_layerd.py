import os
import numpy as np
from PIL import Image

# Path and filename
input_path = r"E:\WHK\V10082"
input_file = '10082_1.tiff'
os.chdir(input_path)

# Load the multi-layer TIFF image
img = Image.open(input_file)

# Define the size of the smaller parts
tile_size = (1024, 1024)  # Size of each smaller part (width, height)

# Create the output directory if it doesn't exist
output_dir = 'split_images_10082_1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each layer (frame) in the TIFF file
layer_index = 0
while True:
    try:
        # Load the current layer
        img.seek(layer_index)
        img_array = np.array(img)

        # Create a specific folder for the current layer
        layer_folder = os.path.join(output_dir, f'layer_{layer_index + 1}')
        if not os.path.exists(layer_folder):
            os.makedirs(layer_folder)

        # Calculate the number of tiles in each dimension
        num_tiles_x = img_array.shape[1] // tile_size[0]
        num_tiles_y = img_array.shape[0] // tile_size[1]

        # Split the layer into smaller parts
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                # Define the box for cropping the image
                box = (j * tile_size[0], i * tile_size[1], (j + 1) * tile_size[0], (i + 1) * tile_size[1])
                tile = img.crop(box)

                # Save the smaller part with layer information in the filename
                tile.save(os.path.join(layer_folder, f'tile_{i}_{j}.tiff'))

        print(f'Successfully split layer {layer_index + 1} into {num_tiles_x * num_tiles_y} parts.')

        # Move to the next layer
        layer_index += 1

    except EOFError:
        # End of the image layers
        break

print(f'Successfully split all layers into smaller parts.')
