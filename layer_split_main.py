import os
import numpy as np
from PIL import Image

# Path and filename
input_path = r"E:\WHK\V10082"
input_file = '10082_1.tiff'
os.chdir(input_path)

# Load the multi-layer TIFF image
img = Image.open(input_file)

# Create the output directory if it doesn't exist
output_dir = f'output_{input_file}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each layer (frame) in the TIFF file
layer_index = 0
while layer_index <6 :
    try:
        # Load the current layer
        img.seek(layer_index)
        img_array = np.array(img)

        # Save the entire layer as a separate TIFF file without splitting
        layer_filename = os.path.join(output_dir, f'layer_{layer_index + 1}.tiff')
        img.save(layer_filename)

        print(f'Successfully saved layer {layer_index + 1} as {layer_filename}.')

        # Move to the next layer
        layer_index += 1

    except EOFError:
        # End of the image layers
        break

print(f'Successfully split all layers and saved them as separate TIFF files.')
