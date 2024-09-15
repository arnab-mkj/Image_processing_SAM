import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import csv

# Set the matplotlib backend to TkAgg for interactive mode
plt.switch_backend('TkAgg')

# Path and filename
path = "E:\WHK\V10082\split_images"  # Adjusted for the provided file path
file = 'tile_0_1.tiff'
os.chdir(path)

# Get image info
image_data = io.imread(file)

# Check if the image is 2D or 3D
if image_data.ndim == 2:
    num_images = 1
    image_data = np.expand_dims(image_data, axis=0)
elif image_data.ndim == 3:
    if image_data.shape[-1] == 3 or image_data.shape[-1] == 4:  # Color image
        num_images = 1
        image_data = np.expand_dims(image_data, axis=0)
    else:
        num_images = image_data.shape[0]
else:
    raise ValueError("Unsupported image dimension: {}".format(image_data.ndim))

def manual_void_marking(image_slice, marker_size=2):
    fig, ax = plt.subplots()
    ax.imshow(image_slice)  # Display original image
    manual_voids = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            manual_voids.append((x, y))
            ax.plot(x, y, 'ro', markersize=marker_size)
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Disconnect the event handler after marking
    fig.canvas.mpl_disconnect(cid)
    return manual_voids

# Prepare for storing manual void information
manual_void_info = []

# Manually mark voids in all image slices
for n in range(num_images):
    image_slice = image_data[n]
    print(f"Manually mark voids for image slice {n + 1}")
    manual_voids = manual_void_marking(image_slice, marker_size=2)
    for x, y in manual_voids:
        manual_void_info.append({
            'image_index': n + 1,
            'x': x,
            'y': y,
            'size': None  # Placeholder for size if needed
        })

    # Plotting and saving manually marked image
    plt.figure(figsize=(10, 8))
    plt.imshow(image_slice)  # Display original image
    for x, y in manual_voids:
        plt.plot(x, y, 'ro', markersize=2)
    plt.savefig(f'Manual_C_Scan_Labeled_{n + 1}.png')
    plt.close()

# Save manual void information to CSV
csv_filename = 'manual_voids_info.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['image_index', 'x', 'y', 'size'])
    writer.writeheader()
    for info in manual_void_info:
        writer.writerow(info)

print(f'Manual void detection completed. Results saved in {csv_filename}')
