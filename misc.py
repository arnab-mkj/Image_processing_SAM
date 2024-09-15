# #extracting the metadata
#
# # from PIL import Image
# # from PIL.ExifTags import TAGS
# # import os
# #
# # path = r"E:\WHK\V10082"
# # file = '10082_1.tiff'
# # os.chdir(path)
# #
# # image = Image.open(file)
# # info_dict = {}
# #
# # for key, value in image.tag.items():
# #     tag_name = TAGS.get(key, key)  # Use the tag name if available, otherwise the key itself
# #     info_dict[tag_name] = value
# #
# # print(info_dict)
#
# #checking for noise
# import os
# import numpy as np
# from skimage import io
# import matplotlib.pyplot as plt
#
# # Load the TIFF image
# path = r"E:\WHK\V10082"
# file = '10082_1.tiff'
# os.chdir(path)
# image = io.imread(file)
#
# # If the TIFF file has multiple layers, select one
# if image.ndim == 3:
#     image_layer = image[6]  # Select the first layer, or choose a specific layer
# else:
#     image_layer = image
#
# # Display the image to select a ROI manually (or you can automate this step)
# plt.imshow(image_layer, cmap='gray')
# plt.title('Select a uniform region for noise estimation')
# plt.show()
#
# # For automation, you might define the ROI as follows:
# # Example: Select a region starting at (x1, y1) with width w and height h
# x1, y1 = 100, 100  # Starting coordinates
# w, h = 150, 150      # Width and height of the region
# roi = image_layer[y1:y1+h, x1:x1+w]
#
# # Calculate the noise as the standard deviation of pixel values in the ROI
# noise_estimate = np.std(roi)
# print(f"Estimated Noise Level: {noise_estimate}")
#
# # Optionally, visualize the ROI
# plt.imshow(roi, cmap='gray')
# plt.title('Selected ROI for Noise Estimation')
# plt.show()

import cupy as cp
print(cp.cuda.runtime.runtimeGetVersion())
