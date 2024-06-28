import cv2
import numpy as np
import matplotlib.pyplot as plt

def posterize_image_pyr_mean_shift(image_path, spatial_radius, color_radius, num_levels):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Apply pyrMeanShiftFiltering
    filtered_image = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius)

    # Quantize the colors
    quantized_image = np.floor(filtered_image / (256 / num_levels)) * (256 / num_levels)
    quantized_image = np.uint8(quantized_image)

    return quantized_image

# Parameters
image_path = 'train/oily/8.jpg'
spatial_radius = 20  # Spatial window radius
color_radius = 40    # Color window radius
num_levels = 10       # Number of levels for quantization

# Posterize the image
posterized_image = posterize_image_pyr_mean_shift(image_path, spatial_radius, color_radius, num_levels)

# Display the original and posterized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Posterized Image')
plt.imshow(posterized_image)
plt.axis('off')

plt.show()
