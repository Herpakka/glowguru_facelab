import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load the image
img = cv2.imread("train/dry/1.jpg", cv2.IMREAD_GRAYSCALE)

# Perform Haar DWT
coeffs = pywt.dwt2(img, 'haar')

# Display the original and decomposed images
titles = ['Original Image', 'Approximation', 'Horizontal detail', 'Vertical detail']

fig = plt.figure(figsize=(12, 3))
for i, coef in enumerate(coeffs):
    ax = fig.add_subplot(1, 4, i + 1)
    
    # For the approximation (cA), display the entire array
    if i == 0:
        ax.imshow(coef, interpolation="nearest", cmap=plt.cm.gray)
    else:
        # For other components (cH, cV), display the absolute value
        ax.imshow(np.abs(coef), interpolation="nearest", cmap=plt.cm.gray)
    
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
