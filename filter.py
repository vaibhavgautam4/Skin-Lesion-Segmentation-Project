# import cv2
# import numpy as np
# from main import create_dir

# def apply_gabor_filter(image, kernel_size, theta, sigma, frequency):
#     """
#     Apply a Gabor filter to an image.

#     Parameters:
#     - image: Input image (numpy array).
#     - kernel_size: Size of the Gabor kernel.
#     - theta: Orientation of the Gabor filter (in radians).
#     - sigma: Standard deviation of the Gaussian component of the kernel.
#     - frequency: Frequency of the sinusoidal component of the kernel.

#     Returns:
#     - Filtered image.
#     """
#     # Create a Gabor kernel
#     kernel = cv2.getGaborKernel(
#         (kernel_size, kernel_size), sigma, theta, frequency, 1.0, 0, ktype=cv2.CV_32F
#     )

#     # Apply the Gabor filter to the image
#     filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)

#     return filtered_image

# if __name__ == "__main__":
#     # Load an example image
#     results = create_dir("results")
#     image = cv2.imread("D:/segmentation/segment/usic/Skin cancer ISIC The International Skin Imaging Collaboration/Train/actinic keratosis/ISIC_0025780.jpg", cv2.IMREAD_COLOR)

# # Parameters for the Gabor filter
# kernel_size = 21  # Adjust as needed
# theta = np.pi / 4  # Orientation of the filter (45 degrees)
# sigma = 5  # Adjust as needed
# frequency = 0.5  # Adjust as needed

# # Apply the Gabor filter
# filtered_image = apply_gabor_filter(image, kernel_size, theta, sigma, frequency)
# print(image.shape)
# print(filtered_image.shape)
# # Display the original and filtered images
# cv2.imshow("Original Image", image)
# cv2.imshow("Filtered Image", filtered_image)
# # line = np.ones((450, 10, 3)) * 255
# # cat_images = np.concatenate([image, line, filtered_image ], axis = 1)
# # save_image_path = f"results/filter/filterimg.jpg"
# # cv2.imwrite(save_image_path, cat_images )
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the skin image
image = cv2.imread('D:/segmentation/segment/usic/Skin cancer ISIC The International Skin Imaging Collaboration/Train/actinic keratosis/ISIC_0025780.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

# Perform edge detection
edges = cv2.Canny(blurred_image, 30, 150)

# Find contours in the edges image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for the detected contours (hair)
hair_mask = np.zeros_like(gray_image)

# Draw the detected contours (hair) on the mask
cv2.drawContours(hair_mask, contours, -1, 255, thickness=cv2.FILLED)

# Invert the hair mask
inverted_hair_mask = cv2.bitwise_not(hair_mask)

# Use the inverted hair mask to remove hair from the original image
hair_removed_image = cv2.bitwise_and(image, image, mask=inverted_hair_mask)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(132)
plt.imshow(inverted_hair_mask, cmap='gray')
plt.title('Hair Mask')

plt.subplot(133)
plt.imshow(cv2.cvtColor(hair_removed_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Hair Removed')

plt.show()

