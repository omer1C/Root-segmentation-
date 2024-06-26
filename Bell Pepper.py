import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color2gray(color_image):
    return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

def unsharp_mask(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    sharpened = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened

def first_order_filter(sharpened_image, desired_ratio):
    # Compute the histogram of the shifted grayscale image
    histogram_1 = cv2.calcHist([sharpened_image], [0], None, [256], [0, 256])

    most_common_val = np.argsort(histogram_1.flatten())[-1]
    # greatest_magnitude = histogram_1.flatten()[most_common_val]

    #Base case - How to fing the ratio
    # cutoff_val = [val for val in range(0, 255) if val > most_common_val and histogram_1.flatten()[val] <=
    #               0.22*greatest_magnitude][0]

    cutoff_val = find_gray_value_for_desired_ratio(histogram_1, desired_ratio)
    clear_values = range(0, cutoff_val)

    mask = np.ones_like(sharpened_image) * 255
    for value in clear_values:
        mask[sharpened_image == value] = 0

    first_order_filter = cv2.bitwise_and(sharpened_image, mask)

    # SR_bicubic_p1_x2 image desired ratio
    # histogram_2 = cv2.calcHist([first_order_filter], [0], None, [256], [0, 256])
    # background = histogram_2.flatten()[0]
    # total_pixels = np.sum(histogram_2)
    # desired_ratio = background/total_pixels
    # print(f'Desired background ratio: {desired_ratio}')

    return first_order_filter

def find_gray_value_for_desired_ratio(histogram, desired_ratio):
    # Step 1: Calculate the total number of pixels
    total_pixels = np.sum(histogram)

    # Step 2: Calculate the desired number of pixels with value 0
    desired_zero_pixels = total_pixels * desired_ratio

    # Step 3: Find the gray value using the cumulative sum
    cumulative_sum = np.cumsum(histogram)
    gray_value = np.searchsorted(cumulative_sum, desired_zero_pixels)

    return gray_value

def mean_shift_filter(filtered_image):
    # Step 3: Convert the grayscale image back to a color image with 3 channels
    color_image_gray = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    shifted_image = cv2.pyrMeanShiftFiltering(color_image_gray, 10, 60)
    # Compute the connected components and labels
    gray_shifted_image = cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY)
    return gray_shifted_image

def thresholding(filtered_image):
    # Apply thresholding to segment regions
    _, thresholded_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Compute the connected components and labels on the threshold image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded_image)

    # Find the label of the largest cluster (excluding background label 0)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask for the brightest cluster
    largest_label_image = np.uint8(labels == largest_label) * 255

    return largest_label_image

def morphology_filter(filtered_image):
    # Apply Opening Filter
    kernel = np.ones((2, 2), np.uint8)
    opening_mean = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply Closing Filter (Minimal as Possible)
    closing_mean = cv2.morphologyEx(opening_mean, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing_mean

def fine_tuning(filtered_image):
    # Compute the connected components and labels on the threshold image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_image)

    # Find the label of the largest cluster (excluding background label 0)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask for the brightest cluster
    largest_label_image = np.uint8(labels == largest_label) * 255

    return largest_label_image

def root_only(segmented_image):
    # open_iterations = int(np.floor(scaling_factor + np.sqrt(scaling_factor)))
    # close_iterations = 2*int(scaling_factor)
    update = scale_factor/2 # Base case is x2
    root_only = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=int(update*2))
    root_only = cv2.morphologyEx(root_only, cv2.MORPH_CLOSE, kernel=np.ones((2, 2), np.uint8), iterations=int(update*2))
    root_only_rotated = cv2.morphologyEx(cv2.rotate(root_only, cv2.ROTATE_180), cv2.MORPH_CLOSE,
                                         kernel=np.ones((2, 2), np.uint8), iterations=int(update*2))
    root_only = cv2.bitwise_or(root_only, cv2.rotate(root_only_rotated, cv2.ROTATE_180))
    root_only = fine_tuning(root_only)

    return root_only

def histogram_plot(image, image_name):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure(f'Histogram of {image_name}')
    plt.scatter(range(len(histogram)), histogram, marker='o')
    plt.xlabel('Grayscale Value'), plt.ylabel('Magnitude')
    plt.title(f'Histogram of {image_name}')
    plt.xlim([0, 255]), plt.grid(True)
    return plt

def contour_detection(hairs_only):
    contours, _ = cv2.findContours(hairs_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours: {len(contours)}")
    return len(contours), contours

def root_length(root_only):
    contours, _ = cv2.findContours(root_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert contour points to a NumPy array
    contour_array = np.array(contours).reshape(-1, 2)

    # Extract x coordinates
    x_coords = contour_array[:, 0]

    # Find the minimum and maximum x,y values
    min_x, min_y = np.min(x_coords), contour_array[np.argmin(x_coords), 1]
    max_x, max_y = np.max(x_coords), contour_array[np.argmax(x_coords), 1]

    length = math.sqrt((max_x-min_x)**2 + (max_y-min_y)**2)
    print(f"Root length: {length}")

    return length

def roots_hair_density(root_length, hairs_num):
    print(f"Root hairs density: {hairs_num/root_length}")
    return hairs_num/root_length

desired_ratio = 0.8667385578155518

path = r'/Users/omercohen/PycharmProjects/FinalProject/Bell_Images/'
image_list = ['SR_bicubic_p1_x2', 'SR_bicubic_p1_x3', 'SR_bicubic_p1_x4', 'SR_bicubic_p1_x8', 'SR_P1_X2', 'SR_P1_X3',
         'SR_P1_X4', 'SR_P1_X8']
# image = 'LR_P1'
image = image_list[1]
scale_factor = int(image[-1:])
color_image = cv2.imread(path+image + '.png')
scaling_factor = color_image.shape[0] / 438
# Based on SR_bicubic_p1_x2 image


# Step 1: Convert into Grayscale
gray_image = color2gray(color_image)
plt.figure(image)
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title(image)
plt.axis('off')

# Step 2: Apply Gaussian Filter (Un-sharp Mask)
sharpened_image = unsharp_mask(gray_image)
# plt.figure('Sharpened Image')
plt.subplot(3, 3, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
# plt = histogram_plot(sharpened_image, 'Sharpened Image')

# Step 3: First order filter - remove most of the background
first_order_filter = first_order_filter(sharpened_image, desired_ratio)
# plt.figure('First-Order Filter')
plt.subplot(3, 3, 3)
plt.imshow(first_order_filter, cmap='gray')
plt.title('First-Order Filter')
plt.axis('off')
# plt = histogram_plot(first_order_filter, 'First-Order Filter')

# Step 4: Apply Mean Shift filtering on the color image
filtered_image = mean_shift_filter(first_order_filter)
plt.subplot(3, 3, 4)
plt.imshow(filtered_image, cmap='gray')
plt.title(' Mean Shift Filter')
plt.axis('off')

# Step 5: Apply Thresholding using Otsu
segmented_image = thresholding(filtered_image)
# plt.figure('Thresholding')
plt.subplot(3, 3, 5)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

# # Step 6: Apply Morphological Operators
# segmented_image = morphology_filter(filtered_image)
# plt.figure('Morphological Operators')
# plt.subplot(3, 4, 6)
# plt.imshow(segmented_image, cmap='gray')
# plt.title('Morphological Operators')
# plt.axis('off')

# # Step 7: Fine-Tuning by removing non-related clusters
# segmented_image = fine_tuning(filtered_image)
# # plt.figure('Segmented Image')
# plt.subplot(3, 4, 7)
# plt.imshow(segmented_image, cmap='gray')
# plt.title('Segmented Image')
# plt.axis('off')

# Step 8: Filter the Root Only from the Segmented Image
root_only = root_only(segmented_image) #TODO : change back to segmented_image
# plt.figure('Root Only Image')
plt.subplot(3, 3, 6)
plt.imshow(root_only, cmap='gray')
plt.title('Root Only Image')
plt.axis('off')

# Step 9: Filter the Root-Hairs Only from the Segmented Image
hairs_only = cv2.subtract(segmented_image, root_only)
# plt.figure('Root-Hairs Only Image')
plt.subplot(3, 3, 7)
plt.imshow(hairs_only, cmap='gray')
plt.title('Root-Hairs Only Image')
plt.axis('off')

# Step 10: Contour Detection
hairs_num, hairs_contours = contour_detection(hairs_only)

# Step 11: Calculate Root Hair Density
root_length = root_length(root_only)
density = roots_hair_density(root_length, hairs_num)

# Draw contours on the original image for visualization
image_with_contours = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, hairs_contours, -1, (0, 255, 0), 2)
# plt.figure('Root Hair Contours on The Sharpened Image')
plt.subplot(3, 3, 8)
plt.imshow(image_with_contours)
plt.title(f'{hairs_num} Root Hair Contours')
plt.axis('off')

plt.tight_layout()
plt.show()


#TODO: remove this part :
# # This part is about the fabric images
# true_hairs_number = int(image[14:16])
# image_number = image[1:2]
# print(f'Image number {image_number} originally have {true_hairs_number} hairs root and we discover {hairs_num}. Accuracy of {(hairs_num/true_hairs_number)*100}%')
#
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# # Plot the root image
# axs[0].imshow(root_only, cmap='gray')
# axs[0].set_title('Root')
# axs[0].axis('off')
#
# # Plot the hairs image
# axs[1].imshow(image_with_contours, cmap='gray')
# axs[1].set_title('Hairs')
# axs[1].axis('off')
#
# # Plot the fabric picture
# axs[2].imshow(gray_image, cmap='gray')
# axs[2].set_title('Fabric Picture')
# axs[2].axis('off')
#
# # Adjust the layout
# plt.tight_layout()
# plt.show()