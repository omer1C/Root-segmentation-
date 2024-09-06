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

def first_order_filter(sharpened_image):
    # Compute the histogram of the shifted grayscale image
    histogram = cv2.calcHist([sharpened_image], [0], None, [256], [0, 256])

    cutoff_val = find_gray_value_for_desired_ratio(histogram, desired_ratio)
    clear_values = range(0, cutoff_val)

    mask = np.ones_like(sharpened_image) * 255
    for value in clear_values:
        mask[sharpened_image == value] = 0

    first_order_filter = cv2.bitwise_and(sharpened_image, mask)

    return first_order_filter

def find_desired_ratio(first_order_filter):
    histogram = cv2.calcHist([first_order_filter], [0], None, [256], [0, 256])
    background = histogram.flatten()[0]
    total_pixels = np.sum(histogram)
    desired_ratio = background/total_pixels
    print(f'Desired background ratio: {desired_ratio}')

    return desired_ratio

def find_gray_value_for_desired_ratio(histogram, desired_ratio):
    # Step 1: Calculate the total number of pixels
    total_pixels = np.sum(histogram)

    # Step 2: Calculate the desired number of pixels with value 0
    desired_zero_pixels = total_pixels * desired_ratio

    # Step 3: Find the gray value using the cumulative sum
    cumulative_sum = np.cumsum(histogram)
    gray_value = np.searchsorted(cumulative_sum, desired_zero_pixels)

    return gray_value

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
    opening_mean = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel, iterations=2)

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
    temp_ratio = round(find_desired_ratio(segmented_image), 2)
    ratio = scaling_factor*segmented_ratio/temp_ratio
    print(f"Ratio: {ratio}, Open Iterations: {int(5*np.sqrt(ratio))}")
    root_only = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=int(5*np.sqrt(ratio)))
    root_only = cv2.morphologyEx(root_only, cv2.MORPH_CLOSE, kernel=np.ones((4, 4), np.uint8), iterations=int(5*ratio))
    root_only = cv2.morphologyEx(root_only, cv2.MORPH_OPEN, kernel=np.ones((4, 4), np.uint8), iterations=int(5*np.sqrt(ratio)))
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

image_list = ['arb_bicubic_x2', 'arb_bicubic_x3', 'arb_bicubic_x4', 'arb_bicubic_x8', 'arb_sr_x2', 'arb_sr_x3',
         'arb_sr_x4', 'arb_sr_x8']

# Base Case - LR Image
path = r'/Users/omercohen/PycharmProjects/FinalProject/Arb_Images/'
image = image_list[3]

desired_ratio = 0.8
segmented_ratio = 0.89

color_image = cv2.imread(path + image + '.png')
# scale_factor = int(image[-1:])
scaling_factor = color_image.shape[0] / 816


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
first_order_filter = first_order_filter(sharpened_image)
# plt.figure('First-Order Filter')
plt.subplot(3, 3, 3)
plt.imshow(first_order_filter, cmap='gray')
plt.title('First-Order Filter')
plt.axis('off')
# plt = histogram_plot(first_order_filter, 'First-Order Filter')

# Step 4: Apply Thresholding using Otsu
segmented_image = thresholding(first_order_filter)
# plt.figure('Thresholding')
plt.subplot(3, 3, 4)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

# Step 5: Filter the Root Only from the Segmented Image
root_only = root_only(segmented_image)
# plt.figure('Root Only Image')
plt.subplot(3, 3, 7)
plt.imshow(root_only, cmap='gray')
plt.title('Root Only Image')
plt.axis('off')

# Step 6: Filter the Root-Hairs Only from the Segmented Image
hairs_only = cv2.subtract(segmented_image, root_only)
# plt.figure('Root-Hairs Only Image')
plt.subplot(3, 3, 8)
plt.imshow(hairs_only, cmap='gray')
plt.title('Root-Hairs Only Image')
plt.axis('off')

# Step 7: Contour Detection
hairs_num, hairs_contours = contour_detection(hairs_only)

# Step 9: Calculate Root Hair Density
root_length = root_length(root_only)
density = roots_hair_density(root_length, hairs_num)

# Draw contours on the original image for visualization
cv2.drawContours(color_image, hairs_contours, -1, (0, 255, 0), 2)
# plt.figure('Root Hair Contours on The Sharpened Image')
plt.subplot(3, 3, 9)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title(f'{hairs_num} Root Hair Contours')
plt.axis('off')

plt.tight_layout()
# Save the figure to a file
# plt.savefig(os.path.join(path, f'{image} processing.png'), dpi=500, bbox_inches='tight')
plt.show()
