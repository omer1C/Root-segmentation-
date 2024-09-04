# Root segmentation

Within the era of climate changes and increase in global demographics, the demand for reliable
food sources, and agricultural products in particular, is a constant need.
The purpose of this project is to create a robust method for measuring major root trait, and by that
providing insights into plant health. Our objective is to develop an algorithm to accurately detect
and count root hairs and test the impact of super-resolution. The innovation lies in the idea to
remove the main root from the segmented image and thereby isolate the root hairs. We propose a
method which first performs image segmentation over the raw image to extract the root with its’
branches from the soil background, then uses morphological operators to remove the main root
without branches and finally detects and counts root hairs by mapping each root hair to a separate
contour. The goal is to achieve over 80% precision in detecting and counting root hairs, where
super-resolution images expected to outperform non-super-resolution images.

# Project goals

The project aims to develop a non-invasive method for assessing plant health and nutrient uptake
by leveraging agricultural root image segmentation. Our goal is to propose a solution that
determines the plant's condition in situ—whether it receives adequate nutrition, water, and other
vital factors—without uprooting it. This approach utilizes image segmentation and classic
algorithms, avoiding neural networks, to accurately separate the root structure from the soil
background. The focus is on detecting and counting root hairs, crucial indicators of root health and
nutrient absorption efficiency. This methodology seeks to enhance agricultural practices by
providing insights into plant health directly from field imagery, facilitating informed decisions for
optimal crop management. Our project aims to achieve a precision score exceeding 80% in the task
of detecting and counting root hairs. Additionally, we seek to determine if super-resolution images
consistently yield higher accuracy scores compared to non-super-resolution images on average.

# Bell-Pepper Root Algorithm:
### Original Image : 
![LR_P1](https://github.com/omer1C/Root-segmentation-/assets/135855862/00130b5c-8064-4c07-9cd5-876d980c90b5)

## step 1:
Convert the color image into grayscale and apply Un-sharp filter to create sharpened image, which consists of subtracting the output of a Gaussian filter from the grayscale image:
sharpened=α ∙grayscale-β∙blured,where α=1.5,β=0.5

### Figure for demonstration:
<img width="288" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/fcfb7fb5-3bde-4289-b84c-f5ef6a62c9d3">

## step 2 : 
Apply First-order filter to remove most portions of the background.
The assumption is that the largest group of pixels within the histogram is related to the background, and by the nature of the root, the pixel group related to it will be spread in the brighter grayscale range.
Therefore, we defined cutoff value to be the first value that gets half (inspired by -3db) of the greatest quantity, which related to the background, and has brighter gray value, and set all pixels with lower gray value to 0 (black).
### Figures for demonstration:
#### Histogram:
<img width="295" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/734b215e-e725-432c-8414-3b789d1af9e5">

#### First-order filter output:
<img width="321" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/8726209b-ffb2-412e-9c20-9891351b9826">

## step 3 : 
Apply Mean-Shift filter to find clusters within the image.
### Figure for demonstration:
<img width="326" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/6e5a0176-f708-4b0b-ab96-a3757f74e3a6">

## step 4 : 
Apply thresholding using Otsu algorithm, and find the largest cluster, which we assume related to the root.
### Figure for demonstration:
<img width="341" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/450ec57e-5a82-4f35-b3e4-023af5c8427a">

## step 5 : 
Apply Morphological Opening operator on the image to remove outer noise not related to the root and Closing operator to fill gaps within the root.
### Figure for demonstration:
<img width="479" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/4be7a569-2db5-4fe0-b828-e043e5342982">

## step 6 : 
Apply fine-tuning to remove outer noise by finding again the largest cluster.
This is the final step to get a segmented image.
### Figure for demonstration:
<img width="541" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/2a690d7c-14d3-48f1-ac21-9980fc9ecf38">

## step 7 : 
Filter the root only from the segmented image by applying several iterations of Morphological Opening operator to remove most of the root-hairs and then applying several iterations of Closing operator for complementary corrections.
### Figure for demonstration:
<img width="385" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/01f43393-cc63-449f-8b9b-6f65e92e61fe">

## step 8 : 
Filter the root-hairs only from the segmented image by subtracting the root only from the segmented image. 
### Figure for demonstration:
<img width="405" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/74a5c89a-147e-4493-aefe-818d34d71c75">

### step 9 :
Count the number of root-hairs by detecting the number of contours.

### Figure for demonstration:
<img width="392" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/026d9dea-34f9-459d-bb00-75764c1318aa">


Calculate the root’s length, by calculating the Euclidian distance between the leftmost and rightmost pixels, assuming that the root is relatively linear shaped.


Calculate the root-hair density by dividing the number of root-hairs by the root’s length.

<img width="246" alt="image" src="https://github.com/omer1C/Root-segmentation-/assets/135855862/fefe0351-ad71-4fc4-80c7-571dc5306147">


## Note : 
We should have a second thought of applying Mean-Shift filter at step #3. The result of applying thresholding (step #4) right after first-order filter (step #2) may be better.

# ARBIDIOPSIS Root Algorithm : 

This algorithm based on the same principles of the Bell-Pepper algorithm but there are several changes : 

## step 2 : 
we defined cutoff value to be the first value that gets 35% (inspired by -3db) of the greatest quantity, which related to the background, and has brighter gray value, and set all pixels with lower gray value to 0 (black).

### Figures for demonstration:
Histogram:     
![img.png](README_IMAGES/img.png) ![img_1.png](README_IMAGES/img_1.png)

Then we skip on applying Mean-Shift filter and moved directly to Otsu algorithm. 

## The results : 
### The Original Image : 
<img src="https://github.com/omer1C/Root-segmentation-/blob/a5351ec844ee0c8bd58af862f47ed8e3d4faad54/README_IMAGES/img_2.png" alt="img_2.png" width="400"/>

### Final Bitwise fine-tuning Image : 
<img src="https://github.com/omer1C/Root-segmentation-/blob/a5351ec844ee0c8bd58af862f47ed8e3d4faad54/README_IMAGES/img_3.png" alt="img_2.png" width="400"/>

### Hairs detection : 
Finding the haris happens in the same way:

### Root Only :

<img src="https://github.com/omer1C/Root-segmentation-/blob/a5351ec844ee0c8bd58af862f47ed8e3d4faad54/README_IMAGES/img_5.png" alt="img_2.png" width="400"/>

### Hairs Only : 

<img src="https://github.com/omer1C/Root-segmentation-/blob/a5351ec844ee0c8bd58af862f47ed8e3d4faad54/README_IMAGES/img_4.png" alt="img_2.png" width="400"/>

