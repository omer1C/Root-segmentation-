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

# Approach, Design, and Solution 
To address the challenge of counting root hairs, our approach began by recognizing the importance of preprocessing the image effectively. Starting with segmentation yielded a clear representation of the root in white against a black background. Once we obtained this binary image and understood the root's structure, our next step focused on isolating the branching hairs from the main root. By identifying distinct contours within the segmented image, we aimed to accurately count the individual root hairs.

# Block Diagram 
<img width="650" alt="image" src="https://github.com/user-attachments/assets/6907dc64-255b-4a51-82fc-084b33fde398">

# Proposed Algorithm:
### Original Image : 
| ![Bell Pepper](https://github.com/user-attachments/assets/776b94c7-faec-4c67-9fe3-695dac3fce9f) | ![Arabidopsis](https://github.com/user-attachments/assets/ceef5e3b-f7a9-455b-bffb-7f90e6cd4e4e) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

                                                                             

## Step 1:
Convert the color image into grayscale and apply un-sharp filter to create sharpened image, which consists of subtracting the output of a Gaussian filter from the grayscale image:
sharpened=α ∙grayscale-β∙blured,where α=1.5,β=0.5
### Gray scale images : 
| ![Bell Pepper](https://github.com/user-attachments/assets/a2f541da-257e-47bb-aca8-0e351c3aa381) | ![Arabidopsis](https://github.com/user-attachments/assets/2f6fb3ad-11fc-455c-a34d-8ace56bb779d) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

### Sharpened images : 
| ![Bell Pepper](https://github.com/user-attachments/assets/a132fdcf-e3bd-41b3-9438-2dc98662608f) | ![Arabidopsis](https://github.com/user-attachments/assets/7791c28c-34cd-4954-af59-060b277dc19b) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

## Step 2 : 
Apply First-order filter to remove most portions of the background.
The assumption is that the largest group of pixels within the histogram is related to the background, and by the nature of the root, the pixel group related to it will be spread in the brighter grayscale range.
We set a threshold value that corresponds to 0.78% of the most common pixel value and has a higher grayscale value than the most common pixel value, assigning all other pixels to 0 (black). 

#### Bell Pepper Histogram:
| ![Before First Order Filter](https://github.com/user-attachments/assets/ed8fea03-242b-4fbf-9fe1-03138ce5f36c) | ![After First Order Filter](https://github.com/user-attachments/assets/86837dca-fd42-460f-b264-94319b016fac) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Before First Order Filter**                                                                                 | **After First Order Filter**                                                                                 |

#### Arabidopsis Histogram:
| ![Before First Order Filter](https://github.com/user-attachments/assets/54f4dbb3-dd66-4479-a1a2-94c7f4de36af) | ![After First Order Filter](https://github.com/user-attachments/assets/88cbdabf-c428-4d58-a535-75b318676a4a) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Before First Order Filter**                                                                                 | **After First Order Filter**                                                                                 |

### First Order Filter Output:
| ![Bell Pepper](https://github.com/user-attachments/assets/62ecc088-36a0-4d9c-9207-665ea04a206d) | ![Arabidopsis](https://github.com/user-attachments/assets/e99fcd9a-eb44-4137-af85-8c650f36ad55) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

## Step 3 (Bell Pepper only) : 
Dividing the image into clusters by apply Mean-Shift filter to find clusters within the image.
#### Mean Shift Filter:
![image](https://github.com/user-attachments/assets/28fddc0a-2ef2-4bd8-a9d1-64453ca7c109)


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

