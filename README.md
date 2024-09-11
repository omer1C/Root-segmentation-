# Root Segmentation

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

# Example Usage 

## Clone The Repository:
```
git clone https://github.com/omer1C/Root-segmentation-.git
```
## Requirements 
Make sure navigate to the Root-segmentation- dir, you can use:
```
cd Root-segmentation-/
```
To install the requirements please run:
```
pip install -r requirements.txt
```
## Run The Code
In order to run the Bell Pepper algorithm add your save path and run:
```
python3 main.py --plant_type bell_pepper --save_path YOUR_SAVE_PATH/bell_example
```

In order to run the Arbidiopsis algorithm add your save path and run:
```
python3 main.py --plant_type arbidiopsis --save_path YOUR_SAVE_PATH/arb_example
```
# Project Goals

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
<img width="650" alt="image" src="https://github.com/user-attachments/assets/b7dfcd3c-520d-4f71-b202-994067556d72">


# Proposed Algorithm:
### Original Image : 
| ![Bell Pepper](https://github.com/user-attachments/assets/776b94c7-faec-4c67-9fe3-695dac3fce9f) | ![Arabidopsis](https://github.com/user-attachments/assets/ceef5e3b-f7a9-455b-bffb-7f90e6cd4e4e) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

                                                                             

## Step 1 : Convert to Gray Scale Image and Sharpen the Image
Convert the color image into grayscale and apply un-sharp filter to create sharpened image, which consists of subtracting the output of a Gaussian filter from the grayscale image:
sharpened=α ∙grayscale-β∙blured,where α=1.5,β=0.5
### Gray scale images : 
| ![Bell Pepper](https://github.com/user-attachments/assets/a2f541da-257e-47bb-aca8-0e351c3aa381) | ![Arabidopsis](https://github.com/user-attachments/assets/2f6fb3ad-11fc-455c-a34d-8ace56bb779d) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

### Sharpened Images : 
| ![Bell Pepper](https://github.com/user-attachments/assets/a132fdcf-e3bd-41b3-9438-2dc98662608f) | ![Arabidopsis](https://github.com/user-attachments/assets/7791c28c-34cd-4954-af59-060b277dc19b) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

## Step 2 : First Order Filter
Apply First-order filter to remove most portions of the background.
The assumption is that the largest group of pixels within the histogram is related to the background, and by the nature of the root, the pixel group related to it will be spread in the brighter grayscale range.
We set a threshold value that corresponds to 0.78% of the most common pixel value and has a higher grayscale value than the most common pixel value, assigning all other pixels to 0 (black). 

#### Bell Pepper Histogram :
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

## Step 3 : Mean Shift Filter (Bell Pepper Only) 
Dividing the image into clusters by apply Mean-Shift filter to find clusters within the image.
#### Mean Shift Filter:
![image](https://github.com/user-attachments/assets/28fddc0a-2ef2-4bd8-a9d1-64453ca7c109)


## Step 4 : Extract the Root
Apply thresholding using Otsu algorithm, and find the largest cluster, which we assume related to the root.

#### Otsu Algorithm Output : 
| ![Bell Pepper](https://github.com/user-attachments/assets/51c79baf-34d6-4520-9a0c-4a74fb24cf4d) | ![Arabidopsis](https://github.com/user-attachments/assets/a4c6f846-f348-4dec-bdf6-8f7d89e48ca9) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

#### Extracting the Lagest Component :
| ![Bell Pepper](https://github.com/user-attachments/assets/de95ca46-7de2-4ad5-aa38-5f3e3a46f442) | ![Arabidopsis](https://github.com/user-attachments/assets/b18f7349-919e-4e3a-bba4-2bc576dcb119) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

## Step 5 : Isolating the Main Root
Filter the root only from the segmented image by applying several iterations of Morphological Opening operator to remove most of the root-hairs and then applying several iterations of Closing operator for complementary corrections.
#### Bell Pepper :
![image](https://github.com/user-attachments/assets/566da957-ac4e-4ab5-9c7e-c49fb68cb5aa)

#### Arabidopsis :
![image](https://github.com/user-attachments/assets/c20b10f0-d60e-46b1-889e-6dc803aed29c)

## Step 6 : Extracting and Counting the Hairs
Filter the root-hairs only from the segmented image by subtracting the root only from the segmented image. 
#### Hairs Only :
| ![Bell Pepper](https://github.com/user-attachments/assets/22a6f5ac-d751-4d8e-9745-7f0e1e13ee8a) | ![Arabidopsis](https://github.com/user-attachments/assets/04128c5c-453d-4743-9e27-dd1b938b4776) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

#### Final Results :
| ![Bell Pepper](https://github.com/user-attachments/assets/e5c792a1-2ad2-43e7-be2f-ae32d655daab) | ![Arabidopsis](https://github.com/user-attachments/assets/ca344d1b-117e-4712-ad09-c715362b19c1) |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| **Bell Pepper**                                                                                 | **Arabidopsis**                                                                                 |

