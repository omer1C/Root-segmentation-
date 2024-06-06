# Root-segmentation

This project is about segmentation of images taken using a pipe camera which took pictures from inside the soil of roots in different soil types.
By analyzing the root (number of hairs, density, length of the root, etc.) it is possible to determine the quality of the soil for agricultural purposes.

# Bell-Pepper Root Algorithm:
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


