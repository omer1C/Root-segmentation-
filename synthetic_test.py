# # This part is about the synthetic images
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