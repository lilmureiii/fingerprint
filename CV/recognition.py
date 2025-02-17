import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# de zogezegde "live" image 
input = cv2.imread("./thumb.jpg", cv2.IMREAD_GRAYSCALE)

input = cv2.resize(input, None, fx=1, fy=1)

# omzetten naar grijswaarden
# equ = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

# # # equ = cv2.equalizeHist(input)


# 🔹 **Contrastverbetering met CLAHE** (beter dan histogram equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
input = clahe.apply(input)

# 🔹 **Ruis verwijderen met een median blur**
input = cv2.medianBlur(input, 3)
# grijswaarden omzetten naar 3 kanalen  anders kan je niet stacken
# equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

# res = np.hstack((input, equ))
# res = equ
# res = equ
        # cv2.namedWindow("result_equ.jpg", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("result_equ.jpg", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("img", res)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ## SKIP - orientation map
        # fd, hog_image = hog(res, orientations=9, pixels_per_cell=(24, 24),
        #                 cells_per_block=(1, 1), visualize=True, multichannel=None,feature_vector=False
        #               )

## gabor filtering
w_45 =  0.5
w_135 = 0.5

kernel_size = 31
img_45 = cv2.filter2D(input,cv2.CV_64F,cv2.getGaborKernel( (kernel_size,kernel_size), 2, np.deg2rad(45),np.pi/4,0.5,0 ))
img_135 = cv2.filter2D(input,cv2.CV_64F,cv2.getGaborKernel( (kernel_size,kernel_size), 2, np.deg2rad(135),np.pi/4,0.5,0 ))


filtered = img_45*w_45+img_135*w_135
filtered = filtered/np.amax(filtered)*255

# negatieve waarden verwijderen zodat het in de rabge oast 
filtered = np.maximum(filtered, 0)  
# normaliseren
filtered = filtered / np.amax(filtered) * 255  
filtered = np.uint8(filtered) 


thresh = threshold_otsu(filtered)
binary_image = filtered > thresh

# 🔹 **Morphological filtering om ruis te verwijderen**
binary_image = binary_image.astype(np.uint8) * 255
kernel = np.ones((3, 3), np.uint8)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # Verwijder kleine gaten
# perform skeletonization
skeleton = skeletonize(binary_image > 0)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(filtered, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

skeleton = skeleton.astype(np.float32)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

# fig.tight_layout()
# print(skeleton.dtype, skeleton.shape, skeleton.min(), skeleton.max())

plt.show()



# # matching with db
# fingerprint_db_img = cv2.imread("./fingerprint2.jpg")
# if equ is None:
#     print(f"Error: {fingerprint_db_img} could not be loaded!")
#     exit(1)


# sift = cv2.SIFT_create()
# keypoints_1, descriptors_1 = sift.detectAndCompute(equ, None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_db_img, None)

# print(f"Keypoints in fingerprint_img: {len(keypoints_1)}")
# print(f"Keypoints in fingerprint_db_img: {len(keypoints_2)}")


# if descriptors_1 is None or descriptors_2 is None:
#     print("Error: No descriptors found in the image")
#     exit(1)

# # finding nearest match with KNN algorithm 

# flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict())
# matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
# match_points = []
   
# for p, q in matches:
#     if p.distance < 0.75*q.distance:
#         match_points.append(p)

# # keypoint detection / detecting the fingerprint matched ID
# keypoints = 0
# if len(keypoints_1) <= len(keypoints_2):
#     keypoints = len(keypoints_1)            
# else:
#     keypoints = len(keypoints_2)

# # if (len(match_points) / keypoints)>0.95:
# print("% match: ", len(match_points) / keypoints * 100)

# # geen idee waarom 
# print("Fingerprint ID: " + str(keypoints)) 

# result = cv2.drawMatchesKnn(equ, keypoints_1, fingerprint_db_img, 
#                             keypoints_2, matches1to2=[match_points], outImg=None, matchColor=(0, 155, 0), 
#                              singlePointColor=(0, 255, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# result = cv2.resize(result, None, fx=1, fy=1)

# de twee afbeeldingen bij elkaar zetten en zo vergelijken zodat je een overzicht hebt 

# cv2.namedWindow("result.jpg", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("result.jpg", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.imshow("result.jpg", result)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ----------------------------------------------------
    # from utils.crossing_number import calculate_minutiaes
    # from skimage.morphology import skeletonize as skelt
    # from skimage.morphology import thin
    # def skeletonize(image_input):
    #     """
    #     https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
    #     Skeletonization reduces binary objects to 1 pixel wide representations.
    #     skeletonize works by making successive passes of the image. On each pass, border pixels are identified
    #     and removed on the condition that they do not break the connectivity of the corresponding object.
    #     :param image_input: 2d array uint8
    #     :return:
    #     """
    #     image = np.zeros_like(image_input)
    #     image[image_input == 0] = 1.0
    #     output = np.zeros_like(image_input)

    #     skeleton = skelt(image)

    #     """uncomment for testing"""
    #     thinned = thin(image)
    #     thinned_partial = thin(image, max_iter=25)
        
    #     def minu_(skeleton, name):
    #         cv2.imshow('thin_'+name, output)
    #         cv2.bitwise_not(output, output)
    #         minutias = calculate_minutiaes(output, kernel_size=5); cv2.imshow('minu_'+name, minutias)
    #     # minu_(output, 'skeleton')
    #     # minu_(output, 'thinned')
    #     # minu_(output, 'thinned_partial')
    #     # cv.waitKeyEx()

    #     output[skeleton] = 255
    #     cv2.bitwise_not(output, output)

    #     return output


    # def thinning_morph(image, kernel):
    #     """
    #     Thinning image using morphological operations
    #     :param image: 2d array uint8
    #     :param kernel: 3x3 2d array unint8
    #     :return: thin images
    #     """
    #     thining_image = np.zeros_like(image)
    #     img = image.copy()

    #     while 1:
    #         erosion = cv2.erode(img, kernel, iterations = 1)
    #         dilatate = cv2.dilate(erosion, kernel, iterations = 1)

    #         subs_img = np.subtract(img, dilatate)
    #         cv2.bitwise_or(thining_image, subs_img, thining_image)
    #         img = erosion.copy()

    #         done = (np.sum(img) == 0)

    #         if done:
    #           break

    #     # shift down and compare one pixel offset
    #     down = np.zeros_like(thining_image)
    #     down[1:-1, :] = thining_image[0:-2, ]
    #     down_mask = np.subtract(down, thining_image)
    #     down_mask[0:-2, :] = down_mask[1:-1, ]
    #     cv2.imshow('down', down_mask)

    #     # shift right and compare one pixel offset
    #     left = np.zeros_like(thining_image)
    #     left[:, 1:-1] = thining_image[:, 0:-2]
    #     left_mask = np.subtract(left, thining_image)
    #     left_mask[:, 0:-2] = left_mask[:, 1:-1]
    #     cv2.imshow('left', left_mask)

    #     # combine left and down mask
    #     cv2.bitwise_or(down_mask, down_mask, thining_image)
    #     output = np.zeros_like(thining_image)
    #     output[thining_image < 250] = 255

    #     return output
# ----------------------------------------------------
