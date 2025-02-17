import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.morphology import thin
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


# de zogezegde "live" image 
input = cv2.imread("./thumb2.jpg", 2)

input = cv2.resize(input, None, fx=1, fy=1)

# omzetten naar grijswaarden
# equ = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

# # # equ = cv2.equalizeHist(input)


# Contrastverbetering met CLAHE** (beter dan histogram equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
input = clahe.apply(input)

# ruis verwijderen met een median blur**
input = cv2.medianBlur(input, 3)

## gabor filtering
w_45 =  0.5
w_135 = 0.5

kernel_size = 24
img_45 = cv2.filter2D(input,cv2.CV_64F,cv2.getGaborKernel( (kernel_size,kernel_size), 2, np.deg2rad(45),np.pi/4,0.5,0 ))
img_135 = cv2.filter2D(input,cv2.CV_64F,cv2.getGaborKernel( (kernel_size,kernel_size), 2, np.deg2rad(135),np.pi/4,0.5,0 ))


filtered = img_45*w_45+img_135*w_135
filtered = filtered/np.amax(filtered)*255

# negatieve waarden verwijderen zodat het in de rabge oast 
filtered = np.maximum(filtered, 0)  
# normaliseren
filtered = filtered / np.amax(filtered) * 255  
filtered = np.uint8(filtered) 


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# Ridge detection
def detect_ridges(gray, sigma=0.1):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


# Visualisatie van de originele afbeelding en de gedetecteerde randen
def plot_images(*images):
    images = list(images)
    n = len(images)

    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    # plt.show()



max_ridges, min_ridges = detect_ridges(filtered, sigma=0.15)
# plot_images(filtered, max_ridges, min_ridges)

ridge_thresh = threshold_otsu(-min_ridges)
ridge_binary = (-min_ridges) > ridge_thresh

# filtering om ruis te verwijderen
ridge_binary = ridge_binary.astype(np.uint8) * 255
kernel = np.ones((3, 3), np.uint8)
ridge_binary = cv2.morphologyEx(ridge_binary, cv2.MORPH_CLOSE, kernel)

# skeletonization en thinning op ridge-detectie
skeleton = skeletonize(ridge_binary > 0, method='lee').astype(np.float32)
thinned = thin(ridge_binary).astype(np.float32)

# Visualisatie
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(filtered, cmap='gray')
ax[0].set_title('Filtered Image')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap='gray')
ax[1].set_title('Skeleton (Ridges)')
ax[1].axis('off')

ax[2].imshow(thinned, cmap='gray')
ax[2].set_title('Thinned (Ridges)')
ax[2].axis('off')

plt.tight_layout()
plt.show()

# als afbeelding opslaan 
cv2.imwrite("skeleton2.png", (skeleton * 255).astype(np.uint8))  

# als np array opslaan = sneller en exacter + voor berekeningen 
np.save("skeleton2.npy", skeleton)

## hierboven maar anders: 

    # thresh = threshold_otsu(filtered)
    # binary_image = filtered > thresh

    # # Morphological filtering om ruis te verwijderen**
    # binary_image = binary_image.astype(np.uint8) * 255
    # kernel = np.ones((3, 3), np.uint8)
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # Verwijder kleine gaten

    # # perform skeletonization
    # skeleton = skeletonize(image=binary_image > 0, method='lee')


    # skeleton = skeleton.astype(np.float32)



    # img = thin(binary_image)
    # display results
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)

    # ax = axes.ravel()



    # ax[2].imshow(img, cmap=plt.cm.gray)
    # ax[2].axis('off')
    # ax[2].set_title('thinned', fontsize=20)
    # skeleton = skeleton.astype(np.float32)

    # ax[1].imshow(skeleton, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('skeleton', fontsize=20)

    # ax[0].imshow(filtered, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('original', fontsize=20)
    # fig.tight_layout()
    # plt.show()






