import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
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

def minutiae_at(pixels, i, j, kernel_size):
    """
    https://airccj.org/CSCP/vol7/csit76809.pdf pg93
    Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
    Then the crossing number algorithm will look at 3x3 pixel blocks:

    if middle pixel is black (represents ridge):
    if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
    if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation

    :param pixels:
    :param i:
    :param j:
    :return:
    """
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:

        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
                   (0, 1),  (1, 1),  (1, 0),            # p8    p4
                  (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2   p3
                   (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8      p4
                  (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6   p5

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values)-1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"

def calculate_minutiaes(im, kernel_size=3):
    biniry_image = np.zeros_like(im)
    biniry_image[im<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

    # iterate each pixel minutia
    for i in range(1, x - kernel_size//2):
        for j in range(1, y - kernel_size//2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae != "none":
                cv2.circle(result, (i,j), radius=2, color=colors[minutiae], thickness=2)

    return result

def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    https://books.google.pl/books?id=1Wpx25D8qOwC&lpg=PA120&ots=9wRY0Rosb7&dq=poincare%20index%20fingerprint&hl=pl&pg=PA120#v=onepage&q=poincare%20index%20fingerprint&f=false
    :param i:
    :param j:
    :param angles:
    :param tolerance:
    :return:
    """
    cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
            (0, 1),  (1, 1),  (1, 0),            # p8    p4
            (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):

        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"

def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop" : (0, 0, 255), "delta" : (0, 128, 255), "whorl": (255, 153, 255)}

    for i in range(3, len(angles) - 2):             # Y
        for j in range(3, len(angles[i]) - 2):      # x
            # mask any singularity outside of the mask
            mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W*5)**2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    cv2.rectangle(result, ((j+0)*W, (i+0)*W), ((j+1)*W, (i+1)*W), colors[singularity], 3)

    return result

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

# minutiaes en singularities op de thinned image
minutias = calculate_minutiaes(thinned)
#singularities_img

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






