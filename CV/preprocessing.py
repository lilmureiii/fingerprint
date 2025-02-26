import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.filters import threshold_otsu
from skimage.filters import meijering
from skimage.morphology import thin
import numpy as np

class FingerprintPreprocessor:
    def __init__(self, image_path):
        """Initialiseer met een afbeelding."""
        # self.image = cv2.imread(image_path, 2)
        self.image = image_path
        if self.image is None:
            raise ValueError("Kon afbeelding niet laden. Controleer het pad.")
        self.preprocessed_img = None
        self.skeleton = None
        self.thinned = None
        self.minutiaes = None
    
    def enhance_contrast(self): 
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.image = clahe.apply(self.image)

        # ruis verwijderen met een median blur**
        self.image = cv2.medianBlur(self.image, 3)

    def gabor_filtering(self,kernel_size=32, w_45=0.5, w_135=0.5):
        kernel_size = 35
        img_45 = cv2.filter2D(self.image,cv2.CV_64F,cv2.getGaborKernel((kernel_size,kernel_size), 2, np.deg2rad(45),np.pi/4,0.5,0 ))
        img_135 = cv2.filter2D(self.image,cv2.CV_64F,cv2.getGaborKernel((kernel_size,kernel_size), 2, np.deg2rad(135),np.pi/4,0.5,0 ))

        filtered = img_45*w_45+img_135*w_135
        filtered = filtered/np.amax(filtered)*255

        # negatieve waarden verwijderen zodat het in de rabge past 
        filtered = np.maximum(filtered, 0)  

        # normaliseren
        filtered = filtered / np.amax(filtered) * 255
        self.preprocessed_img = np.uint8(filtered) 
    
    def detect_ridges(self): 
        self.preprocessed_img = cv2.adaptiveThreshold(self.preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10)
        # Meijering-filter om richels te versterken
        # ridge_enhanced = meijering(self.preprocessed_img, sigmas=range(1, 3), black_ridges=False)
        # self.preprocessed_img = (ridge_enhanced * 255).astype(np.uint8)
    

    def skeletonize_image(self): 
        ridge_thresh = threshold_otsu(-self.preprocessed_img) * 1.8
        ridge_binary = (-self.preprocessed_img) > ridge_thresh

        ridge_binary = ridge_binary.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        ridge_binary = cv2.morphologyEx(ridge_binary, cv2.MORPH_CLOSE, kernel)

        # self.skeleton = skeletonize(ridge_binary > 0, method='lee').astype(np.float32)
        self.thinned = thin(ridge_binary).astype(np.float32)
        return self.thinned
    
    def skeleton_post_process(self, skeleton):
        kernel = np.ones((5, 5), np.uint8)
        thickened_skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_DILATE, kernel)
    
        return thickened_skeleton

    def minutiae_at(self,pixels, i, j, kernel_size):
        if pixels[i][j] == 1:
            cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
            values = [pixels[i + l][j + k] for k, l in cells]
            crossings = 0
            for k in range(len(values) - 1):
                if values[k] != values[k + 1]:
                    crossings += 1
            crossings //= 2

            if crossings == 1:
                return "ending"
            if crossings == 3:
                return "bifurcation"
        return "none"

    def calculate_minutiaes(self, kernel_size=3):
        (y, x) = self.thinned.shape
        result = cv2.cvtColor(self.thinned, cv2.COLOR_GRAY2RGB)
        colors = {"ending": (255, 0, 0), "bifurcation": (0, 255, 0)}
        for i in range(1, x - kernel_size // 2):
            for j in range(1, y - kernel_size // 2):
                minutiae = self.minutiae_at(self.thinned, j, i, kernel_size)
                if minutiae != "none":
                    cv2.circle(result, (i, j), radius=5, color=colors[minutiae], thickness=-1)
        self.minutiaes = result

    def poincare_index_at(self): 
        """
        nog aanvullen (na de self ook)
        """
        pass

    def calculate_singularities(self): 
        """
        nog aanvullen (na de self ook)
        """
        pass

    def apply_gaussian_blur(self, skeleton, kernel_size=5):
        blurred = cv2.GaussianBlur(skeleton, (kernel_size, kernel_size), 0)
        return blurred

    def preprocess(self):
        """Voer de volledige preprocessing pipeline uit."""
        self.enhance_contrast()
        self.gabor_filtering()
        self.detect_ridges()
        
        self.skeleton_post_process(self.skeletonize_image())



    def visualize(self,thinned): 
        fig, axes = plt.subplots(1, 5, figsize=(12, 4), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title('Filtered Image')
        ax[0].axis('off')

        # ax[1].imshow(skeleton, cmap='gray')
        # ax[1].set_title('Skeleton (Ridges)')
        # ax[1].axis('off')

        ax[2].imshow(thinned, cmap='gray')
        ax[2].set_title('Thinned (Ridges)')
        ax[2].axis('off')

        ax[3].imshow(self.preprocessed_img, cmap='gray')
        ax[3].set_title('Binarized Image (Ridges)')
        ax[3].axis('off')

        ax[4].imshow(self.minutiaes, cmap='gray')
        ax[4].set_title('Minutiaes')
        ax[4].axis('off')

        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    preprocessor = FingerprintPreprocessor()
    thinned = preprocessor.preprocess()
    # minutiaes = preprocessor.calculate_minutiaes

