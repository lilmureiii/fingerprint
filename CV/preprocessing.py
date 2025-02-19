import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.morphology import thin
import numpy as np

class FingerprintPreprocessor:
    def __init__(self, image_path):
        """Initialiseer met een afbeelding."""
        self.image = cv2.imread(image_path, 2)
        self.image = cv2.resize(self.image, None, fx=1, fy=1)
        if self.image is None:
            raise ValueError("Kon afbeelding niet laden. Controleer het pad.")
        self.preprocessed_img = None
        self.skeleton = None
        self.thinned = None
    
    def enhance_contrast(self): 
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.image = clahe.apply(self.image)

        # ruis verwijderen met een median blur**
        self.image = cv2.medianBlur(self.image, 3)

    def gabor_filtering(self,kernel_size=32, w_45=0.5, w_135=0.5):
        kernel_size = 32
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
        self.preprocessed_img = cv2.adaptiveThreshold(self.preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 10)


    def skeletonize_image(self): 
        ridge_thresh = threshold_otsu(-self.preprocessed_img)
        ridge_binary = (-self.preprocessed_img) > ridge_thresh

        ridge_binary = ridge_binary.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        ridge_binary = cv2.morphologyEx(ridge_binary, cv2.MORPH_CLOSE, kernel)

        self.skeleton = skeletonize(ridge_binary > 0, method='lee').astype(np.float32)
        self.thinned = thin(ridge_binary).astype(np.float32)

    def preprocess(self):
        """Voer de volledige preprocessing pipeline uit."""
        self.enhance_contrast()
        self.gabor_filtering()
        self.detect_ridges()
        self.skeletonize_image()

    def minutiae_at(self): 
        """
        nog aanvullen (na de self ook)
        """
        pass

    def calculate_minutiaes(self,  kernel_size=3): 
        """
        nog aanvullen
        """
        pass

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

    def visualize(self): 
        fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title('Filtered Image')
        ax[0].axis('off')

        ax[1].imshow(self.skeleton, cmap='gray')
        ax[1].set_title('Skeleton (Ridges)')
        ax[1].axis('off')

        ax[2].imshow(self.thinned, cmap='gray')
        ax[2].set_title('Thinned (Ridges)')
        ax[2].axis('off')

        ax[3].imshow(self.preprocessed_img, cmap='gray')
        ax[3].set_title('Binarized Image (Ridges)')
        ax[3].axis('off')

        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    preprocessor = FingerprintPreprocessor("./thumb.jpg")
    preprocessor.preprocess()
    preprocessor.visualize()

    