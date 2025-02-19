import cv2
import os
from preprocessing import FingerprintPreprocessor
import matplotlib.pyplot as plt
import numpy as np

path1 = "./thumb.jpg"
path2 = "./thumb2.jpg"

# # preprocessor-object aan en verwerk de afbeelding
# fingerprint = FingerprintPreprocessor(path)
# thinned, skeleton = fingerprint.preprocess()


def match_minutiae(minutiae1, minutiae2, threshold=10):
    """
    compare minutiae lists of 2 fingerprints by using the euclidean distance: 
        np.linalg.norm(np.array(m1) - np.array(m2)) < threshold

    use a set so that it doesn't match the same points several times
    """
    matches = []
    matched_points_1 = set() 
    matched_points_2 = set()
    
    for i, m1 in enumerate(minutiae1):
        for j, m2 in enumerate(minutiae2):
            if i not in matched_points_1 and j not in matched_points_2 and np.linalg.norm(np.array(m1) - np.array(m2)) < threshold:
                matches.append((m1, m2))
                matched_points_1.add(i)
                matched_points_2.add(j)
    
    return len(matches) / max(len(minutiae1), len(minutiae2)) * 100 

def extract_minutiae(image_path):
    """extract minutiae points from a fingerprint image."""
    processor = FingerprintPreprocessor(image_path)
    processor.preprocess()
    processor.calculate_minutiaes()
    return processor.minutiaes



minutiae1 = extract_minutiae(path1)
minutiae2 = extract_minutiae(path2)

""" 
show the minutiae points
"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(minutiae1, cmap='gray')
ax[0].set_title('minutiae afbeelding 1')
ax[0].axis('off')

ax[1].imshow(minutiae2, cmap='gray')
ax[1].set_title('minutiae afbeelding 2')
ax[1].axis('off')


plt.tight_layout()
plt.show()

match_score = match_minutiae(minutiae1, minutiae2)
print(f"Minutiae match percentage: {match_score:.2f}%")
