import os
import matplotlib.pyplot as plt
from preprocessing import FingerprintPreprocessor
from skimage.filters import unsharp_mask
import cv2
import numpy as np

# Map met opgeslagen vingertopafbeeldingen
input_folder = "vingertop_images"

# Lijst voor de thinned afbeeldingen en bijbehorende bestandsnamen
thinned_images = []
filenames = []

# Verwerk elke afbeelding in de map
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        print(f"Verwerken van: {filename}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # img = unsharp_mask(image, radius=5, amount=2)
        # img = np.uint8(img * 255) 
        try:
            # Initialiseer de preprocessor met de afbeelding
            preprocessor = FingerprintPreprocessor(image)
            preprocessor.preprocess()
            
            # Voeg de thinned afbeelding toe aan de lijst
            thinned_images.append(preprocessor.thinned)
            filenames.append(filename)

        except Exception as e:
            print(f"Fout bij verwerken van {filename}: {e}")

# Visualiseer alle thinned afbeeldingen
fig, axes = plt.subplots(1, len(thinned_images), figsize=(15, 5))
for ax, img, filename in zip(axes, thinned_images, filenames):
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Thinned: {filename}")
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Alle thinned afbeeldingen zijn verwerkt en weergegeven!")
