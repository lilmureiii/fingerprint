# Fingerprint Recognition Project - Jonas and Marie-Eve
## Installation 

Install the dependencies with `pip install -r requirements.txt`

## Dependencies 

The project relies on the following libraries:
- OpenCV
- NumPy
- Matplotlib
- Scikit-Image
- SciPy
- PyFingerprint

See `requirements.txt` for a full list of dependencies.
<br>

## Preprocessing
The [preprocessing.py](./CV/preprocessing.py) script is responsible for preparing fingerprint images for further analysis. It performs several steps to enhance the quality of the fingerprint patterns and extract key features:

- **Contrast Enhancement**
    <br>Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast.
    <br>Applies median blur to remove noise.

- **Gabor Filtering**
    <br>Applies Gabor filters at 45° and 135° angles to enhance fingerprint ridges.

- **Ridge Detection**
    <br>Converts the enhanced image to a binary format using adaptive thresholding.

- **Skeletonization**
    <br>Thins the fingerprint ridges to a single-pixel width for minutiae extraction.

- **Minutiae Detection**
    <br>Identifies ridge endings and bifurcations, marking them in the image.

- **Singularity Detection** (To be implemented)

## Testing

The [test.py](./CV/test.py) script processes multiple fingerprint images in a folder to visualize their preprocessed (thinned) versions. It follows these steps:

- **Batch Processing**
    <br>Iterates over all fingerprint images in the `vingertop_images` folder.
    <br>Loads each image in grayscale.

- **Preprocessing**
    <br>Uses `FingerprintPreprocessor` to enhance and skeletonize the fingerprints.

- **Visualization**
    <br>Displays all preprocessed (thinned) fingerprints in a single figure.
    <br>Helps verify the preprocessing quality before matching.

## Recognition

The [recognition.py](./CV/recognition.py) script is responsible for matching fingerprint images by extracting and comparing minutiae points. It follows these steps:

- **Minutiae Extraction**
    <br>The script loads fingerprint images and extracts minutiae points using the `FingerprintPreprocessor`.

- **Matching Algorithm**
    <br>The `match_minutiae` function calculates the Euclidean distance between minutiae points of two images.
    <br>A threshold-based approach is used to determine if two fingerprints match.

- **Database Comparison**
    <br>The script compares a given fingerprint against all images in a fingerprint database.
    <br>Matches are sorted by similarity score, and the highest match is identified.
    <br>If no match exceeds the threshold (e.g., 95%), the system states that the fingerprint is not in the database.

- **Visualization (Optional)**
    <br>The script can display the extracted minutiae and thinned images for verification.

## Conclusion project
As we approached the end of the base for the fingerprint recognition project, we came against some issues. We concluded that the resolution of the pictures taken by a phone camera needed to be really high to extract enough ridges for the minutiae extraction. We tried a lot of filtering on the images to enhance the ridges but that didn't improve much. What we also concluded is that using flash whilst taking the picture improved the ridge detection.
