# Fingerprint / Face Recognition Project - Jonas and Marie-Eve
## Installation 

Install the dependencies with `pip install -r requirements.txt`

## Dependencies 

The project relies on the following libraries for the fingerprint recognition:
- OpenCV
- NumPy
- Matplotlib
- Scikit-Image
- SciPy
- PyFingerprint

And the project relies on the following libraries for the face recognition: 
- OpenCV
- face-recognition
- Numpy
- Csv
- Datetime

*Remark*: <br>
If `pip install face_recognition` does not want to work, you have to download Cmake (online) and install Cmake with `pip install cmake`. Download Visual Studio Installer for C++ and inside this app install Visual Studio Build tools for 2022. If this is installes go back to your terminal and install dlib and face_recognition.

See `requirements.txt` for a full list of dependencies.
<br>


<details>

<summary><h2>Fingerprint Recognition</h2></summary>

 ### Preprocessing
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
    <br>Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
    Then the crossing number algorithm will look at 3x3 pixel blocks:

    if middle pixel is black (represents ridge):
    if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
    if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation

- **Singularity Detection** (To be implemented)
    <br> There is some code you can use from <https://github.com/cuevas1208/fingerprint_recognition/blob/master/utils/poincare.py> but you have to slightly change it because we don't use every argument they give with the funtion.

### Testing

The [test.py](./CV/test.py) script processes multiple fingerprint images in a folder to visualize their preprocessed (thinned) versions. It follows these steps:

- **Batch Processing**
    <br>Iterates over all fingerprint images in the `vingertop_images` folder.
    <br>Loads each image in grayscale.

- **Preprocessing**
    <br>Uses `FingerprintPreprocessor` to enhance and skeletonize the fingerprints.

- **Visualization**
    <br>Displays all preprocessed (thinned) fingerprints in a single figure.
    <br>Helps verify the preprocessing quality before matching.

### Recognition

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

### Conclusion project
As we approached the end of the base for the fingerprint recognition project, we came against some issues. We concluded that the resolution of the pictures taken by a phone camera needed to be really high to extract enough ridges for the minutiae extraction. We tried a lot of filtering on the images to enhance the ridges but that didn't improve much. What we also concluded is that using flash whilst taking the picture improved the ridge detection.

</details>


<details>

<summary><h2>Face Recognition - live_face2.py </h2></summary>

- **Webcam**
    <br>First a connection is made with the webcam of the computer to get live footage.
    <br>A rectangle is placed around all faces that the script recognizes.

- **Face recognition**
    <br>The footage is matched with the face encodings from the image database. 
    <br>Every face is checked to find a match and if a match is found the name of the recognised face is shown in the rectangle.
    <br>If there is no match, the rectangle shows "Unknown"

- **Registration**
    <br>If a match is found the system registers the name in a csv file. 
    <br>The csv file contains the name of the person, date and time.
    <br>There is also a check: 
    - if a face is already registered in the system on that day it will not be registered again if it recognizes the face again.
    <br>The logging of the attendance happens in the csv file `attendance.csv`


### Conclusion project

</details>