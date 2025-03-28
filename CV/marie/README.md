# Fingerprint / Face Recognition Project: the application - Marie-Eve
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
- CustomTkinter
- PIL
- PyODBC
- Base64

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

<summary><h2>Face Recognition</h2></summary>
<summary><h3>The application</h3></summary>
For the project I built an interface primary for the face recognition, but also included the fingerprint detection to register attendance. The application is built in python, combining OpenCV, Tkinter (CustomTkinter) and an SQL server database.

**Important features**
- Face recognition: loads saved facial information from the database en compares is with the live webcam image.  
- Fingerprint recognition: there is already a button for this in the interface but no code connected to it. For the full fingerprint recognition see [fingerprint.py](./fingerprint.py).
- Database integration: uses an SQL Server connection to save the attendance of students. For the moment this is only a local database. But with [encode.py](./encode.py) all faces can be encoded in the school server where the real database is.
- Graphic user interface: to register attendances and look at all attendances.
    - Made with different frames. One for the attendance list for the day, one to look up all attendances and one to register a person for that day. 
    - If a person is already registered for that day the application will tell that the person was already registered.

**How it works**
1. The application connects to an SQL database and retrieves known face encodings.
    - If there are no face encodings yet, run the encode script to encode all faces for the recognition.

2. A live webcam feed is started for face registration.

3. Upon successful recognition, presence is automatically stored in the database.

4. Users can manually view the attendance list via the UI.

5. The application provides navigation between different sections:
    - Current presence
    - All registered presences
    - Facial registration
    - (Future) fingerprint registration

### Conclusion project
The face recognition works much better than the fingerprint recognition. Even with a poor webcam quality the application succeeds to extract important features of a face and then compare it with other faces. For the moment you can only use the python script locally and not on the server since we do not have the rights to install dependencies and modules on the server. So if this application would be used in the future on the server database, only a column for the face encodings has to be added to the database server and then the [encode.py](./encode.py) has to be executed. After this the application should be usable in the server environment. 

</details>