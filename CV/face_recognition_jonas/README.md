# Face Recognition with OpenCV and Face_Recognition

This project uses `OpenCV` and `face_recognition` to perform facial recognition through a live webcam feed. It can search for a person based on a stored face and log the recognition in a CSV file.

## Features

1. **Store Faces**: A user can take a photo and save it as a recognized face.
2. **Search for a Person**: The system looks for a specified person through the webcam.
3. **Real-time Face Recognition**: When a known face is detected, it is recorded with the date and time.

## Installation

Make sure you have Python installed (Python 3.x recommended) and install the required packages with:
`pip install requirements.txt`

## Usage

### 1. Saving a new face

Run the script to save a face:

`python register_face.py`

- Enter a name of the person
- Press the spacebar to capture a photo
- The face will be stored as a .pkl file.

### 2. Searching for a Person

Run the script for face recognition

`python live_face_recognition.py`

- Enter the name of the person you are looking for.
- The webcam will start, and the system will search for the specified person.
- When the face recognized, it is logged recognized_persons.csv

## Project Structure

```

/project-folder
│── faces/                  # Contains stored faces in .pkl format
│── recognize_person.py     # Script to recognize a person
│── save_face.py            # Script to save a face
│── recognized_persons.csv  # CSV file with recognized persons
│── README.md               # This guide
```
