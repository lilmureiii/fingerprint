import csv
from datetime import datetime
import os
import face_recognition
import cv2
import numpy as np

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("./faces/obama.png")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("./faces/biden.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
marie_image = face_recognition.load_image_file("./faces/Marie2.jpg")
marie_face_encoding = face_recognition.face_encodings(marie_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding, 
    marie_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden", 
    "Marie-Eve"
]

csv_file = "attendance.csv"

def log_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Controleer of de naam al is geregistreerd vandaag
    already_logged = False
    if os.path.exists(csv_file):
        with open(csv_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0] == name and row[1] == date:
                    already_logged = True
                    break

    # Voeg de nieuwe aanwezigheid toe als deze nog niet is geregistreerd
    if not already_logged:
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, date, time, True])
        print(f"{name} is geregistreerd als aanwezig.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    print(f"Detected face locations: {face_locations}")  # Debugging

    face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

    if len(face_landmarks_list) > 0:
        print("Successfully detected landmarks.")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    else:
        print("No landmarks found, skipping encoding.")
        face_encodings = []

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # print(f"De naam van de herkende persoon: {name}")
        
        log_attendance(name)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
