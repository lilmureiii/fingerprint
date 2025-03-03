import csv
from datetime import datetime
import os
import face_recognition
import cv2
import numpy as np
from tkinter import *

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

db = "./faces"
known_face_encodings = []
known_face_names = []

for image in os.listdir(db): 
    naam = ""
    name = os.path.splitext(image)[0]
    name = name.split("_")
    for n in name: 
        naam += n + " "
    name = naam

    path  = os.path.join(db, image)
    img = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Create arrays of known face encodings and their names

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

name = "Unknown"

def recognition(): 
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Find all the faces and face encodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

        if len(face_landmarks_list) > 0:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        else:
            face_encodings = []

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # print(f"De naam van de herkende persoon: {name}")
            lbl.configure(text = f"{name}is registered into the system!")
            lbl.grid(column=2, row=0)
            btn.destroy()

            log_attendance(name)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (153, 204, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (20,40), font, 1.0, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
# create root window
root = Tk()

# root window title and dimension
root.title("MWECAU face attendance system")
# Set geometry(widthxheight)
root.geometry('350x200')

# adding a label to the root window


lbl = Label(root, text = "Register Attendance")
lbl.grid()

btn = Button(root, text = "Register" ,
             fg = "black", command=recognition)
btn.grid(column=1, row=0)


# Execute Tkinter
root.mainloop()