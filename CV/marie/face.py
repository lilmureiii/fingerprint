import face_recognition
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk


video_capture = cv2.VideoCapture(0)

def register(known_encodings, known_names):
        ret, frame = video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            person = "Unknown"

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                print(f"matches: {matches}")
                print(f"faces: {face_encodings}")
                print(f"namen: {known_names}")

                best_match_index = np.argmin(face_distances)

                if matches and matches[best_match_index]:
                    print(f"best index: {known_names[best_match_index]}")
                    person = known_names[best_match_index]

            pil_image = Image.fromarray(rgb_frame).resize(
                (300, 300), Image.Resampling.LANCZOS)
            return pil_image, person

# print(f"Gehoorde gezichten: {known_face_names}")
# print(f"Aantal gezichten: {len(known_face_encodings)}")





# ---------------------------- encodings laden ---------------------------------
# db = "C:\\hogent\\Stage\\CV\\face_recognition\\faces"
# encoding_file = "C:\\hogent\\Stage\\CV\\face_recognition\\encodings.pkl"
# import pickle

# # Laad de encodings uit het .pkl bestand
# with open(encoding_file, 'rb') as f:
#     known_face_encodings, known_face_names = pickle.load(f)

# # Bekijk de geladen encodings
# print("Gehoorde gezichten:", known_face_names)
# print("Gehoorde gezichten:", known_face_encodings[3])
# print(len(known_face_encodings))
# def load_encodings():
#     if os.path.exists(encoding_file):
#         with open(encoding_file, "rb") as f:
#             known_face_encodings, known_face_names = pickle.load(f)
#     else:
#         known_face_encodings = []
#         known_face_names = []
#         for image in os.listdir(db):
#             name = os.path.splitext(image)[0].replace("_", " ")
#             img = face_recognition.load_image_file(os.path.join(db, image))
#             encoding = face_recognition.face_encodings(img)
#             if encoding:  # Zorg dat er een encoding is
#                 known_face_encodings.append(encoding[0])
#                 known_face_names.append(name)

#         with open(encoding_file, "wb") as f:
#             pickle.dump((known_face_encodings, known_face_names), f)

#     return known_face_encodings, known_face_names
# # Create arrays of known face encodings and their names

