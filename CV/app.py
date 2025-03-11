import csv
from datetime import datetime
import os
import struct
import face_recognition
import cv2
import numpy as np
from tkinter import *
import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
from preprocessing import FingerprintPreprocessor
import matplotlib.pyplot as plt
import pyodbc
import base64



# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
# -------------------------------SQL-connection---------------------------------


conn = pyodbc.connect(
    f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;'
)
cursor = conn.cursor()
# ophalen van database 
cursor.execute("SELECT f_name, surname, face_encodings FROM person")
db_faces = cursor.fetchall()

def  blub():# ------------------------- fingerprint recognition ----------------------------
    path1 = "./fingerprint_db2/thumb.jpg"
    dir = "./fingerprint_db2"

    # # preprocessor-object aan en verwerk de afbeelding
    # fingerprint = FingerprintPreprocessor(path)
    # thinned, skeleton = fingerprint.preprocess()


    def should_skip_by_ratio(minutiae1, minutiae2, ratio_threshold=1.5):
        """
        Return True if the ratio of the Euclidean distances between two minutiae points 
        exceeds the ratio_threshold.
        """
        for m1 in minutiae1:
            for m2 in minutiae2:
                distance1 = np.linalg.norm(np.array(m1) - np.array(m2))
                for m3 in minutiae1:
                    for m4 in minutiae2:
                        distance2 = np.linalg.norm(np.array(m3) - np.array(m4))
                        if distance1 != 0 and distance2 != 0:
                            ratio = max(distance1, distance2) / \
                                min(distance1, distance2)
                            if ratio > ratio_threshold:
                                return True
        return False


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
                # Bereken de Euclidische afstand tussen de coördinaten (x, y) van m1 en m2
                # Gebruik alleen de x, y coördinaten
                distance = np.linalg.norm(np.array(m1[:2]) - np.array(m2[:2]))
                if i not in matched_points_1 and j not in matched_points_2 and distance < threshold:
                    matches.append((m1, m2))
                    matched_points_1.add(i)
                    matched_points_2.add(j)

        return len(matches) / max(len(minutiae1), len(minutiae2)) * 100


    def extract_minutiae(image_path):
        """extract minutiae points from a fingerprint image."""
        processor = FingerprintPreprocessor(image_path)
        processor.preprocess()
        processor.calculate_minutiaes()
        return processor.minutiaes, processor.thinned

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


csv_file = "C:\\hogent\\Stage\\CV\\face_recognition\\attendance.csv"

ctk.set_appearance_mode("system")

# ---------------------------- app -------------------------------
known_face_encodings = []
known_face_names = []

for row in db_faces:
    face_id = row[0]
    name = row[1]
    encoding_str = row[2]  # Dit is de opgeslagen string in je database

    # Voeg padding toe aan de base64 string, zodat de lengte een veelvoud van 4 is
    padding = '=' * (4 - len(encoding_str) % 4)
    encoding_str = encoding_str + padding
    # print(f"Encoding string: {encoding_str}")
    # print(f"Length after padding: {len(encoding_str)}")


    # Laad het object met pickle (dit zou een numpy-array moeten zijn)
    try:
    # Decodeer de base64 string naar bytes
        pickled_data_bytes = base64.b64decode(encoding_str)
        float_array = np.frombuffer(pickled_data_bytes, dtype=np.float32)
        # print(f"Decoded float array: {float_array}")
        known_face_encodings.append(float_array)
        known_face_names.append(name)
    except Exception as e:
        print(f"Fout bij laden van encoding voor {name}: {e}")

# print(f"Gehoorde gezichten: {known_face_names}")
# print(f"Aantal gezichten: {len(known_face_encodings)}")


# import pyodbc
# import base64
# import struct

# # Verbinding maken met de database
# conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;')
# cursor = conn.cursor()

# # Jouw numerieke array
# encoded_data = [
#     0.00212624, 0.18151706, 0.08942953, -0.02890676, -0.13226941, 0.04053922,
#     -0.03351211, -0.07746178, 0.05412486, -0.05690681, 0.24183123, -0.05402805,
#     -0.25743562, -0.05157364, 0.0613832, 0.12605765, -0.14946279, -0.06552804,
#     -0.20102565, -0.06235639, -0.0110338, -0.04100168, 0.02593716, -0.07272571,
#     -0.20662078, -0.23533882, -0.04712093, -0.10228271, -0.02789488, -0.16243912,
#     0.07824215, -0.03084397, -0.14441717, -0.05164244, -0.04710121, -0.02985836,
#     -0.01722826, -0.0584734, 0.13154452, 0.02626333, -0.17892294, 0.13025913,
#     0.00733513, 0.21587217, 0.29474819, -0.00523588, 0.04717015, -0.08853559,
#     0.13626249, -0.24181871, 0.05437325, 0.03498966, 0.1680707, 0.05769799,
#     0.14244418, -0.07317239, 0.02392226, 0.18972902, -0.20062508, 0.03737673,
#     0.05043882, -0.0548215, 0.03411493, -0.04349241, 0.12301467, 0.11024217,
#     -0.00977706, -0.09036207, 0.19369121, -0.09519792, -0.11277232, 0.0307876,
#     -0.06616236, -0.12772177, -0.3352626, 0.00107875, 0.27036458, 0.13602163,
#     -0.29611105, -0.07562297, -0.04549199, 0.00518061, 0.03369437, 0.00834188,
#     -0.03430062, -0.13692574, -0.05488765, -0.01437552, 0.24373627, -0.10935871,
#     0.0070437, 0.2237342, 0.04802172, -0.14844161, 0.05571188, -0.03593138,
#     -0.11785193, -0.01510649, -0.11769232, -0.05191773, -0.02262584, -0.15878251,
#     -0.03008006, 0.0665702, -0.2624118, 0.16402596, -0.01417359, -0.07986996,
#     -0.05631729, -0.04092158, 0.01244716, 0.06576459, 0.22907104, -0.24998786,
#     0.20107329, 0.2281817, -0.05759402, 0.07731683, 0.01206536, 0.06270903,
#     -0.03904252, 0.03572973, -0.08620855, -0.13285442, -0.00430031, -0.02123464,
#     0.00961275, 0.06677756
# ]
# print(len(encoded_data))

# # Omzetten van de array van floats naar bytes
# byte_data = struct.pack(f'128f', *encoded_data)

# # Base64 encoderen
# base64_encoded = base64.b64encode(byte_data).decode('utf-8')

# # Voeg de encoding toe aan de database voor de gegeven person_id
# cursor.execute("UPDATE person SET face_encodings = ? WHERE f_name != 'Marie-Eve'", (base64_encoded))
# conn.commit()


# Sluit de databaseverbinding
cursor.close()
conn.close()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;'

        self.title("MWECAU face attendance system")

        large_text = ctk.CTkLabel(self,
                                  text="Welcome on the attendance platform",
                                  font=("Arial", 28, "bold"),
                                  )
        large_text.pack(pady=20)

        self.person = "Unknown"

        # Frames-container
        frames_container = ctk.CTkFrame(self)
        frames_container.pack(expand=True, fill="both", padx=10, pady=10)

        self.person_frame = ctk.CTkFrame(
            frames_container, width=200, height=200)
        self.person_frame.pack(side="left", expand=True, fill="both", padx=10)

        # Aanwezigheids-frame
        self.att_frame = ctk.CTkFrame(frames_container, width=200, height=200)
        self.att_frame.pack(side="right", expand=True, fill="both", padx=10)

        # Aanwezigheidslijst Label
        self.attendance_label = ctk.CTkLabel(
            self.att_frame, text="Attendance List", font=("Arial", 18, "bold"))
        self.attendance_label.pack(pady=10)

        # Aanwezigheidslijst Tekstvak
        self.attendance_text = ctk.CTkTextbox(self.att_frame, height=10)
        self.attendance_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.load_attendance()

        # Persoon label
        self.nameLabel = ctk.CTkLabel(
            self.person_frame, text="Click 'Register' to confirm your attendance!", font=("Arial", 18, "bold"))
        self.nameLabel.pack(pady=10)

        # Foto Label
        self.picLabel = ctk.CTkLabel(self.person_frame, text="")
        self.picLabel.pack(pady=10)
        self.pic = None

        # Registratie-knop
        self.button = ctk.CTkButton(
            self, text="Check fingerprint", command=self.test)
        self.button.pack(pady=10)
        video_capture.open(0)

        # Registratie-knop
        self.button = ctk.CTkButton(
            self, text="Register", command=self.start_camera)
        self.button.pack(pady=10)
        video_capture.open(0)

    def start_camera(self):
        if not video_capture.isOpened():
            video_capture.open(0)
        self.nameLabel.configure(text="Getting ready... Please wait!")
        self.after(1000, self.countdown, 4)

    def test(self):
        # Map met opgeslagen vingertopafbeeldingen
        input_folder = 'C:\\hogent\\Stage\\CV\\vingertop_images'

        # Lijst voor de thinned afbeeldingen en bijbehorende bestandsnamen
        thinned_images = []
        filenames = []

        # Verwerk elke afbeelding in de map
        for filename in os.listdir(input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(input_folder, filename)
                # img = unsharp_mask(image, radius=5, amount=2)
                # img = np.uint8(img * 255)
                try:
                    # Initialiseer de preprocessor met de afbeelding
                    preprocessor = FingerprintPreprocessor(image_path)
                    preprocessor.preprocess()

                    # Voeg de thinned afbeelding toe aan de lijst
                    thinned_images.append(preprocessor.thinned)
                    filenames.append(filename)

                except Exception as e:
                    print(f"Fout bij verwerken van {filename}: {e}")

        self.match_fingerprint()
        # Visualiseer alle thinned afbeeldingen
        # fig, axes = plt.subplots(1, len(thinned_images), figsize=(15, 5))
        # for ax, img, filename in zip(axes, thinned_images, filenames):
        #     ax.imshow(img, cmap='gray')
        #     ax.set_title(f"Thinned: {filename}")
        #     ax.axis('off')

        # plt.tight_layout()
        # plt.show()

        # print("Alle thinned afbeeldingen zijn verwerkt en weergegeven!")

    # def match_fingerprint(self):
    #     minutiae1, thinned1 = extract_minutiae(path1)
    #     matches = []
    #     for file in os.listdir(dir):
    #         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             file_path = os.path.join(dir, file)
    #             minutiae2, thinned2 = extract_minutiae(file_path)
    #             match_score = match_minutiae(minutiae1, minutiae2)
    #             matches.append((file, match_score))

    #         # print(f"Minutiae match percentage: {match_score:.2f}%")

    #     print("Alle matches: ", matches)
    #     highest = None
    #     matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)
    #     print("Alle matches gesorteerd: ", matches_sorted)

    #     # Haal de hoogste match op
    #     threshold = 95
    #     highest = matches_sorted[0]
    #     if highest[1] >= threshold:
    #         print("Best match is", highest)
    #     else:
    #         print("This person has no fingerprints in the database and therefore can't be identified in the system")

    def countdown(self, seconds):
        if seconds > 0:
            self.nameLabel.configure(
                text=f"Getting ready... {seconds} seconds left!")
            self.after(1000, self.countdown, seconds-1)  # countdown
        else:
            self.register()  # if timer is zero,go to register function to take picture

    def load_attendance(self):
        """Load csv in frame """
        self.attendance_text.delete("1.0", "end")
        try:
            with open("C:\\hogent\\Stage\\CV\\face_recognition\\attendance.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    self.attendance_text.insert("end", " | ".join(row) + "\n")
        except FileNotFoundError:
            self.attendance_text.insert("end", "Attendance file not found.")
        video_capture.release()
        return
        # lijst label
        # self.attLabel = ctk.CTkLabel(
        #     self.person_frame, text="Click 'Register' to confirm your attendance!", font=("Arial", 18, "bold"))
        # self.attLabel.pack(pady=10)

    def log_attendance(self, name):
        conn = pyodbc.connect(self.conn_str)
        cursor = conn.cursor()

        # Haal person_id op uit de person-tabel
        cursor.execute("SELECT person_id FROM person WHERE surname = ?", (name,))
        person = cursor.fetchone()

        if person is None:
            print(f"Persoon '{name}' niet gevonden in database.")
            cursor.close()
            conn.close()
            return
        
        person_id = person[0]  # Haal de ID op
        
        # Haal huidige datum en tijd op
        now = datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
    
        # Insert into Attendance
        cursor.execute("INSERT INTO Attendance (person_id, CheckInTime) VALUES (?, ?)", 
                       (person_id, time))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Attendance gelogd voor {name} op {time}")


    def register(self):
        ret, frame = video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            self.person = "Unknown"

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                print(f"matches: {matches}")
                print(f"faces: {face_encodings}")
                print(f"namen: {known_face_names}")

                best_match_index = np.argmin(face_distances)

                if matches and matches[best_match_index]:
                    print(f"best index: {known_face_names[best_match_index]}")
                    self.person = known_face_names[best_match_index]
                    self.log_attendance(self.person)

            pil_image = Image.fromarray(rgb_frame).resize(
                (300, 300), Image.Resampling.LANCZOS)
            self.pic = ImageTk.PhotoImage(pil_image)
            self.picLabel.configure(image=self.pic)
            self.picLabel.image = self.pic

            self.nameLabel.configure(
                text=f"{self.person} registered!" if self.person != "Unknown" else "Not recognized")
            self.load_attendance()

        self.button.configure(self, text="Add registration",
                              command=self.start_camera)

    def on_closing(self):
        video_capture.release()
        cv2.destroyAllWindows()
        self.destroy()


app = App()
app.mainloop()
app.protocol("WM_DELETE_WINDOW", app.on_closing)
