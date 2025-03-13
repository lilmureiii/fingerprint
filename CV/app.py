import csv
from datetime import datetime
import os
import face
import cv2
import numpy as np
from tkinter import *
import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
from preprocessing import FingerprintPreprocessor
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


ctk.set_appearance_mode("system")

# ---------------------------- GEZICHTEN INLADEN ---------------------------------

known_face_encodings = []
known_face_names = []

for row in db_faces:
    face_id = row[0]
    name = row[1]
    encoding_str = row[2]  

    # Voeg padding toe aan de base64 string, zodat de lengte een veelvoud van 4 is
    padding = '=' * (4 - len(encoding_str) % 4)
    encoding_str = encoding_str + padding

    try:
    # Decodeer de base64 string naar bytes
        pickled_data_bytes = base64.b64decode(encoding_str)
        float_array = np.frombuffer(pickled_data_bytes, dtype=np.float32)
        # print(f"Decoded float array: {float_array}")
        known_face_encodings.append(float_array)
        known_face_names.append(name)
    except Exception as e:
        print(f"Fout bij laden van encoding voor {name}: {e}")

cursor.close()
conn.close()

# ---------------------------- APP ---------------------------------

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
    def countdown(self, seconds):
        if seconds > 0:
            self.nameLabel.configure(
                text=f"Getting ready... {seconds} seconds left!")
            self.after(1000, self.countdown, seconds-1)  # countdown
        else:
            pil_image, person = face.register(known_face_encodings, known_face_names) 
            self.log_attendance(person)
            self.person = person
            self.pic = ImageTk.PhotoImage(pil_image)
            self.picLabel.configure(image=self.pic)
            self.picLabel.image = self.pic

            self.nameLabel.configure(
                text=f"{self.person} registered!" if self.person != "Unknown" else "Not recognized")
            self.load_attendance()

            self.button.configure(self, text="Add registration",
                              command=self.start_camera) 

    def load_attendance(self):
        """Load csv in frame"""
        self.attendance_text.delete("1.0", "end")
        
        conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;')
        cursor = conn.cursor()

        # Haal de aanwezigheid op
        cursor.execute("SELECT f_name, surname, CheckInTime FROM Attendance a JOIN person p ON p.person_id = a.person_id")
        rows = cursor.fetchall()
        
        # Loop door elke rij en voeg toe aan de textbox
        for row in rows:
            f_name, surname, check_in_time = row
            self.attendance_text.insert("end", f"{f_name} {surname} | {check_in_time}\n")
        
        conn.close()    

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


    

    def on_closing(self):
        video_capture.release()
        cv2.destroyAllWindows()
        self.destroy()


app = App()
app.mainloop()
app.protocol("WM_DELETE_WINDOW", app.on_closing)
