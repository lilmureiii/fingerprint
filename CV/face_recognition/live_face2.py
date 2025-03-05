import csv
from datetime import datetime
import os
import face_recognition
import cv2
import numpy as np
from tkinter import *
import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
import time



# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

db = "./faces"
encoding_file = "encodings.pkl"

def load_encodings():
    if os.path.exists(encoding_file):
        with open(encoding_file, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
    else:
        known_face_encodings = []
        known_face_names = []
        db = "./faces"
        for image in os.listdir(db): 
            name = os.path.splitext(image)[0].replace("_", " ")
            img = face_recognition.load_image_file(os.path.join(db, image))
            encoding = face_recognition.face_encodings(img)
            if encoding:  # Zorg dat er een encoding is
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)
        
        with open(encoding_file, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names
# Create arrays of known face encodings and their names

csv_file = "attendance.csv"
ctk.set_appearance_mode("system")        


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

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

        self.person_frame = ctk.CTkFrame(frames_container, width=200, height=200)
        self.person_frame.pack(side="left", expand=True, fill="both", padx=10)

        # Aanwezigheids-frame
        self.att_frame = ctk.CTkFrame(frames_container, width=200, height=200)
        self.att_frame.pack(side="right", expand=True, fill="both", padx=10)

        # Aanwezigheidslijst Label
        self.attendance_label = ctk.CTkLabel(self.att_frame, text="Attendance List", font=("Arial", 18, "bold"))
        self.attendance_label.pack(pady=10)

        # Aanwezigheidslijst Tekstvak
        self.attendance_text = ctk.CTkTextbox(self.att_frame, height=10)
        self.attendance_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.load_attendance()

        # Persoon label
        self.nameLabel = ctk.CTkLabel(self.person_frame, text="Click 'Register' to confirm your attendance!", font=("Arial", 18, "bold"))
        self.nameLabel.pack(pady=10)

        # Foto Label
        self.picLabel = ctk.CTkLabel(self.person_frame, text="")
        self.picLabel.pack(pady=10)
        self.pic = None

        # Registratie-knop
        self.button = ctk.CTkButton(self, text="Register", command=self.start_camera)
        self.button.pack(pady=10)
        video_capture.open(0) 

    
    def start_camera(self):
        self.nameLabel.configure(text="Getting ready... Please wait!")
        self.after(1000,self.countdown, 4) 

        

    def countdown(self, seconds):
        if seconds > 0:
            self.nameLabel.configure(text=f"Getting ready... {seconds} seconds left!")
            self.after(1000, self.countdown, seconds-1)  # countdown
        else:
            self.register() # if timer is zero,go to register function to take picture

    def load_attendance(self):
        """Load csv in frame """
        self.attendance_text.delete("1.0", "end")
        try:
            with open("attendance.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    self.attendance_text.insert("end", " | ".join(row) + "\n")
        except FileNotFoundError:
            self.attendance_text.insert("end", "Attendance file not found.")

    def log_attendance(self,name):
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        already_logged = False
        if os.path.exists(csv_file):
            with open(csv_file, "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0] == name and row[1] == date:
                        already_logged = True
                        break

        if not already_logged:
            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([name, date, time, True])
            print(f"{name} is geregistreerd als aanwezig.")


    def register(self): 
        known_face_encodings, known_face_names = load_encodings()

        ret, frame = video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            self.person = "Unknown"
            if face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.4)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                best_match_index = np.argmin(face_distances)

                if matches and matches[best_match_index]:
                    self.person = known_face_names[best_match_index]
                    self.log_attendance(self.person)

        
            pil_image = Image.fromarray(rgb_frame).resize((300, 300), Image.Resampling.LANCZOS)
            self.pic = ImageTk.PhotoImage(pil_image)
            self.picLabel.configure(image=self.pic)
            self.picLabel.image = self.pic
        
            self.nameLabel.configure(text=f"{self.person} registered!" if self.person != "Unknown" else "Not recognized")
            self.load_attendance()


        video_capture.release()
        self.button.configure(self, text="Add registration", command=self.start_camera)


        
    def on_closing(self):
        video_capture.release()
        cv2.destroyAllWindows()
        self.destroy()

app = App()
app.mainloop()
app.protocol("WM_DELETE_WINDOW", app.on_closing)

