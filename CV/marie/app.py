import csv
from datetime import datetime
import os
import face
import cv2
import numpy as np
from tkinter import *
import customtkinter as ctk
from PIL import Image, ImageTk
from preprocessing import FingerprintPreprocessor
import pyodbc
import base64



# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
# -------------------------------SQL-connection---------------------------------
conn_gen = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;'
def get_known_faces(): 
    
    conn = pyodbc.connect(conn_gen)
    cursor = conn.cursor()
    # ophalen van database 
    cursor.execute("SELECT f_name, surname, face_encodings FROM person")
    db_faces = cursor.fetchall()
    cursor.close()
    conn.close()

    # ---------------------------- GEZICHTEN INLADEN ---------------------------------

    known_face_encodings = []
    known_face_names = []

    for row in db_faces:
        name = row[1]
        encoding_str = row[2]  

        # Voeg padding toe aan de base64 string, zodat de lengte een veelvoud van 4 is
        padding = '=' * (4 - len(encoding_str) % 4)
        encoding_str = encoding_str + padding

        try:
        # Decodeer de base64 string naar bytes
            pickled_data_bytes = base64.b64decode(encoding_str)
            float_array = np.frombuffer(pickled_data_bytes, dtype=np.float32)
            known_face_encodings.append(float_array)
            known_face_names.append(name)
        except Exception as e:
            print(f"Error while loading the encoding of {name}: {e}")

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = get_known_faces()


# ---------------------------- APP ---------------------------------
ctk.set_appearance_mode("light")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x500")
        self.title("Attendance System")
        self.configure(bg_color="purple", fg_color="purple")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.navigation = NavigationFrame(self)
        self.navigation.grid(row=0, column=0, sticky="nesw")
        
        self.frame1 = AttendanceFrame(self)
        self.frame2 = RegisterFrame(self)
        self.frame3 = AllAttendanceFrame(self)
        
        self.frame1.grid(row=0, column=1, sticky="nsew")
        self.frame2.grid(row=0, column=1, sticky="nsew")
        self.frame3.grid(row=0, column=1, sticky="nsew")

        self.frame1.tkraise()

    def on_closing(self):
        video_capture.release()
        cv2.destroyAllWindows()
        self.destroy()

class NavigationFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.controller = parent
        self.configure(fg_color="purple")
        
        # ctk.CTkLabel(self, text="Navigation", font=("Helvetica", 20)).grid(row=0, column=0, padx=20, pady=10)
        
        ctk.CTkButton(self, text="Today's attendance", command=self.show_frame1).grid(row=1, column=0, padx=20, pady=10)
        ctk.CTkButton(self, text="All attendances", command=self.show_frame3).grid(row=3, column=0, padx=20, pady=10)
        ctk.CTkButton(self, text="Register", command=self.show_frame2).grid(row=2, column=0, padx=20, pady=10)
    
    def show_frame1(self):
        self.controller.frame1.tkraise()
    
    def show_frame2(self):
        self.controller.frame2.tkraise()
    def show_frame3(self):

        self.controller.frame3.tkraise()

class AllAttendanceFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        ctk.CTkLabel(self, text="All attendance list", font=("Arial", 18, "bold")).pack(pady=10)
        
        self.configure(fg_color="purple")
        self.search_entry = ctk.CTkEntry(self, 
            placeholder_text="Enter the student's name",
            height=40,
            width=250,
            font=("Helvetica", 16),
            corner_radius=10,
            # fg_color="purple",
            # text_color="white"
        )
        self.search_entry.pack(side="top", pady=5)

        self.search_button = ctk.CTkButton(self, text="Search", command=self.load_attendance)
        self.search_button.pack(side="top", pady=5)
        self.attendance_text = ctk.CTkTextbox(self, height=10)
        self.attendance_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.load_attendance()

    def load_attendance(self):
        search_term = self.search_entry.get().strip()
        self.attendance_text.delete("1.0", "end")
        conn = pyodbc.connect(conn_gen)
        cursor = conn.cursor()
        cursor.execute("SELECT f_name, surname, CheckInTime FROM Attendance a JOIN person p ON p.person_id = a.person_id")
        if search_term:
            cursor.execute("""
                SELECT f_name, surname, CheckInTime 
                FROM Attendance a 
                JOIN person p ON p.person_id = a.person_id
                WHERE p.f_name LIKE ? OR p.surname LIKE ?
            """, (f"%{search_term}%", f"%{search_term}%"))
        else:
            cursor.execute("""
                SELECT f_name, surname, CheckInTime 
                FROM Attendance a 
                JOIN person p ON p.person_id = a.person_id
            """)

        rows = cursor.fetchall()

        # Resultaten weergeven
        if rows:
            for row in rows:
                f_name, surname, check_in_time = row
                self.attendance_text.insert("end", f"{f_name} {surname} | {check_in_time}\n")
        else:
            self.attendance_text.insert("end", "No results found.\n")

        conn.close()


class AttendanceFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        ctk.CTkLabel(self, text="Today's attendance list", font=("Arial", 18, "bold")).pack(pady=10)
        self.attendance_text = ctk.CTkTextbox(self, height=10)
        self.attendance_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.load_attendance()
        self.configure(fg_color="purple")
    def load_attendance(self):
        self.attendance_text.delete("1.0", "end")
        today = datetime.now().strftime("%Y-%m-%d")
        conn = pyodbc.connect(conn_gen)
        cursor = conn.cursor()
        cursor.execute("SELECT f_name, surname, CheckInTime FROM Attendance a JOIN person p ON p.person_id = a.person_id WHERE CONVERT(date, CheckInTime) = ?", (today,))
        rows = cursor.fetchall()
        for row in rows:
            f_name, surname, check_in_time = row
            self.attendance_text.insert("end", f"{f_name} {surname} | {check_in_time}\n")
        conn.close()

class RegisterFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.person = "Unknown"
        ctk.CTkLabel(self, text="Register", font=("Arial", 18, "bold")).pack(pady=10)
        self.nameLabel = ctk.CTkLabel(self, text="Click 'Register' to confirm your attendance!", font=("Arial", 18))
        self.nameLabel.pack(pady=10)
        self.picLabel = ctk.CTkLabel(self, text="")
        self.picLabel.pack(pady=10)
        
        ctk.CTkButton(self, text="Register", fg_color="#324dbe", command=self.start_camera).pack(pady=10)
    
    def start_camera(self):
        if not video_capture.isOpened():
            video_capture.open(0)
        self.nameLabel.configure(text="Getting ready... Please wait!")
        self.after(1000, self.countdown, 4)
    
    def countdown(self, seconds):
        if seconds > 0:
            self.nameLabel.configure(text=f"Getting ready... {seconds} seconds left!")
            self.after(1000, self.countdown, seconds-1)
        else:
            pil_image, person = face.register(known_face_encodings, known_face_names)
            self.log_attendance(person)
            self.person = person
            self.pic = ImageTk.PhotoImage(pil_image)
            self.picLabel.configure(image=self.pic)
            self.picLabel.image = self.pic
            ctk.CTkButton.configure(self, text="Add registration",command=self.start_camera)

    def log_attendance(self, name):
        conn = pyodbc.connect(conn_gen)
        cursor = conn.cursor()
        cursor.execute("SELECT person_id FROM person WHERE surname = ?", (name,))
        person = cursor.fetchone()
        if person is None:
            self.nameLabel.configure(text=f"Person {name}")
            cursor.close()
            conn.close()
            return
    
        person_id = person[0]
        today = datetime.now().strftime("%Y-%m-%d")

        # is the student already registered?
        cursor.execute("SELECT COUNT(*) FROM Attendance WHERE person_id = ? AND CONVERT(date, CheckInTime) = ?", (person_id, today))
        already_registered = cursor.fetchone()[0]

        if already_registered > 0:
            self.nameLabel.configure(text=f"{name} is already registered for today.")
            cursor.close()
            conn.close()
            return
        else: 
            self.nameLabel.configure(text=f"{name} registered!")

        # Log de aanwezigheid
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO Attendance (person_id, CheckInTime) VALUES (?, ?)", (person_id, now))
        conn.commit()

        cursor.close()
        conn.close()

        self.master.frame1.load_attendance()




app = App()
app.protocol("WM_DELETE_WINDOW", app.on_closing)
app.mainloop()


