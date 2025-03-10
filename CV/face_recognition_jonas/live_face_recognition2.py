from database import get_connection  # Importeer de databaseconnectie
import face_recognition
import cv2
import pickle
import numpy as np
import csv
import os
import threading
from datetime import datetime

# Verbind met de database
conn, cursor = get_connection()

# Haal alle gezichtsgegevens uit de database
cursor.execute("SELECT name, encoding FROM Faces")
known_faces = cursor.fetchall()
conn.close()

if not known_faces:
    print("❌ Geen gezichtsgegevens gevonden in de database. Voeg eerst een gezicht toe.")
    exit()

known_face_encodings = []
known_face_names = []
herkende_personen = set()  # Set om dubbele herkenningen te voorkomen

for name, encoding in known_faces:
    known_face_encodings.append(pickle.loads(encoding))
    known_face_names.append(name)

# CSV-bestand voor herkenning bijhouden
CSV_FILE = "herkende_personen.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Naam", "Datum", "Tijd"])

# Webcam-thread class
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FPS, 30)  # Probeer 30 FPS af te dwingen
        self.stream.set(3, 640)  # Verklein de breedte
        self.stream.set(4, 480)  # Verklein de hoogte
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# Start de webcam met threading
video_stream = VideoStream()
print("📹 Live gezichtsherkenning gestart... Druk op 'Q' om te stoppen.")

# Start gezichtsherkenning in een aparte thread
def recognize_faces():
    while True:
        frame = video_stream.read()
        if frame is None:
            continue

        # Converteer frame naar RGB voor face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Gebruik HOG voor snelheid
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Onbekend"

            # Bepaal de beste match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Controleer of de persoon al herkend is om dubbele invoer te voorkomen
                if name not in herkende_personen:
                    herkende_personen.add(name)
                    now = datetime.now()
                    datum = now.strftime("%Y-%m-%d")
                    tijd = now.strftime("%H:%M:%S")

                    # Voeg herkenning toe aan CSV
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, datum, tijd])
                    print(f"✅ {name} herkend en opgeslagen op {datum} om {tijd}")

            # Teken rechthoek en naam
            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Onbekend" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Toon de live video met herkenning
        cv2.imshow("Live Gezichtsherkenning", frame)

        # Stop met 'Q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Start de herkennings-thread
recognition_thread = threading.Thread(target=recognize_faces)
recognition_thread.daemon = True
recognition_thread.start()

# Houd de hoofdthread open totdat 'q' wordt ingedrukt
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_stream.stop()
cv2.destroyAllWindows()

print(f"✅ Proces voltooid. Gegevens opgeslagen in {CSV_FILE}")
