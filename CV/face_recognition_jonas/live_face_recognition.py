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

# Vraag naar de naam van de persoon die gezocht moet worden
zoek_naam = input("👤 Voer de naam in van de persoon die je wilt zoeken: ")

# Haal het gezicht uit de database
cursor.execute("SELECT encoding FROM Faces WHERE name = ?", (zoek_naam,))
result = cursor.fetchone()
conn.close()

if result is None:
    print(f"❌ Geen gezichtsgegevens gevonden voor {zoek_naam}. Voeg eerst een gezicht toe.")
    exit()

known_face_encoding = pickle.loads(result[0])  # Decodeer de gezichtsdata

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
print(f"📹 Op zoek naar {zoek_naam}... Druk op 'Q' om te stoppen.")
gevonden = False

while not gevonden:
    frame = video_stream.read()
    if frame is None:
        continue

    # Converteer frame naar RGB voor face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Gebruik face_recognition voor gezichtsdetectie
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        match = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.6)

        if match[0]:
            gevonden = True
            now = datetime.now()
            datum = now.strftime("%Y-%m-%d")
            tijd = now.strftime("%H:%M:%S")

            # Voeg herkenning toe aan CSV
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([zoek_naam, datum, tijd])
            print(f"✅ {zoek_naam} gevonden en opgeslagen op {datum} om {tijd}")
            break

    # Toon de live video met herkenning
    cv2.imshow("Gezichtsherkenning", frame)

    # Stop met 'Q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_stream.stop()
cv2.destroyAllWindows()

print(f"✅ Proces voltooid. Gegevens opgeslagen in {CSV_FILE}")
