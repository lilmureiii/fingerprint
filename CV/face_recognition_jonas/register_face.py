from database import get_connection  # Importeer de databaseconnectie
import face_recognition
import cv2
import pickle

# Verbind met de database
conn, cursor = get_connection()

# Vraag om gebruikersgegevens
name = input("Voer de naam in van de persoon: ").strip()
email = input("Voer het e-mailadres in (optioneel): ").strip()
phone = input("Voer het telefoonnummer in (optioneel): ").strip()

# Start webcam
print("📸 Openen van de webcam... Druk op SPATIE om een foto te nemen.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Druk op SPATIE om een foto te maken", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):  # Spatiebalk
        break

cap.release()
cv2.destroyAllWindows()

# Verwerk de genomen foto
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
face_encodings = face_recognition.face_encodings(rgb_frame)

if len(face_encodings) == 0:
    print("❌ Geen gezicht gevonden in de afbeelding. Probeer opnieuw.")
    conn.close()
    exit()

face_encoding = face_encodings[0]
encoding_blob = pickle.dumps(face_encoding)  # Converteer naar binair formaat

# Opslaan in de database
cursor.execute("INSERT INTO Faces (name, encoding, email, phone) VALUES (?, ?, ?, ?)",
               (name, encoding_blob, email, phone))
conn.commit()
conn.close()

print(f"✅ Gezichtsgegevens opgeslagen voor {name} in de database!")
