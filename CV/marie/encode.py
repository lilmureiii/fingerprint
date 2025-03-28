import numpy as np
import pyodbc
import base64
import cv2
import face_recognition
import struct 

# Verbinding maken met de database
conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;')
cursor = conn.cursor()


cursor.execute("SELECT person_id, photo_path FROM person WHERE photo_path LIKE 'data%'")
students  = cursor.fetchall()

for student in students:
    person_id = student[0]  
    photo_path = student[1]  

    try:
        base64_encoded = photo_path.split(',')[1]

        missing_padding = len(base64_encoded) % 4
        if missing_padding != 0:
            base64_encoded += '=' * (4 - missing_padding)

        image_data = base64.b64decode(base64_encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        face_locations = face_recognition.face_locations(img, model="hog")
        face_encodings = face_recognition.face_encodings(img, face_locations)

        if face_encodings:
            byte_data = struct.pack(f'{len(face_encodings[0])}f', *face_encodings[0])

            base64_face_encoding = base64.b64encode(byte_data).decode('utf-8')

            cursor.execute("UPDATE person SET face_encodings = ? WHERE person_id = ?", (base64_face_encoding, person_id))
            conn.commit()

            print(f"Face encoding toegevoegd voor student met ID: {person_id}")
        else:
            print(f"Geen gezicht gevonden voor student met ID: {person_id}")
    
    except Exception as e:
        print(f"Fout bij student met ID {person_id}: {e}")

conn.close()