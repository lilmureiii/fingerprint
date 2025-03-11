import pyodbc
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")
import pandas as pd 

user = os.getenv("user")
password = os.getenv("password")


conn = pyodbc.connect(
    f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=HP-MAEVE;DATABASE=attendance;Encrypt=no;Trusted_Connection=yes;'
)
cursor = conn.cursor()

cursor.execute("SELECT surname, photo_path from person")

columns = [column[0] for column in cursor.description]

rows = [tuple(row) for row in cursor.fetchall()]
cursor.close()
conn.close()

df = pd.DataFrame(rows, columns=columns)

import cv2

foto_path = 