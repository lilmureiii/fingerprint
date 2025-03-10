import pyodbc

def get_connection():
    """Maak een connectie met SQL Server en retourneer de verbinding en cursor."""
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=LAPTOP-JANNES;"
        "DATABASE=FaceRecognitionDB;"
        "Trusted_Connection=yes;"
    )
    return conn, conn.cursor()
