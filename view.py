import sqlite3, pickle

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

print("📌 Faces Table:")
for row in cursor.execute("SELECT id, name FROM faces"):
    print(row)

print("\n📌 Recognition Log:")
for row in cursor.execute("""
    SELECT r.id, f.name, r.timestamp, r.similarity
    FROM recognition_log r
    JOIN faces f ON r.person_id = f.id
    ORDER BY r.timestamp DESC
"""):
    print(row)

conn.close()
