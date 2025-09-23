import sqlite3

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM faces")
cursor.execute("DELETE FROM recognition_log")
conn.commit()
conn.close()

##########################################################################################
# import pyodbc
# # Azure SQL connection
# server = 'shanicer1.database.windows.net'
# database = 'azure_face'
# username = 'shanicer_admin'
# password = '#Shadowsonic2003'
# driver = '{ODBC Driver 18 for SQL Server}'

# conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
# conn = pyodbc.connect(conn_str)
# cursor = conn.cursor()

# cursor.execute("DELETE FROM faces")
# cursor.execute("DELETE FROM recognition_log")

# conn.commit()
# conn.close()