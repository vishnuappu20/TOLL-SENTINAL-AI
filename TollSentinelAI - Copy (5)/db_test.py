import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="toll_sentinel_ai"
)

print("Database connected successfully!")