# test_sqli.py
def unsafe_query(user_input):
    import sqlite3
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{user_input}'"  # UNSAFE - SQLi
    cursor.execute(query)
    return cursor.fetchall()
