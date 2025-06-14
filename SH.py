# session_hijacking_vuln.py
from flask import Flask, request, make_response

app = Flask(__name__)

# Insecure session storage using cookies without HttpOnly or Secure flags
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == "admin" and password == "admin123":
        resp = make_response(f"Welcome {username}!")
        # ⚠️ Vulnerability: No Secure, HttpOnly, or SameSite attributes
        resp.set_cookie("sessionID", "ABC123")  # Session ID easily accessible via JS or network sniffing
        return resp
    else:
        return "Invalid credentials", 401

@app.route('/dashboard')
def dashboard():
    session_id = request.cookies.get("sessionID")
    if session_id == "ABC123":
        return "Sensitive admin data"
    else:
        return "Access denied", 403

if __name__ == '__main__':
    app.run(debug=True)
