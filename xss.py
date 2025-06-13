# Simulated Python XSS vulnerability for testing
# ⚠️ Do NOT use this pattern in production

from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    name = request.args.get('name', '')
    # Vulnerable to reflected XSS if unsanitized input is inserted into HTML
    return f"<h1>Hello {name}</h1>"

if __name__ == '__main__':
    app.run(debug=True)
