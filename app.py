import os
from flask import Flask, render_template, request
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template("index.html")

# Route for processing user data
@app.route('/process', methods=['POST'])
def process():
    user_data = request.form["data"]
    result = "Processed: " + user_data
    return result

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)