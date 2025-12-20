import os
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Waves App!"

@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return data if data else "No Data Received"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)