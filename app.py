from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Waves App!"

@app.route('/data', methods=['POST'])
def data_handler():
    data = request.json
    processed_data = process_data(data)
    return jsonify(processed_data)

def process_data(data):
    # Example data processing logic
    return {key: value[::-1] if isinstance(value, str) else value for key, value in data.items()}

if __name__ == '__main__':
    app.run(debug=True)