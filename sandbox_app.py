# Updated sandbox_app.py with minimal edits, circuit breaker logic, and auto-refresh implementation

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import time

def current_milli_time():
    return round(time.time() * 1000)

app = Flask(__name__)

# Basic state tracking
last_request_time = current_milli_time()
request_count = 0
breaker_enabled = False
breaker_reset_time = current_milli_time()

# Configuration for circuit breaker
MAX_REQUESTS = 100
WINDOW_TIME_MS = 60 * 1000  # 1 minute
BREAKER_TIMEOUT_MS = 300 * 1000  # 5 minutes

@app.route("/data", methods=["GET", "POST"])
def handle_data():
    global last_request_time, request_count, breaker_enabled, breaker_reset_time
    
    now = current_milli_time()

    # Auto-refresh state if window time has passed
    if now - last_request_time > WINDOW_TIME_MS:
        request_count = 0
        last_request_time = now
        breaker_enabled = False

    # Breaker enable logic
    if breaker_enabled:
        if now - breaker_reset_time > BREAKER_TIMEOUT_MS:
            breaker_enabled = False
            request_count = 0
            return jsonify({"message": "Circuit breaker reset. System online."}), 200
        return jsonify({"error": "Service is temporarily unavailable. Circuit breaker active."}), 503

    # Increment request count
    if not breaker_enabled:
        request_count += 1

    # Trip breaker if too many requests
    if request_count > MAX_REQUESTS:
        breaker_enabled = True
        breaker_reset_time = now
        return jsonify({"error": "Too many requests. Circuit breaker tripped."}), 429

    # Example response
    return jsonify({"message": "Request successful.", "time": now}), 200

if __name__ == "__main__":
    app.run(debug=True)