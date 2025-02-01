from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Enable CORS for all domains (adjust if needed)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize rate limiter (5 requests/min per IP)
limiter = Limiter(
    get_remote_address,  # Identifies user by IP
    app=app,
    default_limits=["5 per minute"]
)

# Directory for static files (CSS, JS, Images)
STATIC_FOLDER = "static"

# Serve static files (CSS, JS, Images)
@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files like CSS, JavaScript, and images."""
    return send_from_directory(STATIC_FOLDER, filename)

# Homepage route
@app.route("/")
def homepage():
    """Render the homepage with the CT scan upload UI."""
    return render_template("index.html")

# Example protected API route with rate limiting
@app.route("/api/example", methods=["GET"])
@limiter.limit("5 per minute")  # Apply rate limit per IP
def example_api():
    """Example API route with rate limiting."""
    return jsonify({"message": "This is a rate-limited API endpoint."})

# Error handler for rate limit exceedance
@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle too many requests (HTTP 429)."""
    return jsonify({"error": "Too many requests. Please try again later."}), 429

if __name__ == "__main__":
    app.run(debug=True)
