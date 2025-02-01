import os
import magic  # For MIME type detection
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_MIME_TYPES = ["application/dicom", "image/png", "image/jpeg"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_safe_file(file):
    """Check if the uploaded file is of a valid MIME type."""
    mime = magic.Magic(mime=True)
    file_mime = mime.from_buffer(file.read(2048))  # Read first 2KB to detect MIME type
    file.seek(0)  # Reset file pointer after reading

    return file_mime in ALLOWED_MIME_TYPES

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle secure file uploads with MIME type validation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    if not is_safe_file(file):
        return jsonify({"error": "Invalid file type!"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    return jsonify({"message": "File uploaded successfully", "file_path": file_path})

if __name__ == "__main__":
    app.run(debug=True)
