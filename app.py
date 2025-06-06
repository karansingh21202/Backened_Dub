# app.py

import os
import uuid
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename

from processor import voice_dub

# ─── Configuration ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav"}

# Create the uploads folder if it doesn’t exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_and_process():
    """
    1) Check “file” in request.files
    2) Save it under uploads/<uuid>_<origname>.wav
    3) Call voice_dub(...) which will produce uploads/<uuid>_processed.wav
    4) send_file(...) so React can download the result
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .wav files are allowed"}), 400

    # Secure the original filename, append a UUID to avoid collisions
    original_name = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    saved_filename = f"{unique_id}_{original_name}"
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
    file.save(input_path)

    # Determine output path
    output_filename = f"{unique_id}_processed.wav"
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

    try:
        # Call the processor's voice_dub function
        metadata = voice_dub(input_path, output_path)
    except Exception as e:
        # On any failure in your Colab‐logic, return a 500 error
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    # Stream the processed WAV back to the client
    return send_file(
        output_path,
        mimetype="audio/wav",
        as_attachment=True,
        download_name="processed_audio.wav",
    )


if __name__ == "__main__":
    # Flask will listen on 0.0.0.0:5000 by default
    app.run(host="0.0.0.0", port=5000, debug=True)
