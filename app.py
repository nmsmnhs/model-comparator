import os
import uuid
import time
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from ml_model import get_stats

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
CORS(app)

UPLOAD_DIR = "uploads"
MAX_AGE_SECONDS = 60 * 60  # delete files older than 1 hour
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_upload_path(session_id):
    return os.path.join(UPLOAD_DIR, f"{session_id}.csv")


def cleanup_old_uploads():
    """Delete upload files older than MAX_AGE_SECONDS."""
    now = time.time()
    for filename in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(path) and (now - os.path.getmtime(path)) > MAX_AGE_SECONDS:
            try:
                os.remove(path)
            except OSError:
                pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    cleanup_old_uploads()  # prune stale files on every new upload

    file = request.files.get("csvFileInput")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    if df.empty:
        return jsonify({"error": "Uploaded CSV is empty"}), 400

    session_id = str(uuid.uuid4())
    df.to_csv(get_upload_path(session_id), index=False)
    session["session_id"] = session_id

    return jsonify({
        "columns": df.columns.tolist(),
        "rows": len(df),
        "session_id": session_id
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    target_col = data.get("target")
    session_id = data.get("session_id") or session.get("session_id")

    if not target_col:
        return jsonify({"error": "No target column selected"}), 400
    if not session_id:
        return jsonify({"error": "No dataset found. Please upload a CSV first."}), 400

    path = get_upload_path(session_id)
    if not os.path.exists(path):
        return jsonify({"error": "Dataset not found. Please re-upload."}), 400

    try:
        df = pd.read_csv(path)
        result = get_stats(df, target_col)
    finally:
        # Delete the file immediately after analysis — no need to keep it
        try:
            os.remove(path)
        except OSError:
            pass

    return jsonify(result)


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    app.run(debug=True)