import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from ml_model import get_stats

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_upload_path(session_id):
    return os.path.join(UPLOAD_DIR, f"{session_id}.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
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
        return jsonify({"error": "No target selected"}), 400

    path = get_upload_path(session_id)
    df = pd.read_csv(path)
    result = get_stats(df, target_col)

    return jsonify(result)

# STEP 3: Results Page
@app.route("/results")
def results():
    return render_template("results.html")

if __name__ == "__main__":
    app.run(debug=True)
