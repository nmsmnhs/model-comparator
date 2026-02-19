import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ml_model import get_stats

app = Flask(__name__)
CORS(app)

UPLOAD_PATH = "temp.csv"

@app.route("/")
def home():
    return render_template("index.html")

# STEP 1: Upload CSV & return column names
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["csvFileInput"]

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    df.to_csv(UPLOAD_PATH, index=False)

    return jsonify({
        "columns": df.columns.tolist()
    })

# STEP 2: Run ML
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    target_col = data.get("target")

    if not target_col:
        return jsonify({"error": "No target selected"}), 400

    df = pd.read_csv(UPLOAD_PATH)
    result = get_stats(df, target_col)

    return jsonify(result)

# STEP 3: Results Page
@app.route("/results")
def results():
    return render_template("results.html")

if __name__ == "__main__":
    app.run(debug=True)
