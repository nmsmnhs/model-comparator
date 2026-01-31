import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ml_model import get_stats

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('dropdown.html')

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["csvFileInput"]

    df = pd.read_csv(file)

    columns = df.columns.tolist()

    # store df temporarily (session / global / cache)
    df.to_csv("temp.csv", index=False)

    return jsonify({
        "columns": columns
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    target_col = request.json["target"]

    df = pd.read_csv("temp.csv")

    result = get_stats(df, target_col)

    return jsonify(result)

@app.route('/results', methods=["POST"])
def show_chart():
    ...


if __name__ == '__main__':
    app.run()