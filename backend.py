import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ml_model import get_stats

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('dropdown.html')

@app.route('/load_csv', methods=["POST"])
def get_csv():
    if request.method == "POST":
        f = request.files['csvFileInput']
        target = request.form["target"]
        if f.filename == '':
            return f"No selected file"
        else:
            df = pd.read_csv(f)
            data = get_stats(df, target)
            return jsonify(data)

@app.route('/results', methods=["POST"])
def show_chart():
    ...


if __name__ == '__main__':
    app.run()