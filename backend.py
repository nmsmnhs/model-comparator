from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('dropdown.html')

@app.route('/load_csv', methods=["POST"])
def get_csv():
    if request.method == "POST":
        f = request.files['csvFileInput']
        if f.filename == '':
            return f"No selected file"
        else:
            return f"{f} loaded"

@app.route('/results', methods=["POST"])
def show_chart():
    ...


if __name__ == '__main__':
    app.run()