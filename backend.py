from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('dropdown.html')

@app.route('/results', methods=["POST"])
def show_chart():
    ...


if __name__ == '__main__':
    app.run()