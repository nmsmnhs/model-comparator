from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET"])
def fetch():
    ...

@app.route('/results', methods=["POST"])
def show_chart():
    ...


if __name__ == '__main__':
    app.run()