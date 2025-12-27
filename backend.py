from flask import Flask

app = Flask(__name__)

@app.route('/')
def fetch():
    ...

if __name__ == '__main__':
    app.run()