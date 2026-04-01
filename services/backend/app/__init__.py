from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route("/api")
def test():
    return jsonify(time = str(datetime.now().time()))
