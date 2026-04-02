from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

@app.route("/api")
def test():
    return jsonify(time = str(datetime.now().time()))

@app.route("/api/identify", methods = ["POST"])
def identify():
    return jsonify(request.form)
