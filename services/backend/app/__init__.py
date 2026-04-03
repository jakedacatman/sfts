from flask import Flask, jsonify, request
from datetime import datetime
from tensorflow.keras.models import load_model 
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)
prefix = "models/"

with open(f"{prefix}decision_tree.pkl", "rb") as file:
    model_tree = pickle.load(file)

model_nn = load_model(f"{prefix}classification_mlp.keras")

classes = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

#mapping for one-hot encoding
colors = {
    "Blue: [0,0,0,0,0,0,0],
    "Blue-White: [1,0,0,0,0,0,0],
    "Orange: [0,1,0,0,0,0,0],
    "Orange-Red: [0,0,1,0,0,0,0],
    "Red": [0,0,0,1,0,0,0],
    "White": [0,0,0,0,1,0,0],
    "Yellow": [0,0,0,0,0,1,0],
    "Yellow-White": [0,0,0,0,0,0,1]
}

spectral_classes = {
    "A": [0, 0, 0, 0, 0, 0],
    "B": [1, 0, 0, 0, 0, 0],
    "F": [0, 1, 0, 0, 0, 0],
    "G": [0, 0, 1, 0, 0, 0],
    "K": [0, 0, 0, 1, 0, 0],
    "M": [0, 0, 0, 0, 1, 0],
    "O": [0, 0, 0, 0, 0, 1]
}

@app.route("/api")
def test():
    return jsonify(time = str(datetime.now().time()))

@app.route("/api/identify", methods = ["POST"])
def identify():
    form = request.form
    X = [ form["temp"], form["lum"], form["rad"], form["mag"], form["color"], form["spec"] ]
    return jsonify({ "prediction": classes[model_tree.predict(X)[0]] })
